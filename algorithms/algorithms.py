import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import classifier, Temporal_Imputer, masking
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import sys


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError
        
    def models_are_equal(self, model1, model2):
        state_dict1 = model1
        state_dict2 = model2
    
        # Check if both models have the same set of keys and if each parameter is the same
        for key in state_dict1:
            if key not in state_dict2:
                return False
            if not torch.equal(state_dict1[key], state_dict2[key]):
                return False
    
        # Check for any extra keys in the second model
        for key in state_dict2:
            if key not in state_dict1:
                return False
    
        return True


class MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU, self).__init__(configs)

        self.encoder = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.encoder, self.classifier)
        
        # used for FL
        self.global_enc = backbone(configs)
        self.prev_enc = backbone(configs)
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        
    def source_train(self, src_dataloader, avg_meter, logger, pretraining_stage=False):
        
        '''if not pretraining_stage:
            # freeze encoder because it is FL aggregated network
            for k, v in self.encoder.named_parameters():
                v.requires_grad = False'''
        
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for key, val in avg_meter.items():
                avg_meter[key].reset()
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.encoder(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.encoder(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step() #updates using src_cls_loss associated with feat. extractor and classifier
                self.tov_optimizer.step() #updates using tov_loss associated with feat. extractor and termporal verifier

                losses = {'cls_loss': src_cls_loss.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        
        if pretraining_stage:
          self.pre_trained_models = [deepcopy(self.network.state_dict()), deepcopy(self.temporal_verifier.state_dict())]
        
        return src_only_model

    def target_train_fl(self, trg_dataloader, fl_method, avg_meter, logger, scaffold_c_global=None):

        # defining best and last model - self.network is the pretrained sequential module containing the feature extractor and classifier
        best_src_risk = float('inf')
        best_model = self.network.state_dict() #holds the best model found during latest training epochs
        last_model = self.network.state_dict() #holds the outcome of latest training epochs (acts like a pointer to self.network)

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        
        if fl_method == 'FedProx':
            global_model_para = list(self.global_enc.parameters())
        
        elif fl_method == 'SCAFFOLD':
            c_global_para = deepcopy(scaffold_c_global)
            c_local_para = deepcopy(self.c_local)
            global_model_para = deepcopy(self.global_enc.state_dict())
        
        elif fl_method == 'MOON':
            moon_criterion = nn.CrossEntropyLoss().to(self.device)
            cos= nn.CosineSimilarity(dim=-1)
        
        cnt = 0
        
        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs_per_fl_round"] + 1):
            for key, val in avg_meter.items():
                avg_meter[key].reset()
            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq = self.encoder(trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.encoder(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss
                
                if fl_method == 'FedProx':
                    # add FedProx regularization
                    fed_prox_reg = 0.0
                    for param_idx, param in enumerate(self.encoder.parameters()):
                        fed_prox_reg += ((self.hparams['fedprox_mu'] / 2) * torch.norm((param - global_model_para[param_idx]))**2)
                    #print(fed_prox_reg)
                    loss += fed_prox_reg
                    
                elif fl_method == 'MOON':
                    #Get class prediction probabilities using global encoder
                    glb_feat, _ = self.global_enc(trg_x)
                    glb_prob = nn.Softmax(dim=1)(self.classifier(glb_feat))
                    
                    #Get class prediction probabilities using previous encoder
                    prev_feat, _ = self.prev_enc(trg_x)
                    prev_prob = nn.Softmax(dim=1)(self.classifier(prev_feat))
                    
                    #Calculate positive and negative similarities
                    posi = cos(trg_prob, glb_prob)
                    logits = posi.reshape(-1,1)
                    nega = cos(trg_prob, prev_prob)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                    
                    logits /= self.hparams['moon_temperature']
                    pseudo_labels = torch.zeros(trg_x.size(0)).to(self.device).long() #all zeros
                    
                    #moon_loss will only propogate to self.encoder because classifier, global_enc, and prev_enc are frozen
                    moon_loss = self.hparams['moon_mu'] * moon_criterion(logits, pseudo_labels) 
                    loss += moon_loss
                    
                    
                loss.backward()
                self.optimizer.step() #updates feat. extractor through classification loss
                self.tov_optimizer.step() #updates feat. extractor through temporal loss
                
                if fl_method == 'SCAFFOLD':
                    net_para = self.encoder.state_dict()
                    for key in net_para:
                        net_para[key] = net_para[key] - self.hparams['learning_rate'] * (c_global_para[key] - c_local_para[key])
                    self.encoder.load_state_dict(net_para)
                
                cnt += 1
                
                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)
            
            self.lr_scheduler.step()
            
            if fl_method == 'SCAFFOLD':
                c_delta_para = deepcopy(self.c_local)
                net_para = self.encoder.state_dict()
                for key in net_para:
                    self.c_local[key] = self.c_local[key]-c_global_para[key]+(global_model_para[key]-net_para[key]) / (cnt*self.hparams['learning_rate'])
                    c_delta_para[key] = self.c_local[key] - c_local_para[key]
    
            # saving the best model based on src risk
            '''if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())'''

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
            
        if fl_method == 'SCAFFOLD':
            return last_model, best_model, c_delta_para

        return last_model, best_model
    
    def target_train_no_fl(self, trg_dataloader, avg_meter, logger):

        # defining best and last model - self.network is the pretrained sequential module containing the feature extractor and classifier
        best_src_risk = float('inf')
        best_model = self.network.state_dict() #holds the best model found during latest training epochs
        last_model = self.network.state_dict() #holds the outcome of latest training epochs (acts like a pointer to self.network)

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        
        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for key, val in avg_meter.items():
                avg_meter[key].reset()
            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq = self.encoder(trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.encoder(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss
                    
                loss.backward()
                self.optimizer.step() #updates feat. extractor through classification loss
                self.tov_optimizer.step() #updates feat. extractor through temporal loss
               
                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()
            
            # saving the best model based on src risk
            '''if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())'''

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model
    
    def get_models(self):
        return deepcopy(self.encoder.state_dict()), deepcopy(self.classifier.state_dict()), deepcopy(self.temporal_verifier.state_dict())
    
    def set_models(self, in_models):
        
        if 'E' in in_models:
          new_encoder = in_models['E']
          self.prev_enc.load_state_dict(self.encoder.state_dict()) #save previous local encoder before updating it in latest FL round     
          self.encoder.load_state_dict(new_encoder)     
          self.global_enc.load_state_dict(new_encoder) #keep a copy of global encoder (FL)     
          
          for k, v in self.global_enc.named_parameters():
            v.requires_grad = False
          for k, v in self.prev_enc.named_parameters():
            v.requires_grad = False
          
        if 'C' in in_models:
          self.classifier.load_state_dict(in_models['C'])     
          
        if 'T' in in_models:
          self.temporal_verifier.load_state_dict(in_models['T'])
        
        #self.network = nn.Sequential(self.encoder, self.classifier)
        
    def set_scaffold_items(self, c_local):
        self.c_local = deepcopy(c_local)




