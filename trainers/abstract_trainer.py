import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import copy
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from algorithms.fl_algorithms import get_fl_algorithm_class

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.fed_type = args.fed_type #type of federation ({fl method} or no FL)
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.run_description = args.run_description if args.run_description is not None else args.da_method

        self.experiment_description = args.dataset


        # paths
        self.home_path = os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))


        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)        


    def sweep(self):
        # sweep configurations
        pass
    
    def train_model_no_fl_mergedtargets(self):
    
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        
        # Initilaize the time series learning algorithm
        self.server_algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.server_algorithm.to(self.device)
        
        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug("=" * 45)
        
        ##do model pretraining on server side
        self.server_algorithm.source_train(self.src_train_dl, self.pre_loss_avg_meters, self.logger, pretraining_stage=True)
        
        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage..........')
        self.logger.debug("=" * 45)

        lm, bm = self.server_algorithm.target_train_no_fl(self.trg_train_dls, self.loss_avg_meters, self.logger) 

        # Save nets
        encoder, classifier, tov = self.server_algorithm.get_model_params()
        nets_to_save = {}
        nets_to_save['global_encoder'] = encoder
        nets_to_save['global_classifier'] = classifier
        nets_to_save['global_tov'] = tov
        
        return  nets_to_save
        
    def train_model_no_fl_multitargets(self):
    
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        self.num_clients = len(self.trg_train_dls)
        
        # Initilaize the time series learning algorithm
        self.server_algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.server_algorithm.to(self.device)
        
        self.client_algorithms = []
        for i in range(self.num_clients):
            ca = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
            ca.to(self.device)
            self.client_algorithms.append(ca)
        
        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug("=" * 45)
        
        ##do model pretraining on server side
        self.server_algorithm.source_train(self.src_train_dl, self.pre_loss_avg_meters, self.logger, pretraining_stage=True)
        
        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage..........')
        self.logger.debug("=" * 45)
        
        #### START: Single-round target domain adaptation across multiple clients (no FL) after full pretraining #######
        
        ##Step 1: Pass a copy of the server model to each client. 
        encoder, classifier, tov = self.server_algorithm.get_model_params() #get from server
        for i in range(self.num_clients):
            self.client_algorithms[i].set_model_params({'E':encoder,'C':classifier,'T':tov}) #send to clients
        
        ##Step 2: Do model updates on client sides
        for i in range(self.num_clients): #iterate over clients
        
            self.logger.debug(f'Training Client {i}..........')
            self.logger.debug("=" * 45)
            
            lm, bm = self.client_algorithms[i].target_train_no_fl(self.trg_train_dls[i], self.loss_avg_meters, self.logger) 

        # Save nets
        encoder, classifier, tov = self.server_algorithm.get_model_params()
        nets_to_save = {}
        nets_to_save['global_encoder'] = encoder
        nets_to_save['global_classifier'] = classifier
        nets_to_save['global_tov'] = tov
        for i in range(self.num_clients):
            client_encoder, _, _ = self.client_algorithms[i].get_model_params()
            nets_to_save[f'client_{i}_encoder'] = client_encoder
        
        return nets_to_save
        
    def train_model_fl(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        fl_algorithm_class = get_fl_algorithm_class(self.fed_type)
        self.num_clients = len(self.trg_train_dls)
        
        # Initilaize the time series learning algorithm
        self.server_algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.server_algorithm.to(self.device)
        
        self.client_algorithms = []
        for i in range(self.num_clients):
            ca = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
            ca.to(self.device)
            self.client_algorithms.append(ca)
        
        # Initialize the FL algorithm
        self.fl_algorithm = fl_algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
          
        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug("=" * 45)
        
        ##do model pretraining on server side
        self.server_algorithm.source_train(self.src_train_dl, self.pre_loss_avg_meters, self.logger, pretraining_stage=True)
        
        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage..........')
        self.logger.debug("=" * 45)
        
        #### START: FL after full pretraining #######
        fl_payload = {}

        if self.fed_type == 'SCAFFOLD':
            c_global_para, _, _ = self.server_algorithm.get_model_params() #variance tracking global net for scaffold
            for i in range(self.num_clients):
                self.client_algorithms[i].set_scaffold_items(c_global_para)
            fl_payload['c_global_para'] = c_global_para
            
                
        for fl_round in range(1, self.hparams["num_fl_rounds"] + 1):
            
            self.logger.debug(f'FL Round {fl_round}..........')
            self.logger.debug("=" * 45)
            
            ##Step 1: Pass a copy of the server model to each client. 
            encoder, classifier, tov = self.server_algorithm.get_model_params() #get from server
            for i in range(self.num_clients):
                self.client_algorithms[i].set_model_params({'E':encoder,'C':classifier,'T':tov}) #send to clients
                
            ##Step 2: Do model updates on client sides, here last and best models only contain encoder + classifier
            self.client_encoders = []
            for i in range(self.num_clients): #iterate over clients
            
                self.logger.debug(f'Training Client {i}..........')
                self.logger.debug("=" * 45)
                
                if self.fed_type == 'SCAFFOLD':
                    c_global_para = fl_payload['c_global_para'] #this is updated after each FL round using the FL algorithm (called below)
                    lm, bm, c_delta_para = self.client_algorithms[i].target_train_fl(self.trg_train_dls[i], self.fed_type, 
                                                              self.loss_avg_meters, self.logger, scaffold_c_global=c_global_para) 
                    if 'c_delta_para ' not in fl_payload:
                        fl_payload['c_delta_para'] = [c_delta_para]
                    else:
                        fl_payload['c_delta_para'].append(c_delta_para)
                else:
                    lm, bm = self.client_algorithms[i].target_train_fl(self.trg_train_dls[i], self.fed_type, self.loss_avg_meters, self.logger) 
                
                
                #get client encoders (feature extractors) that need to be aggregated
                client_enc, _, _ = self.client_algorithms[i].get_model_params()
                self.client_encoders.append(client_enc) 
                
                
            #Step3: FL: Encoder Aggregation
            fl_payload['client_encoders'] = self.client_encoders
            fl_payload = self.fl_algorithm.aggregate(fl_payload) #aggregated encoder block
            encoder = fl_payload['aggregated_encoder']
            self.server_algorithm.set_model_params({'E':encoder}) #Note:server separately preserved a copy of the pre-trained model(see algorithms.py)
            
            if 0:
              #Step4: Server-side update/fine-tuning
              '''
              Note: I am diabling this step due to two reasons:
              1: Our objective is not to fine-tune of source data. It is to learn a target encoder that generalizes across all FL clients.
              2: Fine-tuning breaks the spirit of domain adaptation because the global model will keep oscillating between source-optimal and 
                 target-optimal points. The imputer already serves as a reference of source data distribution. So, we don't have to fine-tune it again. 
              '''
              self.server_algorithm.source_train(self.src_train_dl, self.pre_loss_avg_meters, self.logger)
        

        
        # Finish FL. Get global nets to save. Share them to the clients so that each client has the latest nets
        encoder, classifier, tov = self.server_algorithm.get_model_params()
        if self.fed_type not in ['MOON']:
            for i in range(self.num_clients):
                self.client_algorithms[i].set_model_params({'E':encoder,'C':classifier,'T':tov})
                
        # Save nets
        nets_to_save = {}
        nets_to_save['global_encoder'] = encoder
        nets_to_save['global_classifier'] = classifier
        nets_to_save['global_tov'] = tov
        for i in range(self.num_clients):
            client_encoder, _, _ = self.client_algorithms[i].get_model_params()
            nets_to_save[f'client_{i}_encoder'] = client_encoder
        
        return  nets_to_save
    
    def evaluate(self, test_loader, client_id=None):
        
        if client_id is None:
          encoder = self.server_algorithm.encoder.to(self.device)
          classifier = self.server_algorithm.classifier.to(self.device)
        else:
          encoder = self.client_algorithms[client_id].encoder.to(self.device)
          classifier = self.client_algorithms[client_id].classifier.to(self.device)

        encoder.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels,_ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features, seq_features = encoder(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))
    
    def calculate_metrics(self):
        
        accs = []
        f1s = []
        aurocs = []
        
        self.evaluate(self.trg_test_dl)
        
        # accuracy  
        accs.append(self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item())
        # f1
        f1s.append(self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item())
        # auroc 
        aurocs.append(self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item())
        
        acc = torch.tensor(accs).mean()
        f1 = torch.tensor(f1s).mean()
        auroc = torch.tensor(aurocs).mean()
        
        return acc, f1, auroc
        
    def calculate_metrics_across_clients(self):
        
        accs = []
        f1s = []
        aurocs = []
        
        for i in range(self.num_clients):
        
          self.evaluate(self.trg_test_dl, client_id=i)
          
          # accuracy  
          accs.append(self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item())
          # f1
          f1s.append(self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item())
          # auroc 
          aurocs.append(self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item())
        
        acc = torch.tensor(accs).mean()
        f1 = torch.tensor(f1s).mean()
        auroc = torch.tensor(aurocs).mean()
        
        return acc, f1, auroc
        
    def models_are_equal(model1, model2):
        # Load state dicts
        state_dict1 = model1
        state_dict2 = model2
    
        # Check if the models have the same keys and if corresponding values are the same
        for key in state_dict1:
            if key not in state_dict2:
                return False
            if not torch.equal(state_dict1[key], state_dict2[key]):
                return False
    
        # Ensure no extra parameters in model2
        for key in state_dict2:
            if key not in state_dict1:
                return False
    
        return True
    
    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_ids, trg_ids):
        
        #TODO: should the source data also be obtained as a merger of multiple domains?
        self.src_train_dl = data_generator(self.data_path, src_ids, self.dataset_configs, self.hparams, "train", merge_data=True)
        self.src_test_dl = data_generator(self.data_path, src_ids, self.dataset_configs, self.hparams, "test", merge_data=True)

        #trg_ids contain the ids of the target clients
        if self.fed_type == 'noFL_mergedtargets':
          self.trg_train_dls = data_generator(self.data_path, trg_ids, self.dataset_configs, self.hparams, "train", merge_data=True)
        else:
          self.trg_train_dls = [data_generator(self.data_path, trg_ids[i], self.dataset_configs, self.hparams, "train") for i in range(len(trg_ids))]
        
        #merge all client test data so that we can test the local models on entire test data
        self.trg_test_dl = data_generator(self.data_path, trg_ids, self.dataset_configs, self.hparams, "test", merge_data=True) 
        
    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))
    
    def save_checkpoint(self, home_path, log_dir, nets_to_save):
        save_dict = nets_to_save
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)


    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table 