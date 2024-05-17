# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:56:41 2024

@author: shubhamp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import classifier, Temporal_Imputer, masking
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy


def get_fl_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class FedAvg():
    def __init__(self, backbone, configs, hparams, device):
        pass
    
    def aggregate(self, fl_payload):
        
        client_nets = fl_payload['client_encoders']
        
        new_net = deepcopy(client_nets[0])
        
        w_sum = {}
        w_count = {}
        
        for cnet in client_nets:
          for k, v in cnet.items():
              if k in w_sum:
                  w_sum[k] += v.clone()  # Use clone to ensure no in-place operation
                  w_count[k] += 1
              else:
                  w_sum[k] = v.clone()  # Same as above, make a copy of v
                  w_count[k] = 1

        for k in w_sum:
            new_net[k] = w_sum[k] / w_count[k]
        
        fl_payload['global_encoder'] = new_net
        
        return fl_payload

class FedProx():
    def __init__(self, backbone, configs, hparams, device):
        pass
    
    def aggregate(self, fl_payload):
        
        client_nets = fl_payload['client_encoders']
        
        new_net = deepcopy(client_nets[0])
        
        w_sum = {}
        w_count = {}
        
        for cnet in client_nets:
          for k, v in cnet.items():
              if k in w_sum:
                  w_sum[k] += v.clone()  # Use clone to ensure no in-place operation
                  w_count[k] += 1
              else:
                  w_sum[k] = v.clone()  # Same as above, make a copy of v
                  w_count[k] = 1

        for k in w_sum:
            new_net[k] = w_sum[k] / w_count[k]
        
        fl_payload['global_encoder'] = new_net
        
        return fl_payload
    
class SCAFFOLD():
    def __init__(self, backbone, configs, hparams, device):
        pass
    
    def aggregate(self, fl_payload):
        
        client_nets = fl_payload['client_encoders']
        c_global = fl_payload['c_global'] #global model variance tracker
        c_delta_para = fl_payload['c_delta_para']
        
        new_net = deepcopy(client_nets[0]) #initialize new global net
        
        w_sum = {}
        w_count = {}
        
        for cnet in client_nets:
          for k, v in cnet.items():
              if k in w_sum:
                  w_sum[k] += v.clone()  # Use clone to ensure no in-place operation
                  w_count[k] += 1
              else:
                  w_sum[k] = v.clone()  # Same as above, make a copy of v
                  w_count[k] = 1

        for k in w_sum:
            new_net[k] = w_sum[k] / w_count[k]
        
        ### Also update the global variance tracker
        total_delta = deepcopy(c_global)
        for key in total_delta:
            total_delta[key] = 0.0
        
        n_clients = len(c_delta_para)
        
        for i in range(n_clients):
            for key in total_delta:
                total_delta[key] += c_delta_para[i][key]
        
        for key in total_delta:
            total_delta[key] /= n_clients #take average of deltas
        
        c_global_para = c_global
        for key in c_global_para:
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
            else:
                #print(c_global_para[key].type())
                c_global_para[key] += total_delta[key]
        c_global = c_global_para
        
        fl_payload['global_encoder'] = new_net
        fl_payload['c_global'] = c_global
        
        return fl_payload
        
class MOON():
    def __init__(self, backbone, configs, hparams, device):
        pass
    
    def aggregate(self, fl_payload):
        
        client_nets = fl_payload['client_encoders']
        
        new_net = deepcopy(client_nets[0])
        
        w_sum = {}
        w_count = {}
        
        for cnet in client_nets:
          for k, v in cnet.items():
              if k in w_sum:
                  w_sum[k] += v.clone()  # Use clone to ensure no in-place operation
                  w_count[k] += 1
              else:
                  w_sum[k] = v.clone()  # Same as above, make a copy of v
                  w_count[k] = 1

        for k in w_sum:
            new_net[k] = w_sum[k] / w_count[k]
        
        fl_payload['global_encoder'] = new_net
        
        return fl_payload