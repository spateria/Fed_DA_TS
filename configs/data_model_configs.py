import os
import torch
import random

def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        
        self.domain_ids = list(range(0,19 + 1)) 
        self.scenarios = []
        for did in self.domain_ids:
          src_ids = [did]
          trgt_ids = [other_did for other_did in self.domain_ids if other_did not in src_ids]
          self.scenarios.append( (src_ids,trgt_ids) ) 
        '''self.scenarios = [
                  ([2], [13, 0, 10, 5, 11]), 
                  ([18], [13, 3, 9, 10, 6]), 
                  ([0], [17, 16, 3, 11, 7]), 
                  ([12], [15, 9, 14, 6, 18]), 
                  ([19], [9, 5, 18, 11, 2]), 
                  ([18], [19, 7, 16, 6, 8]), 
                  ([11], [16, 8, 14, 15, 18]), 
                  ([8], [16, 5, 7, 10, 14]), 
                  ([5], [3, 0, 10, 1, 18]), 
                  ([9], [2, 11, 17, 12, 6])
        ]'''
        
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 65 # for my model
        self.AR_hid_dim = 8

        # AR Discriminator
        self.disc_hid_dim = 256
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1
        
class FD():
    def __init__(self):
        super(FD, self).__init__()
        
        self.domain_ids = list(range(0,3 + 1)) 
        self.scenarios = []
        for did in self.domain_ids:
          src_ids = [did]
          trgt_ids = [other_did for other_did in self.domain_ids if other_did not in src_ids]
          self.scenarios.append( (src_ids,trgt_ids) ) 
        
        #self.scenarios = [ ([1],[0,3]) ]
        '''self.scenarios = [
                ([0], [1, 2, 3]), 
                ([1], [0, 2, 3]), 
                ([2], [0, 1, 3]), 
                ([3], [0, 1, 2])
              ]'''
              
        self.sequence_len = 5120         
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.AR_hid_dim = 128
        
class HAR():
    def __init__(self):
        super(HAR, self)
        
        self.domain_ids = list(range(1,30 + 1)) 
        self.scenarios = []
        for did in self.domain_ids:
          src_ids = [did]
          trgt_ids = [other_did for other_did in self.domain_ids if other_did not in src_ids]
          self.scenarios.append( (src_ids,trgt_ids) )
         
        #self.scenarios = [ ([1], [2,3,5]) ] 
        '''self.scenarios = [
                ([8], [22, 1, 29, 14, 24]), 
                ([4], [9, 1, 5, 17, 23]), 
                ([20], [4, 26, 1, 7, 14]), 
                ([28], [6, 7, 24, 12, 5]), 
                ([22], [29, 25, 5, 10, 14]), 
                ([18], [15, 17, 20, 8, 4]), 
                ([19], [20, 6, 22, 13, 21]), 
                ([14], [5, 19, 6, 18, 25]), 
                ([13], [24, 7, 22, 6, 17]), 
                ([26], [11, 2, 10, 27, 14])
              ]'''
              
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # self.features_len = 18 for sequential methods
        self.AR_hid_dim = 128




