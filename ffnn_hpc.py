        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:03:46 2021

@author: simon
"""


import numpy as np
from importlib import reload
import sys
import torch
import torch.nn as nn

import helper_functions
reload(helper_functions)
from helper_functions import accuracy_rate, load_data, capacity, get_good_idx, get_slice, get_k_fold_cv_idx, get_x_sequences_ffnn, train, get_all_accuracy_rates, allocate_x_batch_ffnn, quantile_score, get_competition_preds


import argparse
import sys
import os

parser=argparse.ArgumentParser()

parser.add_argument('--num_hidden', help='Integer number of hidden layers', type=int)
parser.add_argument('--hidden_size', help='Integer size of each hidden layer', type=int)
parser.add_argument('--pred_seq_len', help='Integer number of time steps for each prediction',type=int)
parser.add_argument('--loss', help='Training loss metric, either MSE or L1',type=str)
parser.add_argument('--weight_decay', help='weight_decay, float',type=float)
parser.add_argument('--dropout', help='dropout, float',type=float)
parser.add_argument('--case', help='case, int',type=int)
parser.add_argument('--drop_cols', help='dataframe columns to drop',type=str, default='')

#args = parser.parse_args("--num_hidden=2 --hidden_size=50 --pred_seq_len=25 --loss=MSE --weight_decay=0.01 --dropout=0.1 --case=1".split())
args = parser.parse_args()

num_hidden = args.num_hidden
hidden_size = args.hidden_size
pred_seq_len = args.pred_seq_len
loss = args.loss
weight_decay = args.weight_decay
drop_p = args.dropout
case = args.case
drop_cols = args.drop_cols

#assert torch.cuda.is_available()

outfolder = 'training results'
outfile = f'ffnn_{"_".join(sys.argv[1:]).replace("=","_")}.npz'.lower()
if outfile in os.listdir(outfolder):
    print(f'File {outfile} already exists. Exits.')
    sys.exit()





nn_type = 'ffnn'
allocate_x_batch = allocate_x_batch_ffnn
get_x_sequences = get_x_sequences_ffnn


np.random.seed(2021)

cols_to_drop = []
drop_col_name_dict={'power_curve':('Speed_50m_power_curve','Speed_100m_power_curve'), 'cubed':('Speed_50m_cubed','Speed_100m_cubed'),'season':('season_cos','season_sin'),'time':('time_day_cos','time_day_sin')}
for substring in drop_col_name_dict.keys():
    if substring in drop_cols.lower():
        cols_to_drop.extend(drop_col_name_dict[substring])


x,x_time,y,y_time,time_dif,idx_offset = load_data(case, cols_to_drop)

# Index offset between start and end of training data for one single prediction
# i.e. number of quarters of an hour we wish to train on for each sample
good_idx = get_good_idx(x,y,idx_offset,pred_seq_len)

input_size = x.shape[1] * pred_seq_len
out_size = 1


# define network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.input_linear = nn.Linear(in_features=input_size,
                                      out_features=hidden_size,
                                      bias=True)
        self.output_linear = nn.Linear(in_features=hidden_size,
                                       out_features=out_size,
                                       bias=True)
        self.hidden_list = nn.ModuleList()
        for i in range(num_hidden):
            self.hidden_list.append(nn.Linear(in_features=hidden_size,
                                              out_features=hidden_size,
                                              bias=True))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)
        for hidden_layer in self.hidden_list:
            x = hidden_layer(x)
            x = self.dropout(x)
            x = self.act(x)
        x = self.output_linear(x)

        return x


# setting hyperparameters and gettings epoch sizes
batch_size = 1000
num_epochs = 50
k_fold_size = 6

if loss.lower() == 'mse':
    loss = nn.MSELoss()
elif loss.lower() == 'l1':
    loss = nn.L1Loss()
else:
    raise(Exception('unrecognized loss function'))

optim_params = {'lr': 3e-3, 'weight_decay': weight_decay}
train_loss, valid_loss, net = train(nn_type, x, y, Net, optim_params, num_epochs, batch_size, good_idx, k_fold_size, idx_offset, pred_seq_len, loss, case)
valid_loss[0] = np.sqrt(valid_loss[0])

np.savez(os.path.join(outfolder, outfile), train_loss=train_loss, valid_loss=valid_loss)
