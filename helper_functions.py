#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:00:27 2021

@author: simon
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
import datetime
import torch.nn as nn

data_loc = '/Users/simon/Documents/DTU/9. semester/deep learning/data'

def accuracy_rate(P_P, P_M, Cap):
    return (1 - torch.sqrt(torch.sum(torch.pow(P_M-P_P, 2))) / (Cap * np.sqrt(len(P_P))))


def quantile_score(y, y_pred, q):
    return torch.mean(torch.max(q * (y-y_pred), (1-q) * (y_pred-y)))


def load_data(case):

    df_all = []
    
    files = os.listdir(os.path.join(data_loc,'modified data'))
    for file in sorted(files):
        path = os.path.join(data_loc,'modified data',file)
        name, ext = os.path.splitext(file)
        if ext != '.csv' or name[0]=='.':
            continue
        df = pd.read_csv(path)
        df.name = name
        
        for col_name in df.columns:
            if col_name != 'Date_Time':
                df[col_name]=df[col_name].astype('float64')
            else:
                df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        
        df_all.append(df)
    
    y = torch.Tensor(df_all[case-1]['Park Power [KW]'].values[:,None])
    y_time = df_all[case-1]['Date_Time']
    x = torch.Tensor(df_all[case+2].iloc[:,1:].values)
    x_time = df_all[case+2]['Date_Time']
    
    # make sure all time differences are equal
    assert x_time.diff().min()==x_time.diff().max()
    assert y_time.diff().min()==y_time.diff().max()
    assert x_time.diff().min()==y_time.diff().min()
    # check that training and target data end at the same time
    assert y_time.iloc[-1]==x_time.iloc[-1]
    
    # Time difference between start of training and target data, and corresponding offset in indecies
    time_dif = (y_time.iloc[0]-x_time.iloc[0])
    idx_offset = time_dif.days*24*4+round(time_dif.seconds/(60*15))

    return x,x_time,y,y_time,time_dif,idx_offset


def capacity(case):
    return {1: 49500, 2: 99000, 3: 49500}[case]


def get_good_idx(x,y,idx_offset,pred_seq_len):
    good_idx = []
    for i in range(max(0,-idx_offset+pred_seq_len-1), y.shape[0]):
        if not (torch.isnan(y[i]) or torch.any(torch.isnan(x[i+idx_offset-pred_seq_len+1:i+idx_offset+1]))):
            good_idx.append(i)
    good_idx = np.array(good_idx)
    
    return good_idx
    

def get_slice(batch_idx, batch_size): 
    return range(batch_idx * batch_size, (batch_idx + 1) * batch_size)


#use Stacked Cross-Validation for check appropriate dataset length
def get_stacked_cv_idx(k, good_idx, k_fold_size):
    num_good = len(good_idx)
    good_idx = good_idx.copy()
    np.random.shuffle(good_idx)
    in_valid_bool = np.zeros(num_good, dtype=bool)
    in_train_bool = np.zeros(num_good, dtype=bool)
    
    in_valid_bool[(1+k)*num_good//k_fold_size : (2+k)*num_good//k_fold_size] = 1
    in_train_bool[:(1+k)*num_good//k_fold_size] = 1
    
    valid_idx = good_idx[in_valid_bool]
    train_idx = good_idx[in_train_bool]
    
    return valid_idx, train_idx


def get_k_fold_cv_idx(k, good_idx, k_fold_size):
    num_good = len(good_idx)
    good_idx = good_idx.copy()
    np.random.shuffle(good_idx)
    in_valid_bool = np.zeros(num_good, dtype=bool)
    in_train_bool = np.zeros(num_good, dtype=bool)
    
    in_valid_bool[k*num_good//k_fold_size : (k+1)*num_good//k_fold_size] = 1
    in_train_bool = np.logical_not(in_valid_bool)
    
    valid_idx = good_idx[in_valid_bool]
    train_idx = good_idx[in_train_bool]
    
    return valid_idx, train_idx


def get_x_sequences_ffnn(idx, x_stacked, idx_offset, pred_seq_len, x):
    for i in range(len(idx)):
        x_stacked[i, :] = x[idx[i]+idx_offset-pred_seq_len+1:idx[i]+idx_offset+1, :].reshape(-1)
    return x_stacked


def get_x_sequences_cnn(idx, x_stacked, idx_offset, pred_seq_len, x):
    for i in range(len(idx)):
        x_stacked[i, :, :] = x[idx[i]+idx_offset-pred_seq_len+1:idx[i]+idx_offset+1, :].transpose(0, 1)
    return x_stacked

def get_x_sequences_rnn(idx, x_stacked, idx_offset, pred_seq_len, x):
    for i in range(len(idx)):
        x_stacked[i, :, :] = x[idx[i]+idx_offset-pred_seq_len+1:idx[i]+idx_offset+1, :]
    return x_stacked

def allocate_x_batch_ffnn(batch_size, input_size, pred_seq_len):
    return torch.zeros(batch_size, input_size)

def allocate_x_batch_rnn(batch_size, input_size, pred_seq_len):
    return torch.zeros(batch_size, pred_seq_len, input_size)

def allocate_x_batch_cnn(batch_size, input_size, pred_seq_len):
    return torch.zeros(batch_size, input_size, pred_seq_len)

def train(nn_type, x, y, Net, optim_params, num_epochs, batch_size, good_idx, k_fold_size, idx_offset, pred_seq_len, loss, case):
    q=0.3
    def quantile_score_metric(y_pred, y): 
        return quantile_score(y,y_pred,q)
    def accuracy_rate_metric(y_pred, y): 
        return accuracy_rate(y*capacity(case),y_pred*capacity(case),capacity(case))
    def mseloss(*args): 
        return nn.MSELoss()(*args)
    def maeloss(*args):
        return nn.L1Loss()(*args)
    
    valid_metrics = [mseloss, maeloss, quantile_score_metric, accuracy_rate_metric]
    
    
    
    if nn_type.lower() == 'rnn':
        x_batch = allocate_x_batch_rnn(batch_size, x.shape[1], pred_seq_len)
        x_sequences_fun = get_x_sequences_rnn
    elif nn_type.lower() == 'ffnn':
        x_batch = allocate_x_batch_ffnn(batch_size, x.shape[1]*pred_seq_len, None)
        x_sequences_fun = get_x_sequences_ffnn
    elif nn_type.lower() == 'cnn':
        x_batch = allocate_x_batch_cnn(batch_size, x.shape[1], pred_seq_len)
        x_sequences_fun = get_x_sequences_cnn
    
    
    fold = 0
    valid_idx, train_idx = get_k_fold_cv_idx(fold, good_idx, k_fold_size)
    
    num_samples_train = train_idx.shape[0]
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = valid_idx.shape[0]
    num_batches_valid = num_samples_valid // batch_size
    
    # setting up lists for handling loss/accuracy
    train_loss = np.zeros((num_epochs, k_fold_size))
    
    num_metrics = len(valid_metrics)
    valid_loss = np.zeros((num_metrics, num_epochs, k_fold_size))
    
    for fold in range(k_fold_size):
        net=Net()
        optimizer = optim.Adam(net.parameters(), lr=optim_params['lr'], weight_decay=optim_params['weight_decay'])
        valid_idx, train_idx = get_k_fold_cv_idx(fold, good_idx, k_fold_size)
    
        for epoch in range(num_epochs):
            ## Training
            # Forward -> Backprob -> Update params
            net.train()
            cur_loss = 0
            for i in range(num_batches_train):
                optimizer.zero_grad()
                slce = get_slice(i, batch_size)
                
                x_batch = x_sequences_fun(train_idx[slce], x_batch, idx_offset, pred_seq_len, x)
                
                output = net(x_batch)
                
                # compute gradients given loss
                batch_loss = loss(output, y[train_idx[slce]])
                batch_loss.backward()
                optimizer.step()
                
                cur_loss += batch_loss   
            train_loss[epoch, fold] = cur_loss.detach().numpy() / num_batches_train
        
            ## Validation
            net.eval()
            cur_loss = np.zeros(num_metrics)
            for i in range(num_batches_valid):
                slce = get_slice(i, batch_size)
                
                x_batch = x_sequences_fun(valid_idx[slce], x_batch, idx_offset, pred_seq_len, x)
                output = net(x_batch)
                
                for metric in range(num_metrics):
                    cur_loss[metric] += valid_metrics[metric](output.detach(), y[valid_idx[slce]].detach())
                
            for metric in range(num_metrics):
                valid_loss[metric, epoch, fold] = cur_loss[metric] / num_batches_valid
        
        
            
            
            print(f'Epoch: {epoch}')
        
        print(f'Fold {fold}: Training loss: {train_loss[epoch, fold]:.4f}, \nValidation metrics:', *[f'\n{valid_metrics[i].__name__} {valid_loss[i, epoch, fold]:.4f}' for i in range(num_metrics)], '\n')
        
        
    return train_loss, valid_loss, net


def get_all_accuracy_rates(net, x, y, y_time, x_batch, x_sequences_fun, good_idx, idx_offset, pred_seq_len, case):
    accuracies = []
    times = []
    for idx in range(max(0,-idx_offset+pred_seq_len-1), len(y_time)-96):
        if y_time[idx].time() == datetime.time(0,0):
            x_batch = x_sequences_fun(range(idx, idx+96), x_batch, idx_offset, pred_seq_len, x)
            net.eval()
            P_P = y[idx:idx+96].detach() * capacity(case)
            P_M = net(x_batch).detach() * capacity(case)
            
            ac = accuracy_rate(P_P, P_M, capacity(case))
            if not np.isnan(ac):
                accuracies.append(ac)
                times.append(y_time[idx])
    
    return accuracies, times


def load_competition_data(day, case):
    
    file = os.path.join(data_loc,'competition','modified',f'Competition_case_{case}_day_{day}.csv')

    df = pd.read_csv(file)
    df.name = f'competition_day_{day}_case_{case}'
        
    for col_name in df.columns:
        if col_name != 'Date_Time':
            df[col_name]=df[col_name].astype('float64')
        else:
            df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        
    
    x = torch.Tensor(df.iloc[:,1:].values)
    x_time = df['Date_Time']
    
    # make sure all time differences are equal
    assert x_time.diff().min()==x_time.diff().max()
    
    return x,x_time


def get_competition_preds(day,case,get_x_sequences,allocate_x_batch,input_size,pred_seq_len,net,save):
    file = os.path.join(data_loc,'competition','predictions',f'Competition_case_{case}_day_{day}_predictions.csv')
    x_comp,x_comp_time = load_competition_data(day, case)
    comp_pred_idx = list(range(x_comp.shape[0]-96,x_comp.shape[0]))
    x_batch = allocate_x_batch(len(comp_pred_idx), input_size, pred_seq_len)
    predictions = net(get_x_sequences(comp_pred_idx, x_batch, 0, pred_seq_len, x_comp)).detach().numpy()
    predictions *= capacity(case)
    if save:
        np.savetxt(file,predictions)
    return predictions



            
