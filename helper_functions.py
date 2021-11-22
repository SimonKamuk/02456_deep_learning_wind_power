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


def accuracy_rate(P_P, P_M, Cap):
    n=96
    return (1 - np.sqrt(np.sum(np.power(P_M-P_P, 2))) / (Cap * np.sqrt(n)))


def load_data(case):
    data_loc = '/Users/simon/Documents/DTU/9. semester/deep learning/data'

    df_all = []
    
    files = os.listdir(os.path.join(data_loc,'modified data'))
    for file in sorted(files):
        path = os.path.join(data_loc,'modified data',file)
        name, ext = os.path.splitext(file)
        if ext != '.csv':
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
    for i in range(y.shape[0]):
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


def get_x_sequences_flattened(idx, x_stacked, idx_offset, pred_seq_len, x):
    for i in range(len(idx)):
        x_stacked[i, :] = x[idx[i]+idx_offset-pred_seq_len+1:idx[i]+idx_offset+1, :].view(-1)
    return x_stacked


def get_x_sequences_3d(idx, x_stacked, idx_offset, pred_seq_len, x):
    for i in range(len(idx)):
        x_stacked[i, :, :] = x[idx[i]+idx_offset-pred_seq_len+1:idx[i]+idx_offset+1, :].transpose(0, 1)
    return x_stacked


def train(x_batch, x, y, net, num_epochs, batch_size, good_idx, k_fold_size, x_sequences_fun, idx_offset, pred_seq_len, loss, valid_metrics):
    optimizer = optim.Adam(net.parameters())
    fold = 0
    valid_idx, train_idx = get_k_fold_cv_idx(fold, good_idx, k_fold_size)
    
    num_samples_train = train_idx.shape[0]
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = valid_idx.shape[0]
    num_batches_valid = num_samples_valid // batch_size
    
    # setting up lists for handling loss/accuracy
    train_loss = np.zeros(num_epochs)
    
    num_metrics = len(valid_metrics)
    valid_loss = np.zeros((num_metrics, num_epochs))
    
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
        train_loss[epoch] = np.sqrt(cur_loss.detach().numpy() / num_batches_train)
    
        ## Validation
        net.eval()
        cur_loss = np.zeros(num_metrics)
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            x_batch = x_sequences_fun(valid_idx[slce], x_batch, idx_offset, pred_seq_len, x)
            output = net(x_batch)
            
            for metric in range(num_metrics):
                cur_loss[metric] += valid_metrics[metric](output, y[valid_idx[slce]])
            
        for metric in range(num_metrics):
            valid_loss[metric, epoch] = np.sqrt(cur_loss[metric] / num_batches_valid)
    
    
        
        if epoch % (num_epochs//10) == 0:
            print(f'Epoch {epoch:2d} : Training loss: {train_loss[epoch]:.4f}, Validation:', *[f'{valid_metrics[i]._get_name()} {valid_loss[i, epoch]:.4f}' for i in range(num_metrics)])
            
    return train_loss, valid_loss


def get_all_accuracy_rates(net, x, y, y_time, x_batch, x_sequences_fun, good_idx, idx_offset, pred_seq_len, case):
    accuracies = []
    times = []
    for idx in range(len(y_time)-96):
        if y_time[idx].time() == datetime.time(0,0):
            x_batch = x_sequences_fun(range(idx, idx+96), x_batch, idx_offset, pred_seq_len, x)
            net.eval()
            P_P = y[idx:idx+96].detach().numpy() * capacity(case)
            P_M = net(x_batch).detach().numpy() * capacity(case)
            
            ac = accuracy_rate(P_P, P_M, capacity(case))
            if not np.isnan(ac):
                accuracies.append(ac)
                times.append(y_time[idx])
    
    return accuracies, times
            
            
            
            
