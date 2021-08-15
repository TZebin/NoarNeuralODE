#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 02:23:03 2021

@author: zst19phu
"""
# In[1]:

# Setting seeds to try and ensure we have the same results - this is not guaranteed across PyTorch releases.
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

import numpy as np
np.random.seed(0)


import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import DataLoader, TensorDataset



import pandas as pd

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from SurvNODE.SurvNODE_interval import *

# In[2]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# In[4]:
df = pd.read_csv("/gpfs/home/zst19phu/SurvNODE/Data/dataframe_survNODE_v2.csv")
# In[5]:


df_train = df.sample(n=25000)
df_val = df[~df.index.isin(df_train.index)].sample(n=1000)


def get_dataset(df,Tmax):
    x = torch.from_numpy(np.array(df[['gender_male','ageons','(0,25]','(25,30]','(30,100]','ss_1','ss_2','ss_3']])).float().to(device)
    # x = torch.from_numpy(np.array(df[['gender_male','ageons']])).float().to(device)

    Tstart = torch.from_numpy(np.array(df[["T_start"]])).flatten().float().to(device)
    Tstop = torch.from_numpy(np.array(df[["T_stop"]])).flatten().float().to(device)
    Tstart = Tstart/Tmax*multiplier
    Tstop = Tstop/Tmax*multiplier
    From = torch.from_numpy(np.array(df[["From"]])).flatten().int().to(device)
    To = torch.from_numpy(np.array(df[["To"]])).flatten().int().to(device)
    trans = torch.from_numpy(np.array(df[["Trans"]])).flatten().int().to(device)
    status = torch.from_numpy(np.array(df[["Status"]])).flatten().float().to(device)

    dataset = TensorDataset(x,Tstart,Tstop,From,To,trans,status)
    return dataset


multiplier = 1.
Tmax = max(torch.from_numpy(np.array(df_train[["T_stop"]])).flatten().float().to(device))

train_loader = DataLoader(get_dataset(df_train,Tmax),batch_size=4,num_workers=0)
val_loader = DataLoader(get_dataset(df_val,Tmax),batch_size=4,num_workers=0)

# In[5]:

num_in = 8
num_latent = 20
layers_encoder = [100]*2
dropout_encoder = [0.]*2
layers_odefunc = [1000]*3
dropout_odefunc = []

# trans_matrix = torch.tensor([[0.,1.,0.,0.,0.,1.],
#                               [0.,0.,1.,0.,0.,1.],
#                               [0.,0.,0.,1.,0.,1.],
#                               [0.,0.,0.,0.,1.,1.],
#                               [0.,0.,0.,0.,0.,1.],
#                               [0.,0.,0.,0.,0.,0.]]).to(device)



trans_matrix = torch.tensor([[np.nan,1,np.nan,np.nan,np.nan,1],
                             [np.nan,np.nan,1,np.nan,np.nan,1],
                             [np.nan,np.nan,np.nan,1,np.nan,1],
                             [np.nan,np.nan,np.nan,np.nan,1,1],
                             [np.nan,np.nan,np.nan,np.nan,np.nan,1],
                             [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]).to(device)

encoder = Encoder(num_in,num_latent,layers_encoder, dropout_encoder).to(device)
odefunc = ODEFunc(trans_matrix,num_in,num_latent,layers_odefunc).to(device)
block = ODEBlock(odefunc).to(device)
odesurv = SurvNODE(block,encoder).to(device)

optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = 1e-7, lr=1e-4)


# In[]

# from SurvNODE.SurvNODE import *
from SurvNODE.EarlyStopping import EarlyStopping
#  # In[5]: 
early_stopping = EarlyStopping("/gpfs/home/zst19phu/SurvNODE/Checkpoints/GPU_interval",patience=20, verbose=True)
for i in tqdm(range(3)):
    
    odesurv.train()
    for mini,ds in enumerate(train_loader):
        myloss,t2,_ = loss(odesurv,*ds)
        optimizer.zero_grad()
        myloss.backward()    
        optimizer.step()
        
    odesurv.eval()
    with torch.no_grad():
        lossval = 0
        for _,ds in enumerate(val_loader):
            t1,t2,_ = loss(odesurv,*ds)
            lossval += t1.item()
    
#     scheduler.step()
    early_stopping(lossval/len(val_loader), odesurv)
    if early_stopping.early_stop:
        print("Early stopping")
        break
#model=odesurv.load_state_dict(torch.load('/gpfs/home/zst19phu/SurvNODE/Checkpoints/GPU_interval_checkpoint.pt'))

