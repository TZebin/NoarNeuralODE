#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:15:28 2021

@author: zst19phu
"""
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# In[4]:
df = pd.read_csv("Data/dataframe_survNODE_v2.csv")

df_trainval = df.sample(n=25422)

df_train = df_trainval.sample(n=2500)
df_val = df_trainval[~df_trainval.index.isin(df_train.index)].sample(n=1000)

def get_dataset(df,Tmax):
    x = torch.from_numpy(np.array(df[['gender_male','ageons','(0,25]','(25,30]','(30,100]']])).float().to(device)
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

train_loader = DataLoader(get_dataset(df_train,Tmax),batch_size=8,num_workers=0)
val_loader = DataLoader(get_dataset(df_val,Tmax),batch_size=8,num_workers=0)

# In[5]:
from SurvNODE.SurvNODE import *
num_in = 5
num_latent = 20
layers_encoder = [40]*2
dropout_encoder = [0.]*2
layers_odefunc = [100]*3
dropout_odefunc = []

trans_matrix = torch.tensor([[1,1,np.nan,np.nan,np.nan,1],
                             [np.nan,1,1,np.nan,np.nan,1],
                             [np.nan,np.nan,1,1,np.nan,1],
                             [np.nan,np.nan,np.nan,1,1,1],
                             [np.nan,np.nan,np.nan,np.nan,1,1],
                             [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]).to(device)

encoder = Encoder(num_in,num_latent,layers_encoder, dropout_encoder).to(device)
odefunc = ODEFunc(trans_matrix,num_in,num_latent,layers_odefunc).to(device)
block = ODEBlock(odefunc).to(device)
odesurv = SurvNODE(block,encoder).to(device)

optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = 1e-7, lr=1e-4)

# In[5]:
from SurvNODE.SurvNODE import *
from SurvNODE.EarlyStopping import EarlyStopping
 # In[5]: 
early_stopping = EarlyStopping("Checkpoints/noard200",patience=10, verbose=True)
for i in tqdm(range(200)):
    
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
    # In[6]:
model=odesurv.load_state_dict(torch.load('Checkpoints/noard200_checkpoint_trained.pt'))
    
df_test = pd.read_csv("Data/test1.csv")
test_loader = DataLoader(get_dataset(df_test,Tmax),batch_size=1,num_workers=0)
temp_t = torch.from_numpy(np.linspace(0.,multiplier*1.0,20))
with torch.no_grad():
      for df in test_loader:
          out = odesurv.predict(df[0],temp_t).cpu()
          pvec = torch.einsum("ikj,k->ij",(out[:,0,:,:],torch.tensor([1.,0.,0.,0.,0.,0.])))
          print(out)
          print(pvec)
          fig,ax = plt.subplots(1,1,figsize=(8,6))
          col = ["blue","red","green", "gray", "pink", "cyan"]
          for j in range(pvec.shape[1]):
            ax.plot(np.array(temp_t)*np.array(Tmax.cpu())/multiplier,pvec[:,j],color=col[j],lw=2)
          plt.xlabel("t")
          plt.ylabel("P(X(t)=k)")
          plt.ylim(-0.05,1.02)
          plt.show()
 