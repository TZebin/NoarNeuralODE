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
from SurvNODE.EarlyStopping import EarlyStopping
from SurvNODE.SurvNODE_variational import *


# In[2]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# In[4]:
# df = pd.read_csv("Data/dataframe_survNODE_v2.csv")
df = pd.read_csv("Data/dataframe_survNODEv3.csv")

# In[5]:


# df_train = df.sample(n=25000)
# df_val = df[~df.index.isin(df_train.index)].sample(n=1000)
df_train = df.sample(n=2500)
df_val = df[~df.index.isin(df_train.index)].sample(n=150)


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
layers_prior = [100]*2
dropout_prior = [0.]*2
layers_post = [500]*2
dropout_post = [0.]*2
layers_odefunc = [1000]*3
dropout_odefunc = []

trans_matrix = torch.tensor([[np.nan,1,np.nan,np.nan,np.nan,1],
                              [np.nan,np.nan,1,np.nan,np.nan,1],
                              [np.nan,np.nan,np.nan,1,np.nan,1],
                              [np.nan,np.nan,np.nan,np.nan,1,1],
                              [np.nan,np.nan,np.nan,np.nan,np.nan,1],
                              [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]).to(device)

prior_mean = Prior(num_in,num_latent,layers_prior, dropout_prior).to(device)
prior_var = Prior(num_in,num_latent,layers_prior, dropout_prior).to(device)
post_mean = Post(num_in,num_latent,layers_post, dropout_post).to(device)
post_var = Post(num_in,num_latent,layers_post, dropout_post).to(device)
odefunc = ODEFunc(trans_matrix,num_in,num_latent,layers_odefunc,dropout_odefunc).to(device)
block = ODEBlock(odefunc).to(device)
odesurv = VarSurvNODE(block,prior_mean,prior_var,post_mean,post_var).to(device)


# trans_matrix = torch.tensor([[0.,1.,0.,0.,0.,1.],
#                               [0.,0.,1.,0.,0.,1.],
#                               [0.,0.,0.,1.,0.,1.],
#                               [0.,0.,0.,0.,1.,1.],
#                               [0.,0.,0.,0.,0.,1.],
#                               [0.,0.,0.,0.,0.,0.]]).to(device)



optimizer = torch.optim.Adam(odesurv.parameters(), weight_decay = 1e-7, lr=1e-4)


# In[]

# from SurvNODE.SurvNODE import *
from SurvNODE.EarlyStopping import EarlyStopping
#  # In[5]: 
early_stopping = EarlyStopping("Checkpoints/GPU_variational",patience=20, verbose=True)
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
#model=odesurv.load_state_dict(torch.load('Checkpoints/GPU_variational_checkpoint.pt'))
# In[]
def sample_probs(odesurv,x,initial, points=500, inner_samples=1, outer_samples=1, multiplier=1.):
    """
        sample probabilites from 0 to multiplier at "points" number f points from initial state "initial" (e.g. [1,0,0] in the illness-death case starting out at Health)
    """    
    xrep = x.repeat(inner_samples,1)
    curve_vec = []
    for _ in range(outer_samples):
        surv_ode = odesurv.predict(xrep,torch.from_numpy(np.linspace(0,multiplier,points)).float().to(x.device))
        pvec = torch.einsum("ilkj,k->ilj",(surv_ode[:,:,:,:],initial))
        samp_curves = pvec.cpu().numpy()
        samp_curves = samp_curves.transpose().reshape((initial.shape[0],int(xrep.shape[0]/x.shape[0]),x.shape[0],points))
        curve_vec.append(samp_curves)
    curve_vec = np.concatenate(curve_vec, axis=1)
    curve_mean = np.mean(curve_vec,axis=1)
    curve_upper = np.percentile(curve_vec,95,axis=1)
    curve_lower = np.percentile(curve_vec,5,axis=1)
    return curve_mean, curve_lower ,curve_upper

df_test = pd.read_csv("Data/test1.csv")
test_loader = DataLoader(get_dataset(df_test,Tmax),batch_size=1,num_workers=0)
with torch.no_grad():
      for df in test_loader:
          print(df[0])
          curve_mean, curve_lower, curve_upper = sample_probs(odesurv,df[0],torch.tensor([1.,0.,0.,0.,0.,0.,0.],device=device),inner_samples=200,outer_samples=10)
          fig,ax = plt.subplots(1,1,figsize=(8,6))
          col_vec = ["green","cyan","blue", "gray", "pink", "red"]
          lines = []
          for j in range(curve_mean.shape[0]):
              ax.fill_between(np.linspace(0,1,curve_mean.shape[-1])*Tmax.cpu().max().numpy(), curve_lower[j,0,:], curve_upper[j,0,:],alpha=0.2,color=col_vec[j])
              lines+=ax.plot(np.linspace(0,1,curve_mean.shape[-1])*Tmax.cpu().max().numpy(),curve_mean[j,0,:],color=col_vec[j])
          ax.set_title("Test set1")
          ax.legend(lines[:6], ['Healthy', '1 condition','2', '3','4&+', 'Death'],loc='upper right', frameon=False)
          plt.xlabel("t")
          plt.ylabel("P(X(t)=k),starting state 2")
          plt.ylim(-0.05,1.02)
          plt.show()