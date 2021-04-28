#!/usr/bin/env python3
import numpy as np
import cleaned_config as _CONF
from scipy import io
import tensorflow as tf
import keras
import os


class My_ClassifierCustom_Generator(keras.utils.Sequence) :
  
  def __init__(self, input_fileNames, batch_size = _CONF.batch_size,training = True) :
    self.input_fileNames = input_fileNames
    self.batch_size = batch_size
    self.training = training
    
  def __len__(self) :
    #return (np.ceil(len(self.input_fileNames) / float(self.batch_size))).astype(np.int)
    return len(self.input_fileNames)
  
 

 
  def __getitem__(self, idx) :
    folder = ''
    batch_x_files = self.input_fileNames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_x = np.array([io.loadmat(folder + file_name.strip() + 'proc_inp.mat')['x'] for file_name in batch_x_files])
    
    
    batch_y = np.array([io.loadmat(folder + file_name.strip() + 'Proc_mov_oh.mat')['mov_oh'] for file_name in batch_x_files])
    
    if self.training == True:
        train_weights  = np.array([io.loadmat(folder + file_name.strip() + 'Proc_data_weight.mat')['train_weights']for file_name in batch_x_files])
       
        bmask = np.array([io.loadmat(folder + file_name.strip() +'Proc_data_mask.mat')['mask_weights']for file_name in batch_x_files])

   
        return  batch_x, batch_y,train_weights,bmask
    else:
        bmask = np.array([io.loadmat(folder + file_name.strip() +'Proc_data_mask.mat')['mask_weights']for file_name in batch_x_files])
        batch_x = np.squeeze(batch_x)
        batch_x = np.transpose( batch_x, (0,2,1))
        return  batch_x, np.squeeze(batch_y),np.squeeze(bmask)











