import numpy as np
import cleaned_config as _CONF
from scipy import io
import tensorflow as tf
import keras
import os
class My_ClassifierCustom_Generator(keras.utils.Sequence) :
  
  def __init__(self, input_fileNames, batch_size = _CONF.batch_size,training = True,self_sup = False, encoder_model = None) :
    self.input_fileNames = input_fileNames
    self.batch_size = batch_size
    self.training = training
    self.self_sup =  self_sup
    self.encoder_model = encoder_model
     
  def __len__(self) :
    return len(self.input_fileNames)
  
  def getMask(self) :
    folder = ''
    m = np.empty(shape = (0))
    for i in range(self.__len__()):
         idx = i
     
         batch_x_files = self.input_fileNames[idx * self.batch_size : (idx+1) * self.batch_size]
         bmask = np.array([io.loadmat(folder + file_name.strip() +'Proc_data_mask.mat')['mask_weights']for file_name in batch_x_files])      
         m = np.concatenate((m,np.squeeze(bmask)),axis = 0)
    print(m.shape , "is shape of Mask")
    return m
  def getpos_classes(self):
    folder = ''
    z = np.empty(shape = (0,_CONF.NcatsA))
    y = self.getMask()
    for i in range(self.__len__()):
         idx = i
         batch_x_files = self.input_fileNames[idx * self.batch_size : (idx+1) * self.batch_size]
         batch_z = np.array([io.loadmat(folder + file_name.strip() + 'Proc_pos_oh.mat')['pos_oh'] for file_name in batch_x_files])
         z = np.concatenate((z,np.squeeze(batch_z)),axis = 0)
    print(z.shape , "is shape of Z")
    return z
  def getWeight(self):
    folder = ''
    z = np.empty(shape = (0))
    for i in range(self.__len__()):
         idx = i
         batch_x_files = self.input_fileNames[idx * self.batch_size : (idx+1) * self.batch_size]
         train_weights  = np.array([io.loadmat(folder + file_name.strip() + 'Proc_data_weight.mat')['train_weights']for file_name in batch_x_files])
         z = np.concatenate((z,np.squeeze(train_weights)),axis = 0)
    return z 
  def __getitem__(self, idx) :
    folder = ''
    
    batch_x_files = self.input_fileNames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_x = np.array([io.loadmat(folder + file_name.strip() + 'proc_inp.mat')['x'] for file_name in batch_x_files])
    batch_y = np.array([io.loadmat(folder + file_name.strip() + 'Proc_mov_oh.mat')['mov_oh'] for file_name in batch_x_files])
    if self.training == True:
        train_weights  = np.array([io.loadmat(folder + file_name.strip() + 'Proc_data_weight.mat')['train_weights']for file_name in batch_x_files])
  
        if self.self_sup == True:
            batch_x = np.squeeze(batch_x)
            batch_x = np.transpose( batch_x, (0,2,1))
            x = self.encoder_model.predict(batch_x)  
            return np.squeeze(x),np.squeeze(batch_y),np.squeeze(train_weights)
        else:
            return  np.squeeze(batch_x),np.squeeze(batch_y),np.squeeze(train_weights)
    else:
        bmask = np.array([io.loadmat(folder + file_name.strip() +'Proc_data_mask.mat')['mask_weights']for file_name in batch_x_files])
        batch_x = np.squeeze(batch_x)
        batch_x = np.transpose( batch_x, (0,2,1))
        mask = (1-np.squeeze(bmask)).astype(bool) 
        if self.self_sup == True:
               x = self.encoder_model.predict(batch_x)
               return x,np.squeeze(batch_y),np.squeeze(train_weights)
        else:
               return  batch_x,batch_y











