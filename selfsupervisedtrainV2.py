#!/usr/bin/env python3
from scipy import io
from sklearn.utils import shuffle
import datetime
import time
import classifier_dataGenerator as dg
import numpy as np
import cleaned_config as _CONF
from sklearn.model_selection import train_test_split
from os.path import join, basename, dirname, exists
import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
import trainmodel_cpcV2 as modelcpc
import labelsArrange as la
import random
import os
import sys
#import CPC_TSNEExperiment as tsneplot
from keras.backend import int_shape
import cleaned_aux_module as sp
from tensorflow.keras import regularizers
import argparse

parser = argparse.ArgumentParser(description='Run tensorflow-mnist with Keras')
parser.add_argument('config', type=str, nargs=1,
                    help='Configuration file')

args = parser.parse_args()

config_file = args.config[0]

#####################################################################

## Load parameters from yaml

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open(config_file, 'r') as stream:
    config = load(stream, Loader=Loader)

print('Running with the following configuration from file {0}:'.format(config_file))
print(config)

Ncats = config['Ncats']
encoder_model = config['encoder_model']
batch_size = config['batch_size']
class loss_classifier:

         
        def __init__(self):
             self.weight = tf.ones((batch_size,))
             self.bmask = tf.zeros((batch_size,))
             self.accuracy = None
             self.lossvalue  = None
        def getLossAccuracy(self):
            return self.lossvalue,self.accuracy
        def updateMaskWeight(self,weight,bmask):
            self.weight = weight
            self.bmask = bmask
        def getWeightMask(self):
            return self.weight,self.bmask
        def loss_classifier1(self,y_true,y_pred):     
            weight,bmask = self.getWeightMask()
            
            if Ncats > 1:
                weight =  tf.cast(weight, 'float32')
                softmax =  tf.nn.softmax_cross_entropy_with_logits(
                        labels=y_true,logits=y_pred)
                softmax =  tf.cast(softmax, 'float32')
                loss_classifier = tf.reduce_mean(tf.boolean_mask(
                    weight * softmax,bmask) )          
                predictions = tf.argmax(y_pred,axis=1)
                targets = tf.argmax(y_true,axis=1)
                predictions = tf.cast(predictions, 'float32') 
                targets = tf.cast(targets, 'float32')              
                sm_logit_activations = tf.nn.softmax(y_pred)      
                self.accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(predictions,targets),tf.float32),bmask))
                self.lossvalue = loss_classifier
        
                return  loss_classifier
           


class NCE_loss:
        def __init__(self,k):
            self.error =0.0 
            self.k = k
        def loss_compute(self,y_true,y_pred,neg): 
          term = self.k  
          num = 0.0 
          if config['CPC_k']== 'only':
              min_val = tf.float32.min
              max_val = tf.float32.max
              mult = tf.multiply(y_true,y_pred)  
              num1 = tf.reduce_sum(mult,-1,keepdims = False)
              #num1 = tf.clip_by_value(num1,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))
              num = tf.math.exp(num1)
              #num +=  1e-6 
              denom = num + 1e-15
              const = tf.zeros(denom.shape,  name=None,dtype = tf.float32)
              const = tf.math.maximum(const, 1e-15)
              negative_sample_size = config['negative_sample_size']
              for k in range(negative_sample_size): 
                  denom1 = tf.reduce_sum(tf.multiply(y_pred,neg[:,k,:]),-1,keepdims=False)
                  #denom1 = tf.clip_by_value(denom1,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))
                  denom += tf.math.exp(denom1)   
                  #denom = tf.clip_by_value(denom,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))
              #denom = tf.clip_by_value(denom,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))
              #num1 = tf.clip_by_value(num1,clip_value_min = min_val, clip_value_max = max_val)
              #denom = tf.clip_by_value(denom,clip_value_min = min_val, clip_value_max = 0.99*tf.math.log(max_val))
              #print()
              #if denom.any() > max_val :
              #         denom = tf.clip_by_value(denom,clip_value_min = min_val, clip_value_max = 0.99*tf.math.log(max_val))
              #if num.any() > max_val :
              #          num = tf.clip_by_value(num,clip_value_min = min_val, clip_value_max = max_val)
              #self.error = -1*tf.reduce_mean(num1 - tf.math.log(denom))
              """denom = tf.math.minimum(denom,max_val)
              num = tf.math.minimum(num,max_val)"""
              self.error = -1*tf.reduce_mean(tf.math.log((num/denom)+const))
              #self.error = -1*tf.reduce_mean(tf.math.log(num) - tf.math.log(tf.math.maximum(denom,const)))
              if(tf.math.is_nan(self.error)):
                          #print("term is ", ee )
                          print("Num is ",num)
                          print("Num 1 is ",num1)
                          print("denom is ",denom)
                          print("denom / const is ",tf.math.maximum(denom,const))
                          print("log denom,const is ", tf.math.log(tf.math.maximum(denom,const)))
                          #print("Neg error_k = ",tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(tf.cast(denom,dtype = tf.float32), tf.cast(const,dtype = tf.float32)))))
                          #print("Actua li s", -1*tf.reduce_mean(tf.math.log(tf.cast((num/denom)+tf.cast(const,dtype = tf.float32),dtype=tf.float32))))
                          print("Actual error is ", tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(denom,const))))
                          #print("denom / const is ",tf.math.maximum(denom, tf.cast(const,dtype = tf.float32)))
                          #print("log denom,const is ", tf.math.log(tf.math.maximum(denom, tf.cast(const,dtype = tf.float32))))
                          #print("Neg error_k = ",tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(denom, 1e-15))))
                          sys.exit()
              if(tf.math.is_inf(self.error)):
                          print("Inf occured")
                          #print("term is ", ee )
                          print("Num is ",num)
                          print("Num 1 is ",num1)
                          print("denom is ",denom)
                          print("denom / const is ",tf.math.maximum(denom,const))
                          print("log denom,const is ", tf.math.log(tf.math.maximum(denom,const)))
                          print("Actual error is ", tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(denom,const))))
                          sys.exit()
              #self.error = -1*tf.reduce_mean(tf.math.log(num) - tf.math.log(tf.math.maximum(denom,const)))
              #print("Loss:", self.error,  " probability:", (1e-15 + (num/denom)), "  num:",num,"  denom:",denom)
          else:
              error = 0.0
              min_val = tf.float32.min
              max_val = tf.float32.max
              min_val =  tf.cast(min_val ,tf.float32)
              max_val =  tf.cast(max_val ,tf.float32)
              for ee in range(term):
                  loss_k = 0.0
                  y_pred_k = y_pred[:,ee,:]
                  y_true_k = y_true[ee,:,:]
                  neg_k = neg[ee,:,:,:]
                  mult = tf.multiply(y_true_k,y_pred_k)
                  num1 = tf.reduce_sum(mult,-1,keepdims = False)
                  #num1 = tf.cast(num1,tf.float32) 

                  #num1 = tf.clip_by_value(num1,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))
                  num = tf.math.exp(num1)
                  denom = num + 1e-15
                  #denom = tf.cast(denom,tf.float32)
                  #num1 = tf.clip_by_value(num1,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))
                  const = tf.zeros(denom.shape,  name=None,dtype = tf.float32)
                  const = tf.math.maximum(const, 1e-15)
                  negative_sample_size = config['negative_sample_size']
                  for k in range(negative_sample_size):
                      #denom += tf.cast(tf.math.exp(tf.reduce_sum(tf.multiply(y_pred_k,neg_k[:,k,:]),-1,keepdims=False)) ,dtype = tf.float64)
                      denom1 = tf.reduce_sum(tf.multiply(y_pred_k,neg_k[:,k,:]),-1,keepdims=False)
                      #denom1 = tf.clip_by_value(denom1,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))
                      denom += tf.math.exp(denom1) 
                  
                  #num1 = tf.clip_by_value(num1,clip_value_min = min_val, clip_value_max = max_val)
                  #denom = tf.clip_by_value(denom,clip_value_min = min_val, clip_value_max = 0.99*tf.math.log(max_val))
                  """denom = tf.math.minimum(denom,max_val)
                  num = tf.math.minimum(num,max_val)"""
                  #if tf.math.reduce_max(denom) > max_val :
                  #     denom = tf.clip_by_value(denom,clip_value_min = min_val, clip_value_max = 0.99*tf.math.log(max_val))
                  #if num.any() > max_val :
                  #      num = tf.clip_by_value(num,clip_value_min = min_val, clip_value_max = max_val)
                  #denom = tf.clip_by_value(denom,clip_value_min = tf.cast((tf.float32.min),tf.float32), clip_value_max = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32)) 
                  error_k = -1*tf.reduce_mean(tf.math.log((num/denom)+const))
                  #error_k = tf.constant(np.nan) 
                  if(tf.math.is_nan(error_k)):
                          print("term is ", ee )
                          print("Num is ",num)
                          print("Num 1 is ",num1)
                          print("denom is ",denom)
                          print("denom / const is ",tf.math.maximum(denom,const))
                          print("log denom,const is ", tf.math.log(tf.math.maximum(denom,const)))
                          print("Erro is ",-1*tf.reduce_mean(tf.math.maximum((num1 - tf.math.log(denom)),const)))  
                          #print("Neg error_k = ",tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(tf.cast(denom,dtype = tf.float32), tf.cast(const,dtype = tf.float32)))))
                          #print("Actua li s", -1*tf.reduce_mean(tf.math.log(tf.cast((num/denom)+tf.cast(const,dtype = tf.float32),dtype=tf.float32))))
                          print("Actual error is ", tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(denom,const))))
                          #print("denom / const is ",tf.math.maximum(denom, tf.cast(const,dtype = tf.float32)))
                          #print("log denom,const is ", tf.math.log(tf.math.maximum(denom, tf.cast(const,dtype = tf.float32))))
                          #print("Neg error_k = ",tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(denom, 1e-15))))      
                          sys.exit() 
                  if(tf.math.is_inf(error_k)):
                          print("Inf occured")   
                          print("term is ", ee )
                          print("Num is ",num)
                          print("Num 1 is ",num1)
                          print("denom is ",denom)
                          print("denom / const is ",tf.math.maximum(denom,const))
                          print("log denom,const is ", tf.math.log(tf.math.maximum(denom,const)))
                          print("Erro is ",-1*tf.reduce_mean(tf.math.maximum((num1 - tf.math.log(denom)),const)))
                          print("Actual error is ", tf.reduce_mean(num1 - tf.math.log(tf.math.maximum(denom,const))))
                          sys.exit()
                  error += error_k
              #error = tf.clip_by_value(error,clip_value_min = tf.float32.min, clip_value_max = tf.float32.max)
              """error =  tf.math.minimum(error,max_val)"""
              self.error = error   

          return self.error

def train_model(model_dir,gru_h,k,R=None,model_name='test',randomseed=42, logfile='log.txt', verbose=_CONF.verbose):
    start_time = time.time()
    f = open(model_dir+"LatestRunCPC_NEWConfMatrix.txt", "a")   
    target_names = ['class '+str(i) for i in np.arange(Ncats)]
    
    babies = ['Kotimittaus_VAURAS+33','Kotimittaus_VAURAS+34','Kotimittaus_VAURAS+35',
'Kotimittaus_VAURAS34','Kotimittaus_VAURAS35','Kotimittaus_VAURAS39','Kotimittaus_VAURAS41_kaksoset',
'Kotimittaus_VAURAS43','Kotimittaus_VAURAS46','Kotimittaus_VAURAS47','Kotimittaus_VAURAS51',
'Kotimittaus_VAURAS58','Kotimittaus_VAURAS72','Kotimittaus_VAURAS73','Kotimittaus_VAURAS61',
'Kotimittaus_VAURAS77','Kotimittaus_VAURAS78','Kotimittaus_VAURAS80','Kotimittaus_VAURAS81',
'Kotimittaus_VV54','Kotimittaus_VV55','Kotimittaus_VV61','Kotimittaus_VV62','Kotimittaus_VV63',
'Kotimittaus_VV_xx','Kotimittaus_pilot2','Kotimittaus_vaihe2_VAURAS82','Kotimittaus_vaihe2_VV64',
'Kotimittaus_vaihe2_VV66']
   


    inactive_babies = ['baby3', 'baby20', 'baby5','Kotimittaus_VAURAS61', 'baby14']
    active_babies = ['Kotimittaus_VAURAS72', 'Kotimittaus_VAURAS73', 'Kotimittaus_VAURAS53', 'Kotimittaus_VAURAS41_kaksoset', 'baby17']
    
    
    f.write("tf.test.is_gpu_available()" + str(tf.test.is_gpu_available())+"\n")
    f.write("GPU device :" + tf.test.gpu_device_name()+"\n")
    f.write("TF version is " + tf.__version__ +"\n")

    train_babies = babies
    

    data_folder ='unlabeled_DATA/' 
    folder1 = ''
    folder =''   
    code_size = config['Nlatent2']
    x_r = tf.keras.Input(shape = (_CONF.channels,_CONF.winlen))
    if config['self_supervised'] == 'CPC':
            if encoder_model == 'SENSOR_MODULE1':
                sensor_module = modelcpc.SENSOR_MODULE1('sensor_module', s_channels=_CONF.channels, latent_channels=gru_h//2, output_channels=gru_h, input_channels=_CONF.winlen, dropout_rate=0.3 )
            elif encoder_model == 'SENSOR_MODULE3_Modified':
                sensor_module =  modelcpc.SENSOR_MODULE3_Modified('sensor_module', s_channels=_CONF.channels, latent_channels=gru_h//2, output_channels=gru_h, input_channels=_CONF.winlen, dropout_rate=0.3 )

            elif encoder_model == 'SENSOR_MODULE3':
                sensor_module = modelcpc.SENSOR_MODULE3('sensor_module', s_channels=_CONF.channels, latent_channels=gru_h//2, output_channels=gru_h, input_channels=_CONF.winlen, dropout_rate=0.3 )    
            if config['CPC_k']== 'only':

                      timeseries_model =tf.keras.Sequential([ 
                             tf.keras.layers.GRU(units=gru_h, return_sequences= True,dropout=0.3, name='ar_context'),
                             tf.keras.layers.Dense(units=gru_h, activation='tanh')])
                             #"""tf.keras.layers.Dense(units=gru_h, activation='linear')])"""
                             #tf.keras.backend.clip(min_value = tf.cast((tf.float32.min),tf.float32), max_value = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))])
            else:
                     timeseries_model =tf.keras.Sequential([
                             tf.keras.layers.GRU(units=gru_h, return_sequences= True,dropout=0.3, name='ar_context'),
                             tf.keras.layers.Dense(units=gru_h*k, activation='tanh')])#, activity_regularizer=tf.keras.regularizers.l2(0.001))])
                             #tf.keras.backend.clip(min_value = tf.cast((tf.float32.min),tf.float32), max_value = tf.cast(tf.math.log(0.99*tf.float32.max),tf.float32))])

            sensor_enc =  sensor_module(x_r,training = True)
            enc_model = tf.keras.Model(x_r,sensor_enc)
            sens_enc = enc_model(x_r,training = True)
            sensor_enc1 = tf.expand_dims(sens_enc,axis = 0)
            logit_activations = timeseries_model(sensor_enc1,training = True)#tf.keras.layers.TimeDistributed(timeseries_model)(sensor_op,training=False)

    if config['self_supervised'] == 'wav2vec':
            x_r = tf.keras.Input(shape = (_CONF.channels,_CONF.winlen))
            if encoder_model == 'SENSOR_MODULE1':
                sensor_module = modelcpc.SENSOR_MODULE1('sensor_module', s_channels=_CONF.channels, latent_channels=gru_h//2, output_channels=gru_h, input_channels=_CONF.winlen, dropout_rate=0.3 )
            elif encoder_model == 'SENSOR_MODULE3_Modified':
                sensor_module =  modelcpc.SENSOR_MODULE3_Modified('sensor_module', s_channels=_CONF.channels, latent_channels=gru_h//2, output_channels=gru_h, input_channels=_CONF.winlen, dropout_rate=0.3 )

            elif encoder_model == 'SENSOR_MODULE3':
                sensor_module = modelcpc.SENSOR_MODULE3('sensor_module', s_channels=_CONF.channels, latent_channels=gru_h//2, output_channels=gru_h, input_channels=_CONF.winlen, dropout_rate=0.3 )    
            if config['CPC_k'] == 'only':
                    timeseries_model = modelcpc.WaveNet("wavenet", residual_channels=code_size, output_channels=Ncats, input_channels=code_size,
                postproc_channels=50, dilations=_CONF.timeseries_channels, filter_width=5, dropout=0.3)
            else:
                    timeseries_model = modelcpc.WaveNet("wavenet", residual_channels=code_size, output_channels=Ncats, input_channels=code_size, 
                postproc_channels=50, dilations=_CONF.timeseries_channels, filter_width=5, dropout=0.3)
     
            sensor_enc =sensor_module(x_r,training = False)
    
            logit_activations = timeseries_model(sensor_enc,training=False)
            logit_activations = tf.keras.layers.Dense(units=gru_h, activation='linear')(logit_activations)
        
    model = tf.keras.Model(inputs = [x_r], outputs = [logit_activations], name='self-supervised')
    model.summary() 
    f.write(str(model.summary()))
    #print("babies shape is ", len(babies))
    if _CONF.use_lr_decay:
        learning_rate = tf.train.exponential_decay(_CONF.learning_rate, global_step, 
                                           1000, 0.96, staircase=True)
    else:
        learning_rate = _CONF.learning_rate

    wc = NCE_loss(k=k)
        
    epochs = config['cpc_epochs']
    train_loss_results = []
    

    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate) 
    
    for e in range(epochs):
        loss_epoch = 0.0
        f.write("\n\n\n\n---------EPOCH "+str(e)+"----------------")
        print("EPOCH IS ", e)     
        for iBaby in np.arange(len(babies)):
            babyLoss = 0.0 
            f.write("\n\n i am bbay :"+ str(babies[iBaby]) + " of epoch :"+str(e)+"\n\n")
            acc_data = io.loadmat(data_folder + babies[iBaby] + "/acc_data.mat")['acc_data']

            gyro_data = io.loadmat(data_folder + babies[iBaby] + "/gyro_data.mat")['gyro_data']
            x_r = np.concatenate((acc_data/10.0, gyro_data/100.0),axis=1)
            x_r = sp.frame_sig(x_r,_CONF.winlen,_CONF.hop)

            x_r = sp.preprocess_data(x_r)
            batches_size = x_r.shape[0]#get_shape().as_list()[0]
            batches = tf.range(0, batches_size, 1) 
            #print("batches: ",batches)
            shuffled_batch = tf.random.shuffle(batches,seed  = iBaby)
            shuffled_batch2 = shuffled_batch          
            x_r = tf.transpose(x_r, perm=[0,1,3,2]) 
            x_r = tf.squeeze(x_r)
            for tt in shuffled_batch:
               curr_batch = tt
               tf.random.set_seed(tt)  
               """if _CONF.neg_sampling == 'same_baby':
                     neg_idxs = tf.random.uniform(shape=[_CONF.negative_sample_size], maxval= batches_size, dtype=tf.int32, seed=tt)"""
               x = x_r[tt,:,:,:]
               with tf.GradientTape() as tape:
                 y = model.layers[1](x,training = True)
                 slice_val = _CONF.frames_per_sample - k - 1 
                 preds1 = (model.layers[2])(y,training = True)
                 preds = (model.layers[3])(preds1,training = True)
                 if config['CPC_k']== 'only':
                    neg_samples = tf.zeros([0,config['negative_sample_size'],gru_h])
                    preds =preds[:,:slice_val,:]
                    if config['neg_sampling'] == 'same_batch':
                         targ = tf.roll(y,shift = -k,axis = 0)[:slice_val] 
                         idxs = tf.range(_CONF.frames_per_sample)
                         curr_indx =  np.array([tt])
                         for yy in range(_CONF.frames_per_sample):
                           curr_batch_sample_idx = [yy+k]    
                           ridxs = tf.random.shuffle(idxs)#[:_CONF.frames_per_sample]
                           mask1 = tf.where( curr_batch_sample_idx[0] != ridxs ,ridxs,0) 
                           negIndxChoices = tf.boolean_mask(ridxs, mask1)
                           negsam = tf.gather(y,indices = negIndxChoices,axis = 0)[:config['negative_sample_size']]
                           neg = tf.expand_dims(negsam,axis = 0 )
                           neg_samples = tf.concat([neg_samples,neg],axis = 0)
                         neg_samples = neg_samples[:slice_val,:,:]
                 else: 
                     preds  = tf.squeeze(preds)
                     preds = tf.reshape(preds,[_CONF.frames_per_sample,-1,gru_h])
                     preds = preds[:slice_val,:,:]
                     targ = tf.zeros([0,slice_val,gru_h])
                     neg_samples = tf.zeros([0,slice_val,config['negative_sample_size'],gru_h])
                     if config['neg_sampling'] == 'same_batch':
                        for ps in range(k):
                            neg_sample_k = tf.zeros([0,config['negative_sample_size'],gru_h])
                            targ_k = tf.roll(y,shift = -(ps+1),axis = 0)[:slice_val]
                            targ_k= tf.expand_dims(targ_k,axis = 0)
                            targ = tf.concat([targ,targ_k],axis = 0) 
                            idxs = tf.range(_CONF.frames_per_sample)
                            curr_indx =  np.array([tt])
                            for yy in range(_CONF.frames_per_sample):
                                 curr_batch_sample_idx = [yy+ps+1]
                                 ridxs = tf.random.shuffle(idxs)
                                 mask1 = tf.where( curr_batch_sample_idx[0] != ridxs ,ridxs,0)
                                 negIndxChoices = tf.boolean_mask(ridxs, mask1)
                                 negsam = tf.gather(y,indices = negIndxChoices,axis = 0)[:config['negative_sample_size']]
                                 neg = tf.expand_dims(negsam,axis = 0 )
                                 neg_sample_k = tf.concat([neg_sample_k,neg],axis = 0)
                            neg_sample_k = neg_sample_k[:slice_val,:,:]    
                            neg_sample_k = tf.expand_dims(neg_sample_k ,axis = 0 )
                            neg_samples = tf.concat([neg_samples,neg_sample_k],axis=0)
 
                 if config['neg_sampling'] == 'same_baby':  
                   for n in neg_idxs:
                        neg = (model.layers[1])(tf.squeeze(x_r[n,:slice_val,:,:]),training = False)
                        neg = tf.expand_dims(neg,axis = 0 )
                        neg_samples = tf.concat([neg_samples,neg],axis = 0)
                 loss_value = wc.loss_compute(targ,preds,neg_samples)
                 f.write(str(loss_value)+  " is one of the loss values and probability is "+str(tf.math.exp(-loss_value))+ "\n")
                 gradients = tape.gradient(loss_value ,model.trainable_variables)
                 gradients, _ = tf.clip_by_global_norm(gradients, 5.0) 
                 opt.apply_gradients(zip(gradients, model.trainable_variables))
              
                 if not tf.math.is_nan(loss_value):
                        babyLoss += loss_value  
            babyLoss /= batches_size
            loss_epoch = loss_epoch + babyLoss
                    
        sec = (time.time() - start_time)   
        print("--- After Epoch:",e,"   time elapsed :---",str(datetime.timedelta(seconds=sec)) )
        loss_epoch = loss_epoch/len(babies)
        train_loss_results.append(loss_epoch)
                
        f.write("Epoch is "+str(e)+ " and the epoch loss is "+str(loss_epoch))  
        print("Epoch {:03d}: Loss: {:.3f}".format(e,loss_epoch))
        if e % 1 == 0:
                 
              print("Epoch {:03d}: Loss: {:.3f}".format(e,loss_epoch))
              if not os.path.isdir(model_dir+'epoch_'+str(e)+'/'):

                           os.makedirs(model_dir+'epoch_'+str(e)+'/',exist_ok=True)
              model.save(join(model_dir+'epoch_'+str(e)+'/', 'cpc_Models.h5'))
              #tsneplot.invokeTSNE(res_dir = model_dir+'epoch_'+str(e)+'/',cpc_model= 'cpc_Models.h5',term = _CONF.terms,dim = _CONF.Nlatent,epoch = e) 
    io.savemat(model_dir+'CPC_train_loss_results.mat', {'train_loss_results': train_loss_results})
     
    model.save(join(model_dir, 'FinalCPC_'+str(e)+'.h5')) 
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
  
    axes.set_ylabel("Loss", fontsize=14)
    axes.plot(train_loss_results)
    axes.set_xlabel("Epoch", fontsize=14)
    
    plt.show()
    plt.savefig(model_dir+ 'TraininLoss.png')
    f.close() 
    sec1 = (time.time() - start_time)
    print("--- Total time elapsed :---",str(datetime.timedelta(seconds=sec1)) )
    #print("--- Total seconds elapsed :---", (time.time() - start_time))
    print("SUCCESS")
    return
        
if __name__ == "__main__":
    #result_dir = 'test-results/'
    
    result_dir = config['model_dir'] 
    terms_all = config['terms_all']
    latent_spaceDimensions = config['latent_spaceDimensions']
    for j in latent_spaceDimensions:
        _CONF.Nlatent = j
      
        for k in terms_all:
                _CONF.terms = k
                for i in _CONF.Ntrack:
                    
                     res_dir = result_dir + i + '/Dim_'+str(j)+'/k'+str(k)+'/'
          
                     if not os.path.isdir(res_dir):
              
                          os.makedirs(res_dir,exist_ok=True)

                     train_model(model_dir= res_dir, gru_h= j,k= k)
