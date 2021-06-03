#!/usr/bin/env python3
from scipy import io
from sklearn.utils import shuffle
#import data_generator as dg
import classifier_dataGenerator as dg
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import cleaned_config as _CONF
from sklearn.model_selection import train_test_split
from os.path import join, basename, dirname, exists
import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
import trainmodel_cpcV4 as modelcpc
import labelsArrange as la
import random
import selfsupervisedtrainV3 as train
import os
import sys
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy,Precision,Recall
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import LambdaCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import argparse
import homedata_augmentation as aug
import GenerateLabeledDataForClassification as GD

cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
#ffolder = 'BestModels/ModifiedLoss/SM1/Dim_64/Data_Aug/Softmax/'#MultDense_epo250wDropout/'
#if not os.path.isdir(ffolder):
#       os.makedirs(ffolder,exist_ok=True)
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
dependencies = {
    'SENSOR_MODULE3' : modelcpc.SENSOR_MODULE3,
    'SENSOR_MODULE1' : modelcpc.SENSOR_MODULE1,
    'SENSOR_MODULE3_Modified' : modelcpc.SENSOR_MODULE3_Modified,
    'loss_compute': train.NCE_loss(config['terms_all']).loss_compute,
    'sensorEncoderModule':modelcpc.sensorEncoderModule,
    'WaveNet' : modelcpc.WaveNet,
    'Resblock' : modelcpc.Resblock
}
Ncats = config['Ncats']
def getF1Metrics(M):

    fscore = tf.Variable(tf.zeros((Ncats,1), dtype=tf.dtypes.float64, name=None))
    has_nans = tf.constant([float('NaN'), 1.])
    prec_ = tf.Variable(tf.zeros((Ncats,1), dtype=tf.dtypes.float64, name=None))
    rec_ = tf.Variable(tf.zeros((Ncats,1), dtype=tf.dtypes.float64, name=None))
    for i in range(M.get_shape().as_list()[0]):
            val = tf.math.reduce_sum(M[:,i])
            prec =(M[i,i])/(tf.math.reduce_sum(M[i,:]))
            rec = M[i,i]/(tf.math.reduce_sum(M[:,i]))            
            prec = tf.where(tf.math.is_nan(prec), tf.zeros_like(prec), prec)            
            rec = tf.where(tf.math.is_nan(rec), tf.zeros_like(rec), rec)
            f1 = 2.0*prec*rec/(prec + rec+1e-6)
            f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)     
            f1 =  tf.cast(f1, 'float64')
            prec =  tf.cast(prec, 'float64')
            rec = tf.cast(rec, 'float64')
            fscore[i,0].assign(f1)
            prec_[i,0].assign(prec)
            rec_[i,0].assign(rec)   
    f1val = tf.math.reduce_mean(fscore)
    precval = tf.math.reduce_mean(prec_)
    recval = tf.math.reduce_mean(rec_)    
    return f1val #precval,recval

def getF1Metrics_numpy(M , return_avg=True):
    Ncats = config['Ncats']
    prec = np.zeros(Ncats)
    rec = np.zeros(Ncats)
    f1 = np.zeros(Ncats)
    acc = np.sum(M.diagonal())/np.float32(np.sum(M))
    for i in range(M.shape[0]):
        # Check if target contains current category
        containsCat = np.sum(M[:,i]) > 0
        if containsCat:
            prec[i] = np.float32(M[i,i])/np.float32(np.sum(M[i,:]))
            rec[i] = np.float32(M[i,i])/np.float32(np.sum(M[:,i]))
            if np.isnan(prec[i]):
                prec[i] = 0.0
            if np.isnan(rec[i]):
                rec[i] = 0.0    
            f1[i] = 2.0*prec[i]*rec[i]/(prec[i] + rec[i])
   
            if np.isnan(f1[i]):
                f1[i] = 0.0        
        else:
            prec[i] = np.nan; rec[i] = np.nan; f1[i] = np.nan
    if return_avg:
        prec = np.nanmean(prec)
        rec = np.nanmean(rec)
        f1 = np.nanmean(f1)
    return  f1

class loss_classifier:#(tf.keras.losses.Loss):

        def __init__():#(self,name = "customWeightedEntropy",**kwargs):
             
             #super().__init__(name=name, **kwargs)
             self.accuracy = None
             self.lossvalue  = None
        def getLossAccuracy(self):
            return self.lossvalue,self.accuracy
        #def updateMaskWeight(self,weight):
        #    self.weight = weight
            
        def getWeightMask(self):
            return self.weight



class LossCustom(keras.losses.Loss):
   def __init__(self,**kwargs):
         super(LossCustom, self).__init__( name="")
   def __call__(self, y_true, y_pred,sample_weight):
            weightsB= tf.constant([0.3152, 0.1163, 0.0144, 0.0136, 0.1181, 0.2363, 0.1756, 0.0047, 0.0057])
            #weightsA = tf.constant([0.316200554295792,0.116301335348954,0.0148652053413958,0.0140589569160998,0.119198790627362,0.236470143613001,0.182905013857395])
            weightsB=tf.expand_dims(weightsB,axis =0)
            #weightsA=tf.expand_dims(weightsA,axis =0)
            #sample_weight  = tf.linalg.matmul(y_true , weightsB,transpose_b = True )
            #print(weight.shape, "weight")
            #print(sample_weight.shape,"sample_weight")
            #weight= sample_weight

            Ncats = _CONF.Ncats
            #print("y_true,y_pred",y_true.shape,y_pred.shape)
            if Ncats > 1:
                #sample_weight =  tf.cast(sample_weight, 'float32')
                """softmax =  tf.nn.softmax_cross_entropy_with_logits(
                        labels=y_true,logits=y_pred)"""
                #softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
                #softmax =  tf.cast(softmax, 'float32')
                #print("weight * softmax",weight.shape,softmax.shape)
                #loss_classifier_ = tf.reduce_mean(
                #    sample_weight * softmax )

                """loss_classifier_ = tf.reduce_mean(sample_weight *softmax )"""
                loss_classifier_ = tf.reduce_mean(cce(y_true,y_pred) * sample_weight)
                #predictions = tf.argmax(y_pred,axis=1)
                #targets = tf.argmax(y_true,axis=1)
                #predictions = tf.cast(predictions, 'float32')
                #targets = tf.cast(targets, 'float32')
                #tf.print("predictions : ",predictions,"   targets:",targets)
                #sm_logit_activations = tf.nn.softmax(y_pred)

                #self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,targets),tf.float32))
                #self.lossvalue = loss_classifier_

                return  loss_classifier_



def loss_classifier1(y_true,y_pred,sw):
            Ncats = config['Ncats']
            weightsB= tf.constant([0.3152, 0.1163, 0.0144, 0.0136, 0.1181, 0.2363, 0.1756, 0.0047, 0.0057])
            weightsA = tf.constant([0.316200554295792,0.116301335348954,0.0148652053413958,0.0140589569160998,0.119198790627362,0.236470143613001,0.182905013857395])
            weightsB=tf.expand_dims(weightsB,axis =0)
            weightsA=tf.expand_dims(weightsA,axis =0) 
            #weightsB /= tf.reduce_sum(weightsB)
            #weightsB = (1/weightsB)   #Inverted and normalized by sum
            weight  = tf.linalg.matmul(y_true , weightsB,transpose_b = True )
            loss_classifier_ = tf.reduce_sum(cce(y_true,y_pred) * weight)               
            return  loss_classifier_

def custom_loss(y_true,y_pred):#,sample_weight):
            y_true = tf.reshape(y_true ,[-1,config['Ncats']])
            y_pred = tf.reshape(y_pred,[-1,config['Ncats']])
            Ncats = config['Ncats']
            weightsB= tf.constant([0.3152, 0.1163, 0.0144, 0.0136, 0.1181, 0.2363, 0.1756, 0.0047, 0.0057])
            #weightsB= 
            #weightsB = (1/weightsB)
            #weightsA = tf.constant([0.316200554295792,0.116301335348954,0.0148652053413958,0.0140589569160998,0.119198790627362,0.236470143613001,0.182905013857395])
            #weightsB=tf.expand_dims(1/weightsB,axis =0)
            #if weight_type == 'sum_norm':
            weightsB /= tf.reduce_sum(weightsB)
            #weightsB=tf.expand_dims(1/weightsB,axis =0)
            #weightsA=tf.expand_dims(weightsA,axis =0) 
            weightsB=tf.expand_dims(weightsB,axis =0)
            weight  = tf.linalg.matmul(y_true , weightsB,transpose_b = True )
            #loss_classifier_ =  cce(y_true,y_pred) * weight               
            #return  loss_classifier_
            Ncats = _CONF.Ncats
            
            if Ncats > 1:
                weight =  tf.cast(weight, 'float32')
                softmax =  tf.nn.softmax_cross_entropy_with_logits(
                        labels=y_true,logits=y_pred)
                softmax =  tf.cast(softmax, 'float32')
                loss_classifier_ = tf.reduce_mean(
                    weight * softmax )
            return loss_classifier_ 
            

def retTrue():
    return True

def retFalse():
    return False

def get_f1(M,i):
         prec =(M[i,i])/(tf.math.reduce_sum(M[i,:])+ 1e-15)
         rec = M[i,i]/(tf.math.reduce_sum(M[:,i])+ 1e-15)
         f1 = 2.0*prec*rec/(prec + rec+1e-15)
         return f1

def get_fiNan(M,i):
    prec = has_nans
    rec = has_nans
    f1 = has_nans 
    return f1

has_nans = tf.constant(float('NaN'))
class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.fscore = tf.Variable(tf.zeros((config['Ncats'],1), dtype=tf.dtypes.float32, name=None))
        self.f1val = 0
       
 
    def update_state(self,y_true, y_pred):
        targets = tf.reshape(y_true ,[-1,config['Ncats']])
        predictions = tf.reshape(y_pred,[-1,config['Ncats']])
        targets = tf.argmax(targets,axis = 1)
        predictions = tf.argmax(predictions,axis = 1)
        M = tf.math.confusion_matrix(targets,predictions,num_classes= config['Ncats'])
        M = tf.cast(M,dtype=tf.float32) 
        for i in range(M.get_shape().as_list()[0]):
            val = tf.math.reduce_sum(M[:,i])
            prec =(M[i,i])/(tf.math.reduce_sum(M[i,:])+ 1e-15)
            rec = M[i,i]/(tf.math.reduce_sum(M[:,i])+ 1e-15)
            f1 = 2.0*prec*rec/(prec + rec+1e-15)
            prec = tf.where(val <= 0, has_nans, prec)
            rec = tf.where(val <= 0, has_nans, rec)
            f1 = tf.where(val <= 0, has_nans, f1)
            self.fscore[i,0].assign(f1)
        self.f1val = tf.reduce_mean(tf.boolean_mask(self.fscore, tf.math.is_finite(self.fscore)))

 
    def result(self):
        return self.f1val

    def reset_states(self):
        self.f1val = 0



def classifier(res_dir,cpc_model,ffolder,latent):

  babies = ['Kotimittaus_VAURAS35','Kotimittaus_VAURAS38','Kotimittaus_VAURAS39',
'Kotimittaus_VAURAS41_kaksoset','Kotimittaus_VAURAS42_kaksoset','Kotimittaus_VAURAS43',
'Kotimittaus_VAURAS46','Kotimittaus_VAURAS47','Kotimittaus_VAURAS51','Kotimittaus_VAURAS52',
'Kotimittaus_VAURAS53','Kotimittaus_VAURAS63','Kotimittaus_VV54','Kotimittaus_VV55',
'Kotimittaus_VV_xx','Kotimittaus_pilot1','Kotimittaus_pilot2','baby10','baby11','baby12',
'baby13','baby14','baby15','baby16','baby17','baby18','baby19','baby20','baby21','baby22',
'baby23','baby24','baby25','baby26','baby3','baby4','baby5','baby6','baby7','baby8','baby9']
  half_baby = len(babies)#//2
  np.random.seed(42)
  np.random.shuffle(babies)
  babies= babies[:half_baby]
  model = keras.models.load_model(join(res_dir, cpc_model),custom_objects=dependencies)
  model1 = model.layers[1]
  model1.trainable = False
  #model3  = model.layers[3]
  #print(model.layers[-1].get_layer(index=0))

  """model2 = model.layers[1]
  model2.trainable = False
  XX = model.input
  zz = model2(XX)
  zz = tf.expand_dims(zz,axis = 0)
  layer1 = model.layers[-1].get_layer(index=0)
  layer1.trainable = False
  YY = layer1(zz)
  model1 = tf.keras.Model(XX, YY)
  del model
  print(model1.summary())"""



  

  #print(model.layers[-1].get_layer(index=1))
  #print(model.layers[-1].get_layer(index=2))
  #for layer in model.layers[-1]:
  #   print("Layer is ", layer)
  #XX = model3.input
  #YY = XX = model3.output
  #model2 = model.layers[2]
  #XX = model1.input
  #YY = model3.output
  
  #XX = model.input

  """model2 = tf.keras.Model([XX], [YY])
  print(model2.summary())
  print("WHAT?")"""
  #XX = model3.input
  #print(XX.shape)
  #YY = model.layers[1].output
  #print(YY.shape)
  #model4 = tf.keras.Model(XX, YY)
  #print(model.summary())
  #print(model1.summary())
  #print(model2.summary())
  #print(model3)
  #del model
  #print("xx:",XX.shape) 
  #zz = model1(XX)
  #zz = tf.expand_dims(zz,axis = 0) 
  #layer1 = model.layers[-1].get_layer(index=0)  
  #layer1.trainable = False
  #YY = layer1(zz)
  
  #model_freeze = tf.keras.Model(XX, YY)
  del model
  #print(model_freeze.summary())
  
  ress = res_dir.replace('/','_')
  g = open(ffolder+'ConfusionMatrix.txt','a')
  
  batches = 64#config['SS_batch']
   
  f_score_per_fold = []
  acc_per_fold = []
  loss_per_fold = []

  testf_score_per_fold = []
  testacc_per_fold = []
  testloss_per_fold = []

  fold_no = 1
  samples = 4 #config['testTrainRatio']*len(babies)
  np.random.shuffle(babies)
  folds = config['CPC_folds']
  for q in range(folds):#config['CPC_folds']):   
        #test_babies_indx = tf.random.uniform([int(samples)], minval=0, maxval=len(babies), dtype=tf.dtypes.int32, seed=42, name='test_babies')  
        test_size = int(len(babies)/folds)
        print("Folds is ", folds)
        print("test_size",test_size)
        if q+1 != folds:
             test_babies = babies[q*test_size : (q+1)*test_size]#random.sample(babies, int(samples))
        else :
             test_babies = babies[q*test_size : ]
        train_babies = np.setdiff1d(babies, test_babies)
        print("test_babies : ",test_babies)
        print("train_babies : ",train_babies) 
        x_inp1,y_mov1,data_weight1,y_pos1,mask_ = GD.generate_data(train_babies)
        
        x_test,y_mov_test,data_weight_test,y_pos_test,tm_ = GD.generate_data(test_babies)
        print(data_weight1.shape, "mask.shape")   
        mask = (1-np.squeeze(mask_)).astype(bool)        
        #data_weight1 = data_weight1 * mask
        testmask = (1-np.squeeze(tm_)).astype(bool)
        #data_weight_test=  data_weight_test * testmask
        #print("mask is",mask)
        #print("AFTER",data_weight1.shape, mask.shape)
        #sys.exit()
        g.write("test_babies :"+str(test_babies)+"\n")
        g.write("train_babies :"+str(train_babies)+"\n")
        print(x_inp1.shape,"I am here to check this")
        #print(y_mov1.shape,data_weight1.shape,y_pos1.shape,_.shape) 

        x_inp1 = np.transpose(x_inp1 ,[0,2,1])
        x_test = np.transpose(x_test,[0,2,1])
        print("After transpose",x_inp1.shape)
        #sys.exit()
        #print(x_inp1[0].shape)
        #print(x_inp1[1943].shape) 
        x_inp = model1.predict(x_inp1) 
        x_inp = x_inp[mask]
        y_mov1 = y_mov1[mask]
        data_weight1 = data_weight1[mask]
        y_pos1 = y_pos1[mask]         
        x_inp_test = model1.predict(x_test)
        x_inp_test =x_inp_test[testmask]
        y_mov_test = y_mov_test[testmask]
        data_weight_test = data_weight_test[testmask]
        y_pos_test = y_pos_test[testmask]
        #print( x_inp.shape," x_inp ") 
        x_inp = tf.squeeze(x_inp)
        x_input = tf.keras.layers.Input((config['Nlatent']))
        #x_input = tf.squeeze(x_input ,axis = 0)
        #input_shape = (config['frames_per_sample'],config['Nlatent'])
        #x_input = tf.keras.layers.Input((input_shape))
      
        #x_inp = tf.reshape(x_inp ,[-1,config['frames_per_sample'],config['Nlatent']])
        #y_mov1 = tf.reshape(y_mov1 ,[-1,config['frames_per_sample'],config['Ncats']])
        #data_weight1 = tf.reshape(data_weight1,[-1,config['frames_per_sample']])
        #x_inp_test = tf.reshape(x_inp_test,[-1,config['frames_per_sample'],config['Nlatent']])
        #y_mov_test= tf.reshape(y_mov_test,[-1,config['frames_per_sample'],config['Ncats']])
        if config['ss_classifier'] == 'Softmax':
            x = tf.keras.layers.Dense(units=config['Ncats'], activation='softmax')(x_input)

        elif config['ss_classifier'] == 'Dense2Softmax':
             x = tf.keras.layers.Dense(units = latent, activation='relu')(x_input)
             x = tf.keras.layers.Dropout(0.3)(x)
             x = tf.keras.layers.Dense(units = latent, activation='relu')(x)
             x = tf.keras.layers.Dropout(0.3)(x)
             x = tf.keras.layers.Dense(units=config['Ncats'], activation='softmax')(x)

        elif config['ss_classifier'] == 'DeepDense':
             x = tf.keras.layers.Dense(units = latent, activation='relu')(x_input)
             x = tf.keras.layers.Dropout(0.3)(x)
             x = tf.keras.layers.Dense(units = latent, activation='relu')(x)
             x = tf.keras.layers.Dropout(0.3)(x)
             x = tf.keras.layers.Dense(units = latent, activation='relu')(x)
             x = tf.keras.layers.Dropout(.4)(x)
             x = tf.keras.layers.Dense(units = 64, activation='relu')(x)
             x = tf.keras.layers.Dropout(.4)(x)
             x = tf.keras.layers.Dense(units=config['Ncats'], activation='softmax')(x)
        elif config['ss_classifier'] == 'convnet':
            #batches = 1
            #input_shape = (config['frames_per_sample'],config['Nlatent'])
            #x_input = tf.keras.layers.Input((input_shape))
            #x_inp = tf.reshape(x_inp ,[-1,config['frames_per_sample'],config['Nlatent']])
            #y_mov1 = tf.reshape(y_mov1 ,[-1,config['frames_per_sample'],config['Ncats']])
            #data_weight1 = tf.reshape(data_weight1,[-1,config['frames_per_sample']])
            #x_inp_test = tf.reshape(x_inp_test,[-1,config['frames_per_sample'],config['Nlatent']])
            #y_mov_test= tf.reshape(y_mov_test,[-1,config['frames_per_sample'],config['Ncats']])
            x = tf.keras.layers.Conv1D(128, 3, activation='relu',padding = 'SAME',input_shape=(config['frames_per_sample'],config['Nlatent']))(x_input)
            x = tf.keras.layers.Dropout(0.3)(x)
            x =  tf.keras.layers.Conv1D(128, 3, activation='relu',padding = 'SAME')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Conv1D(128, 3, activation='relu',padding = 'SAME')(x)
            x = tf.keras.layers.Dense(units=config['Ncats'],activation='softmax')(x)
        elif config['ss_classifier'] == 'wavenet':
            input_shape = (config['frames_per_sample'],config['Nlatent'])
            x_input = tf.keras.layers.Input((input_shape))
            x_inp = tf.reshape(x_inp ,[-1,config['frames_per_sample'],config['Nlatent']])
            y_mov1 = tf.reshape(y_mov1 ,[-1,config['frames_per_sample'],config['Ncats']])
            data_weight1 = tf.reshape(data_weight1,[-1,config['frames_per_sample']])
            x_inp_test = tf.reshape(x_inp_test,[-1,config['frames_per_sample'],config['Nlatent']])
            y_mov_test= tf.reshape(y_mov_test,[-1,config['frames_per_sample'],config['Ncats']])
            batches = 1
            #x_inp = np.expand_dims(x_inp,axis  = 0)
            #x_inp = np.expand_dims(x_inp,axis  = -1)
            #x_inp_test = np.expand_dims(x_inp_test,axis=0)
            #x_inp_test = np.expand_dims(x_inp_test,axis=-1)
            #x_input = tf.keras.layers.Input((config['frames_per_sample'],config['Nlatent'],1))
            timeseries_model = modelcpc.WaveNet("wavenet", residual_channels=_CONF.Nlatent2, output_channels=_CONF.NcatsB, input_channels=_CONF.Nlatent,
                postproc_channels=_CONF.Nlatent2, dilations=_CONF.timeseries_channels, filter_width=5, dropout_rate=0.3)
            x_inp = tf.squeeze(x_inp)
            x_inp_test = tf.squeeze(x_inp_test )
            x = timeseries_model(x_input)

        elif config['ss_classifier'] == 'cnn':
            #batches = 1   
            input_shape = (config['frames_per_sample'],config['Nlatent'])
            x_input = tf.keras.layers.Input((input_shape))
            x_inp = tf.reshape(x_inp ,[-1,config['frames_per_sample'],config['Nlatent']])
            y_mov1 = tf.reshape(y_mov1 ,[-1,config['frames_per_sample'],config['Ncats']])
            data_weight1 = tf.reshape(data_weight1,[-1,config['frames_per_sample']])
            x_inp_test = tf.reshape(x_inp_test,[-1,config['frames_per_sample'],config['Nlatent']])
            y_mov_test= tf.reshape(y_mov_test,[-1,config['frames_per_sample'],config['Ncats']])
            x = tf.keras.layers.Conv1D(128, 3, activation='relu',padding = 'SAME',input_shape=(config['frames_per_sample'],config['Nlatent']))(x_input)
            x = tf.keras.layers.Dense(units=config['Ncats'], activation='softmax')(x) 
            
        model = keras.models.Model(inputs=x_input, outputs=x)
        #model.compile(optimizer="Adam", loss= loss_classifier1, metrics=['accuracy',F1_Score()])
        model.compile(optimizer="Adam", loss=LossCustom(),metrics=['accuracy',F1_Score()])
        print(model.summary())
        if not os.path.isdir(ffolder):
             os.makedirs(ffolder,exist_ok=True)
        txt_log = open(ffolder + ress + "loss_history.txt", mode='a', buffering=1)  
        save_op_callback = LambdaCallback(
        on_epoch_end = lambda epoch, logs: txt_log.write(
        str({'epoch': epoch, 'loss': logs['loss'],'accuracy':logs['accuracy'],'f1_score':logs['f1_score']}) + '\n'),
        on_train_end = lambda logs: txt_log.close()
        )
        #x_inp_test = model_freeze.predict(x_test)
        #batches = 1
        steps_per_epochs1 = (len(x_inp))//batches
        print("steps_per_epochs1",steps_per_epochs1)
        print("y_mov1", y_mov1.shape," data_weight1:",data_weight1.shape)
        print("x_inp shape is model fit : ",x_inp.shape)       
        result=  model.fit(
        x=x_inp, y=y_mov1, batch_size= batches , epochs=config['epochs_self'], verbose=1, callbacks=[save_op_callback],
        validation_split=0.0, shuffle=True,
        sample_weight=data_weight1, initial_epoch=0, steps_per_epoch=steps_per_epochs1, validation_freq=1,
        max_queue_size=10, workers=1, use_multiprocessing=False)

        with open(ffolder + ress + "loss_history.txt", 'a') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        if not os.path.isdir(ffolder +res_dir):
                                os.makedirs(ffolder +res_dir,exist_ok=True)
        model.save(join(ffolder +res_dir, 'CPC_classifier'+str(fold_no)+'.h5'))        
        #x_inp_test = model_freeze.predict(x_test,batch_size=100)
        print("  x_inp_test : ",x_inp_test.shape,"   y_mov_test : ",y_mov_test.shape)
        testscores = model.evaluate(x=x_inp_test,y = y_mov_test,sample_weight = data_weight_test)
        testacc_per_fold.append(testscores[1] * 100)
        testloss_per_fold.append(testscores[0])
     
        target_names = [i for i in np.arange(config['Ncats'])]
      
        predictions_y = model.predict(x_inp_test)
        targets = tf.reshape(y_mov_test ,[-1,config['Ncats']])
        predictions_y = tf.reshape(predictions_y,[-1,config['Ncats']])
        targets = tf.argmax(targets,axis = 1)
        #predictions = tf.argmax(predictions,axis = 1)

        
        predictions = np.argmax(predictions_y,axis=-1)
        #targets = np.argmax(y_mov_test,axis=-1)
        M = tf.math.confusion_matrix(targets,predictions,num_classes= config['Ncats'])
        f1 = precision_recall_fscore_support(targets,predictions,average = 'macro')
        testf_score_per_fold.append(f1[2]* 100)
        
        classification_report1 = classification_report(targets,predictions,labels= target_names)
        print(classification_report1 )      
  
        g.write("\n"+ress+"  Fold "+str(q)+":\n")
        g.write("\n\n The confusion matrix for inference Data is "  + str(M)+"\n\n")
        g.write("\n\n The classification report for inference data is " + str(classification_report1))
        g.flush()
        fold_no = fold_no + 1

  f = open(ffolder + ress + "loss_history.txt", 'a')
  f.write(str(model.summary()))


  
  f.write('\n\n\n------------------------------------------------------------------------\n')
  f.write('Inference Score per fold\n')
  for i in range(0, len(acc_per_fold)):
     f.write('------------------------------------------------------------------------\n')
     f.write("> Fold {"+ str(i+1) +"} - Loss: {" + str(testloss_per_fold[i]) + "} - Accuracy: {" +str(testacc_per_fold[i]) + "}  -  F1-Score: {" + str(testf_score_per_fold[i])+"}")
     f.write('\n')
  f.write('\n------------------------------------------------------------------------\n')
  f.write('NOT NEEDED Average scores for all folds on :\n')
  f.write('> F-Score:{' + str(np.mean(testf_score_per_fold))+ '}  (+- {'+str(np.std(testf_score_per_fold)) + '})\n')
  f.write('> Accuracy: {' + str(np.mean(testacc_per_fold))+ '}  (+- {'+ str(np.std(testacc_per_fold)) + '})\n')
  f.write('> Loss: {' + str(np.mean(testloss_per_fold)) + '}\n')
  f.write('------------------------------------------------------------------------\n')
  
  f.close()
  g.close()
  return testf_score_per_fold,testacc_per_fold,[np.mean(testf_score_per_fold),np.mean(testacc_per_fold)]








if __name__ == "__main__":

    Ncats = config['Ncats']
    ss_classifier = config['ss_classifier']
    #model_dir  = _CONF.model_dir
    CPC_k = config['CPC_k']
    CPC_k = CPC_k.upper()
    sensor_enc_model = config['encoder_model']
    epochsNum = config['curr_epoch'][0]
    Dim = config['Nlatent']
    term_num = config['terms_all'][0]
    #cpc_path = model_dir + '/'+CPC_k + '/'+ sensor_enc_model + '/'+'B/Dim_'+str(Dim)+'/'+ 'k'+str(term_num)+'/epoch_'+str(epochsNum)+'/cpc_Models.h5' #"CorrectedCPC/ONLY/SENSOR_MODULE3/woReg/wRegOneclip/B/Dim_128/k3/epoch_50/cpc_Models.h5"

    ffolder = 'Classifier_results/half_num_babies/B/Dim_'+str(Dim)+'/'+'k'+str(term_num)+'/epoch_'+str(epochsNum)+ '/'+  sensor_enc_model+'/'+ss_classifier+'/'
    result_dir =  config['model_dir']
    terms_all =  config['terms_all']
    latent_spaceDimensions = config['latent_spaceDimensions']
    for j in latent_spaceDimensions: 
        accuracy_dict = {}     
        #ffolder = config['ss_model']+ config['encoder_model'] + config['CPC_k']+str(j)+config['ss_classifier']+'/'
        #ffolder = 'OldClassifier_results/'+CPC_k+ '/B/Dim_'+str(Dim)+'/'+'k'+str(term_num)+'/epoch_'+str(epochsNum)+ '/'+  sensor_enc_model+'/'+ss_classifier+'/'
        if not os.path.isdir(ffolder):
              os.makedirs(ffolder,exist_ok=True)
        for k in terms_all: 
                for i in config['Ntrack']:                       
                        res_dir = result_dir + i + '/Dim_'+str(j)+'/k'+str(k)+'/'
                        if not os.path.isdir(res_dir):
                                os.makedirs(res_dir,exist_ok=True)
                        epochs  = config['curr_epoch']
                       
                        for e in epochs:
                            ffolder = 'tanhresults_classifier/'+CPC_k+ '/B/Dim_'+str(j)+'/MaskAppliedafterEncoderPredict/Enc_GRU/Old_NoConcatAug/'+'k'+str(k)+'/epoch_'+str(e)+ '/'+  sensor_enc_model+'/'+ss_classifier+'/'
                            if not os.path.isdir(ffolder):
                                     os.makedirs(ffolder,exist_ok=True)
                            cpc_model='cpc_Models.h5'
                            testf_s,testacc_s,acc = classifier(res_dir = res_dir +'epoch_'+ str(e)+'/',cpc_model=cpc_model,ffolder =ffolder,latent=j)
                            accuracy_dict[res_dir +'epoch_'+ str(e)+'/'] = acc
                print("THE FINAL DICTIONARY IS ", accuracy_dict)
                np.save(ffolder + 'Final_dict' + str(j)+'.npy',accuracy_dict)
        t = open(ffolder + 'Avg_Metric_dict.txt','a')
        t.write(res_dir+'\n'+str(accuracy_dict)+'\n\n')
        t.close()
        t = open(ffolder +'Metric_dict.txt','a')
        t.write( res_dir+'\n'+ str(testf_s)+'\n'+ str(testacc_s)+'\n\n')
        t.close()


