from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from scipy import io
from sklearn.utils import shuffle
#import data_generator as dg
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
import selfsupervisedtrain as train
import os
import SM1_Find_CPC_ModelPerf as modelperf
import sys
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy,Precision,Recall
from sklearn.metrics import classification_report
dependencies = {
    'SENSOR_MODULE3' : modelcpc.SENSOR_MODULE3,
    'SENSOR_MODULE1' : modelcpc.SENSOR_MODULE1,
    'loss_compute': train.NCE_loss().loss_compute,
    'sensorEncoderModule':modelcpc.sensorEncoderModule,
    'WaveNet' : modelcpc.WaveNet,
     'loss_classifier1':modelperf.loss_classifier1,
    'Resblock' : modelcpc.Resblock,
    'F1_Score':modelperf.F1_Score
}

def Inference(res_dir,cpc_model,res_dir_1,cpc_model1):
  model1 = keras.models.load_model(join(res_dir, cpc_model),custom_objects=dependencies)
  XX = model1.input
  YY = model1.layers[1].output
  model1 = tf.keras.Model(XX, YY)
  x_test = np.load('testbaby_x_inp.npy')
  x_test = np.transpose(x_test ,[0,2,1])
  x_inp_test = model1.predict(x_test)
  model = keras.models.load_model(join(res_dir_1, cpc_model1),custom_objects=dependencies)
  predict = model.predict(x_inp_test)
  y_pos_test = np.load('testbaby_y_pos_oh.npy')
  y_mov_test =  np.load('testbaby_y_mov_oh.npy' )
  data_weight_test = np.load('testbaby_data_weights.npy')
  predictions = np.argmax(predict,axis=-1)
  targets = np.argmax(y_mov_test,axis=-1)
  M = tf.math.confusion_matrix(targets,predictions,num_classes=9)
  target_names = ['class '+str(i) for i in np.arange(9)]
  classification_report1 = classification_report(targets,predictions,target_names= target_names)
  print(M ) 
  print(classification_report1)

enc_dir = 'Experiment-results_CPC_SM1/B/Dim_160/k20/epoch_0/'
enc_model = 'cpc_Models.h5'
classifier_dir = 'BestModels/SM1/Experiment-results_CPC_SM1/B/Dim_160/k20/epoch_0/'
classifier_model = 'CPC_classifier1.h5'
getConfMatForModel(enc_dir,enc_model,classifier_dir,classifier_model)

