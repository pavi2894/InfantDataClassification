
#!/usr/bin/env python3
from scipy import io
from sklearn.utils import shuffle
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
import trainV2 as train
import os
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_data(babies,data_folder = 'labeled_DATA/'):
	
    fileName_train=la.getlabeledFileNames(babies,folder = 'Train_Data/',data_folder = data_folder,test=False)  
    file_names = io.loadmat(fileName_train)['Input_fileNames']
    filenames_shuffled = shuffle(file_names)
    filenames_shuffled = filenames_shuffled
    DG_train = dg.My_ClassifierCustom_Generator(filenames_shuffled,_CONF.batch_size)

    x_inp = tf.concat([x for x,y,z in DG_train], axis=0)
    y_mov = tf.concat([y for x,y,z in DG_train], axis=0)
    data_weight = tf.concat([z for x,y,z in DG_train], axis=0)
    y_pos = DG_train.getpos_classes()
    mask = DG_train.getMask()
    return x_inp,y_mov,data_weight,y_pos,mask



babies = ['Kotimittaus_VAURAS35','Kotimittaus_VAURAS38','Kotimittaus_VAURAS39',
'Kotimittaus_VAURAS41_kaksoset','Kotimittaus_VAURAS42_kaksoset','Kotimittaus_VAURAS43',
'Kotimittaus_VAURAS46','Kotimittaus_VAURAS47','Kotimittaus_VAURAS51','Kotimittaus_VAURAS52',
'Kotimittaus_VAURAS53','Kotimittaus_VAURAS63','Kotimittaus_VV54','Kotimittaus_VV55',
'Kotimittaus_VV_xx','Kotimittaus_pilot1','Kotimittaus_pilot2','baby10','baby11','baby12',
'baby13','baby14','baby15','baby16','baby17','baby18','baby19','baby20','baby21','baby22',
'baby23','baby24','baby25','baby26','baby3','baby4','baby5','baby6','baby7','baby8','baby9']

samples = _CONF.testTrainRatio*len(babies)

test_babies = random.sample(babies, int(samples))

test_babies = np.load('Supervised_testbabies.npy')
train_babies = np.setdiff1d(babies, test_babies)
data_folder = 'labeled_DATA/'

x_inp,y_mov,data_weight,y_pos,mask = generate_data(train_babies)
np.save('Dbaby_x_inp.npy', x_inp)
np.save('Dbaby_input_mask.npy',mask)
np.save('Dbaby_y_mov_oh.npy',y_mov)
np.save('Dbaby_y_pos_oh.npy',y_pos)
np.save('Dbaby_data_weights.npy',data_weight)

x_inp,y_mov,data_weight,y_pos,mask = generate_data(test_babies)
np.save('Dtestbaby_x_inp.npy', x_inp)
np.save('Dtestbaby_input_mask.npy',mask)
np.save('Dtestbaby_y_mov_oh.npy',y_mov)
np.save('Dtestbaby_y_pos_oh.npy',y_pos)
np.save('Dtestbaby_data_weights.npy',data_weight)
