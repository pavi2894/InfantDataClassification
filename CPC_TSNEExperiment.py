
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
import trainmodel_cpcV3 as modelcpc
import labelsArrange as la
import random
import selfsupervisedtrainV3 as train
import os
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


dependencies = {
    'SENSOR_MODULE3' : modelcpc.SENSOR_MODULE3,
    'SENSOR_MODULE1' : modelcpc.SENSOR_MODULE1,
    'SENSOR_MODULE3_Modified' : modelcpc.SENSOR_MODULE3_Modified,
    'sensorEncoderModule':modelcpc.sensorEncoderModule,
    'WaveNet' : modelcpc.WaveNet,
    'Resblock' : modelcpc.Resblock
}

#@tf.function
def plot_TSNE(res_dir,cpc_model,DG_test,tsne,true_categories,true_poscategories,image_name,x_inp):
    print("Inside here")
    model = keras.models.load_model(join(res_dir, cpc_model),custom_objects=dependencies)
    model1 = model.layers[1]
    x_enc = model1.predict(DG_test)   
    print(np.any(np.isnan(x_enc))," has nan value")
    print(np.any(np.isinf(x_enc))," has Inf value")
    print(np.all(np.isfinite(x_enc)),"has allfinite value")
    print("CHECK IMP SUMS",np.isnan(x_enc).sum(),np.isinf(x_enc).sum())
    print('x_enc shape is ',x_enc.shape)
    #print('mash shape is ',mask.shape)
    np.save(res_dir +'/enc_output.npy',x_enc )
    tsne_cpc_train_results = tsne.fit_transform(x_enc)
    tsne_1 = tsne_cpc_train_results[:,0]
    tsne_2 = tsne_cpc_train_results[:,1]
    fig = plt.figure()
    fig.suptitle(res_dir, fontsize=10) 
    ax1 = fig.add_subplot(121)
    divider = make_axes_locatable(ax1)
    sc1 = ax1.scatter(tsne_1, tsne_2,c=true_categories,s= 0.4, cmap=plt.cm.get_cmap("jet", config['NcatsB']))
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc1,ticks = range(_CONF.NcatsB),cax = cax,orientation='vertical')
    ax2 = fig.add_subplot(122)
    divider = make_axes_locatable(ax2)
    sc2 = ax2.scatter(tsne_1, tsne_2, c=true_poscategories,s = 0.4, cmap=plt.cm.get_cmap("jet",config['NcatsA']))
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc2,ticks = range(_CONF.NcatsA),cax = cax,orientation='vertical')
    print("path of image is ",res_dir+image_name)
    plt.savefig(TSNE_folder+image_name)
    plt.show()
    plt.close()
    

def invokeTSNE(res_dir,cpc_model= 'cpc_Models.h5',term = _CONF.terms,dim = _CONF.Nlatent,epoch = _CONF.epochs):
    
    test_babies = np.load('Supervised_testbabies.npy')
    fileName_test=la.getlabeledFileNames(test_babies,folder = 'Test_Data/',data_folder = 'labeled_DATA/',test = True)
    file_names2 = io.loadmat(fileName_test)['Input_fileNames']
    filenames_shuffled2 = shuffle(file_names2)
    DG_test = dg.My_ClassifierCustom_Generator(filenames_shuffled2,_CONF.batch_size,training = False)
    y_true = tf.concat([y for x, y in DG_test], axis=0)
    #y_mask = tf.concat([z for x, y,z in DG_test], axis=0)
    y_pos = DG_test.getpos_classes()
    #mask = tf.cast(1-y_mask,tf.bool)
    true_categories = tf.argmax(y_true, axis=1)
    true_poscategories = tf.argmax(y_pos,axis=1)
    tsne = TSNE(n_components=2, init = 'pca',verbose=0, perplexity=25, n_iter=500, learning_rate = 400 )
    image_name = 'TSNE.png'
    plot_TSNE(res_dir,cpc_model,DG_test,tsne,true_categories,true_poscategories,image_name)
 

TSNE_folder = config['TSNE_folder']


if __name__ == "__main__":


    babies = ['Kotimittaus_VAURAS35','Kotimittaus_VAURAS38','Kotimittaus_VAURAS39',
'Kotimittaus_VAURAS41_kaksoset','Kotimittaus_VAURAS42_kaksoset','Kotimittaus_VAURAS43',
'Kotimittaus_VAURAS46','Kotimittaus_VAURAS47','Kotimittaus_VAURAS51','Kotimittaus_VAURAS52',
'Kotimittaus_VAURAS53','Kotimittaus_VAURAS63','Kotimittaus_VV54','Kotimittaus_VV55',
'Kotimittaus_VV_xx','Kotimittaus_pilot1','Kotimittaus_pilot2','baby10','baby11','baby12',
'baby13','baby14','baby15','baby16','baby17','baby18','baby19','baby20','baby21','baby22',
'baby23','baby24','baby25','baby26','baby3','baby4','baby5','baby6','baby7','baby8','baby9']
   
    samples = config['testTrainRatio']*len(babies)
  
    test_babies = random.sample(babies, int(samples))
    print("test_babies ", test_babies)
    train_babies = np.setdiff1d(babies, test_babies)


    #OR test_babies = np.load('Supervised_testbabies.npy')
    if not os.path.isdir('Train_Data/'):
                         os.makedirs('Train_Data/',exist_ok=True)
    fileName_test=la.getlabeledFileNames(babies,folder = 'Train_Data/',data_folder = 'labeled_DATA/',test = True)
    file_names2 = io.loadmat(fileName_test)['Input_fileNames']
    filenames_shuffled2 = shuffle(file_names2)
    DG_test = dg.My_ClassifierCustom_Generator(filenames_shuffled2,config['batch_size'],training = False)
    y_true = tf.concat([y for x, y in DG_test], axis=0)
    #y_mask = DG_test.getMask()
    
    x_inp = tf.concat([x for x, y in DG_test], axis=0) 
   
    
    y_pos = DG_test.getpos_classes()
    #mask = tf.cast(1-y_mask,tf.bool)
    #y_pos = y_pos[mask]
    y_true = tf.reshape(y_true,[-1,config['NcatsB']])
    y_pos = tf.reshape(y_pos,[-1,config['NcatsA']])
    true_categories = tf.argmax(y_true, axis=1)
    true_poscategories = tf.argmax(y_pos,axis=1)

    tsne = TSNE(n_components=2, init = 'pca',verbose=0, perplexity= 40, n_iter=1000, learning_rate = 400 )
    result_dir = config['model_dir']
    terms_all = config['terms_all']
    latent_spaceDimensions =config['latent_spaceDimensions']
     
    for j in latent_spaceDimensions:
        print("latent dimension : ",j) 
        _CONF.Nlatent = j
        for k in terms_all:
                print("terms is ",k)
                _CONF.terms = k
                for i in config['Ntrack']:
                        res_dir = result_dir + i + '/Dim_'+str(j)+'/k'+str(k)+'/'
                        if not os.path.isdir(res_dir):
                                os.makedirs(res_dir,exist_ok=True)
                        if not os.path.isdir(TSNE_folder):
                                      os.makedirs(TSNE_folder,exist_ok=True)
                        epochs = config['curr_epoch']
                        print("Yhe epochs are",epochs)
                        for e in epochs:
                            
                            cpc_model='cpc_Models.h5'
                            image_name = 'TSNE_'+str(j)+'_k'+str(k)+'_epochs'+str(e)+'.png'
                            x_inp =  tf.dtypes.cast(x_inp, tf.float32)
                          
                            if e == 0:
                               if not os.path.isdir(TSNE_folder+'epoch_'+ str(e)):
                                              os.makedirs(TSNE_folder+'epoch_'+ str(e),exist_ok=True)
                               np.save(TSNE_folder +'epoch_'+ str(e)+'/input.npy',x_inp )
                               #np.save(TSNE_folder +'epoch_'+ str(e)+'/input_mask.npy',y_mask)
                               np.save(TSNE_folder +'epoch_'+ str(e)+'/y_mov_oh.npy',y_true)
                               np.save(TSNE_folder +'epoch_'+ str(e)+'/y_pos_oh.npy',y_pos)
                            plot_TSNE(res_dir = res_dir +'epoch_'+ str(e)+'/',cpc_model=cpc_model,DG_test=DG_test,tsne = tsne,true_categories=true_categories,true_poscategories=true_poscategories,image_name=image_name,x_inp = x_inp)
