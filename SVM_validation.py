import cleaned_config as _CONF
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
import trainmodel_cpcV2 as modelcpc
import labelsArrange as la
import random
import selfsupervisedtrain as train
import os
import sys
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy,Precision,Recall
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import LambdaCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
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
import sys
import argparse
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy,Precision,Recall
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import LambdaCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

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
    'loss_compute': train.NCE_loss().loss_compute,
    'sensorEncoderModule':modelcpc.sensorEncoderModule,
    'WaveNet' : modelcpc.WaveNet,
    'Resblock' : modelcpc.Resblock
}

def getConfMatMetrics(M ,tars=None, preds=None, Ncats=None, return_avg=True):

    M = np.array(M)
    print(M)    
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


    #import ipdb;ipdb.set_trace()
    if return_avg:
        prec = np.nanmean(prec)
        rec = np.nanmean(rec)
        f1 = np.nanmean(f1)

    return f1# prec, rec, f1


SVM_path = config['SVM_folder'] 
def classifier(res_dir,cpc_model):
    
    model = keras.models.load_model(join(res_dir, cpc_model),custom_objects=dependencies)
    model1 = model.layers[1]

    ress = res_dir.replace('/','_')
    x_test = np.load('testbaby_x_inp.npy')
    x_test = np.transpose(x_test,[0,2,1])
    num = int(len(x_test)*0.8)
    x_inp1 = x_test[:num,:,:] 
    x_test =  x_test[num:,:,:]
    X = model1.predict(x_inp1)
    x_inp_test = model1.predict(x_test)
    y_pos_test = np.load('testbaby_y_pos_oh.npy')
    y_pos1 = y_pos_test[:num,:]
    y_pos_test =y_pos_test[num:,:]
    y_mov_test =  np.load('testbaby_y_mov_oh.npy' )
    y_mov_test = np.argmax(y_mov_test,axis = -1)
    y_mov_test = np.squeeze(y_mov_test)
    y = y_mov_test[:num]
    y_mov_test=y_mov_test[num:]
    data_weight_test = np.load('testbaby_data_weights.npy')
    data_weight1 = data_weight_test[num:]
    data_weight_test  = data_weight_test[num:]
    X_train = X
    y_train = y
    X_test =  x_inp_test
    y_test = y_mov_test
    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)

    linear_pred = linear.predict(X_test)
    poly_pred = poly.predict(X_test)
    rbf_pred = rbf.predict(X_test)
    sig_pred = sig.predict(X_test)
    accuracy_lin = linear.score(X_test, y_test)
    accuracy_poly = poly.score(X_test, y_test)
    accuracy_rbf = rbf.score(X_test, y_test)
    accuracy_sig = sig.score(X_test, y_test)
    ress = res_dir.replace('/','_')
    #mkdir SVM
    f = open(SVM_path+ress+'.txt','a')
    if not os.path.isdir(SVM_path):
                    os.makedirs(SVM_path,exist_ok=True)  
    f.write("Validation Accuracy : " + ress +  "\n")
    f.write("Accuracy Linear Kernel:"+str(accuracy_lin)+"\n")
    f.write("Accuracy Polynomial Kernel:" + str(accuracy_poly)+"\n")
    f.write("Accuracy Radial Basis Kernel:"+str(accuracy_rbf)+"\n")
    f.write("Accuracy Sigmoid Kernel:"+ str(accuracy_sig)+"\n")
# creating a confusion matrix
    cm_lin = confusion_matrix(y_test, linear_pred)
    cm_poly = confusion_matrix(y_test, poly_pred)
    target_names = ['class '+str(i) for i in np.arange(Ncats)]
    predictions = np.argmax(rbf_pred,axis = -1)
    targets = np.argmax(y_test,axis = -1)
    cm_rbf = confusion_matrix(y_test, rbf_pred)
    classification_report1 = classification_report(y_test, linear_pred,target_names= target_names)
    classification_report2 = classification_report(y_test, poly_pred,target_names= target_names)
    classification_report3 = classification_report(y_test, rbf_pred,target_names= target_names)
    classification_report4 = classification_report(y_test, sig_pred,target_names= target_names)
    f1  = precision_recall_fscore_support(y_test, rbf_pred,average = 'macro')
    val_fscore = f1[2]
    cm_rbf = confusion_matrix(y_test, rbf_pred)
    cm_sig = confusion_matrix(y_test, sig_pred)

    f.write("\nValidation CM:" + ress+"\n")
    f.write("Linar kernel:\n"+str(cm_lin)+"\n")
    f.write("Linear kernel classification report"+"\n"+str(classification_report1)+"\n")
    f.write("Polynomial Kernel\n:"+str(cm_poly)+"\n")
    f.write("polynomial kernel classification report"+"\n"+str(classification_report2)+"\n")
    f.write("Radial Basis Kernel:\n"+str(cm_rbf)+"\n")#+str(classification_report3)+"\n")
    f.write("Radial kernel classification report"+"\n"+str(classification_report3)+"\n")
    f.write("Sigmoid Kernel:\n"+str(cm_sig)+"\n")
    f.write("Sigmoid kernel classification report"+"\n"+str(classification_report4)+"\n")

    ress = res_dir.replace('/','_') 
    f.close() 
    return val_fscore

if __name__ == "__main__":
    

    result_dir = config['model_dir'] 
    #fscore_dict_final = {}
    terms_all = config['terms_all']#[1,2,3,4]
    latent_spaceDimensions =config['latent_spaceDimensions']#140,128,64,32,160,16]#,140,128,64,32,16,8]
    for j in latent_spaceDimensions:
        fscore_dict={}
        #.Nlatent = j
        for k in terms_all:
                #_CONF.terms = k
                for i in config['Ntrack']:
                        res_dir = result_dir + i + '/Dim_'+str(j)+'/k'+str(k)+'/'
                        if not os.path.isdir(res_dir):
                                os.makedirs(res_dir,exist_ok=True)
                        epochs  = config['curr_epoch']#np.arange(0,config['epochs'],1)

                        for e in epochs:

                            cpc_model='cpc_Models.h5'
                            res = res_dir +'epoch_'+ str(e)+'/'
                            ress = res.replace('/','_')
                            val = classifier(res_dir = res_dir +'epoch_'+ str(e)+'/',cpc_model=cpc_model)
                            fscore_dict[ress] = val
                            #fscore_dict_final[ress] = [val,inf]
        res1 = SVM_path +str(j)+'/'
        if not os.path.isdir(res1):
                    os.makedirs(res1,exist_ok=True)                
        np.save(res1+"dict.npy",fscore_dict)
    #np.save("SVM/Final_dict.npy",fscore_dict_final)

