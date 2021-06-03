#!/usr/bin/env python3
from scipy import io
from sklearn.utils import shuffle
#import data_generator as dg
import classifier_dataGeneratorV2 as dg
import numpy as np
import cleaned_configV2 as _CONF
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
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support






class loss_classifier:
         
        def __init__(self):
             self.weight = tf.ones((_CONF.frames_per_sample,))
             self.bmask = tf.zeros((_CONF.frames_per_sample,))
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
            Ncats = _CONF.Ncats
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
           


dependencies = {
    'SENSOR_MODULE3' : modelcpc.SENSOR_MODULE3,
    'SENSOR_MODULE1' : modelcpc.SENSOR_MODULE1,
    'loss_classifier1': loss_classifier().loss_classifier1,
    'sensorEncoderModule':modelcpc.sensorEncoderModule,
    'sensorEncoderModule1':modelcpc.sensorEncoderModule1,
    'WaveNet' : modelcpc.WaveNet,
    'Resblock' : modelcpc.Resblock
}  


def train_model(model_dir,R=None,model_name='test',randomseed=42, logfile='log.txt', verbose=_CONF.verbose):
   f = open(model_dir+"ConfMatrix.txt", "a")    

   babies = ['Kotimittaus_VAURAS35','Kotimittaus_VAURAS38','Kotimittaus_VAURAS39',
'Kotimittaus_VAURAS41_kaksoset','Kotimittaus_VAURAS42_kaksoset','Kotimittaus_VAURAS43',
'Kotimittaus_VAURAS46','Kotimittaus_VAURAS47','Kotimittaus_VAURAS51','Kotimittaus_VAURAS52',
'Kotimittaus_VAURAS53','Kotimittaus_VAURAS63','Kotimittaus_VV54','Kotimittaus_VV55',
'Kotimittaus_VV_xx','Kotimittaus_pilot1','Kotimittaus_pilot2','baby10','baby11','baby12',
'baby13','baby14','baby15','baby16','baby17','baby18','baby19','baby20','baby21','baby22',
'baby23','baby24','baby25','baby26','baby3','baby4','baby5','baby6','baby7','baby8','baby9']
   #folder = 'ClassifierlabeledPreprocess_data/'
   samples = _CONF.testTrainRatio*len(babies)
   f.write("tf.test.is_gpu_available()" + str(tf.test.is_gpu_available())+"\n")
   f.write("GPU device :" + tf.test.gpu_device_name()+"\n")
   f.write("TF version is " + tf.__version__ +"\n")
   half_baby = len(babies)//2
   test_babydict = {}
   macro_avg = [-1]*_CONF.folds
   np.random.seed(42)
   np.random.shuffle(babies)#,seed = 42)
   babies = babies[:half_baby]
   folds = _CONF.folds
   test_size = int(len(babies)/folds)


   for k in range(_CONF.folds):
    f.write("\n\n Fold is "+ str(k))
    if k+1 != folds:
           test_babies = babies[k*test_size : (k+1)*test_size]
    else:
           test_babies = babies[k*test_size : ]
    print("test_babies ", test_babies)
    train_babies = np.setdiff1d(babies, test_babies)
    
    test_babydict[k] = test_babies
    data_folder ='labeled_DATA/'
    #fileName_test='TestlabeledInputfileNames.mat'#la.getlabeledFileNames(babies,folder = 'Test_Data/',data_folder = data_folder,test = True)
    #fileName_train='TrainlabeledInputfileNames.mat'#la.getlabeledFileNames(babies,folder = 'Train_Data/',data_folder = data_folder,test = False)
    fileName_test=la.getlabeledFileNames(test_babies,folder = 'Test_Data/',data_folder = data_folder,test = True)
    fileName_train=la.getlabeledFileNames(train_babies,folder = 'Train_Data/',data_folder = data_folder,test = False)  
    folder1 = ''
    
    file_names = io.loadmat(fileName_train)['Input_fileNames']
    filenames_shuffled = shuffle(file_names)
    #np.save(folder + 'labeledShuffledInputfileNames.npy',filenames_shuffled)#{'filenames_shuffled':filenames_shuffled})
    #X_train_filenames, X_val_filenames = train_test_split(filenames_shuffled, test_size=_CONF.testTrainRatio, random_state=1)
    
    file_names2 = io.loadmat(fileName_test)['Input_fileNames']
    filenames_shuffled2 = shuffle(file_names2)
    #np.save(folder + 'labeledTestShuffledInputfileNames.npy',filenames_shuffled2)#{'filenames_shuffled':filenames_shuffled})
    DG_test = dg.My_ClassifierCustom_Generator(filenames_shuffled2,_CONF.batch_size,training = False)


    #DG_train = dg.My_ClassifierCustom_Generator(X_train_filenames,_CONF.batch_size)
    DG_train = dg.My_ClassifierCustom_Generator(filenames_shuffled,_CONF.batch_size)
    #DG_valid = dg.My_ClassifierCustom_Generator(X_val_filenames,_CONF.batch_size,training = False)
    code_size = _CONF.Nlatent2
    x_r = tf.keras.Input(shape = (24,120))
    
    sensor_module3 = modelcpc.SENSOR_MODULE3('sensor_module', s_channels=24, latent_channels=_CONF.Nlatent//2, output_channels=_CONF.Nlatent, input_channels=_CONF.winlen, dropout_rate=0.3 )
  
    
    timeseries_model = modelcpc.WaveNet("wavenet", residual_channels=_CONF.Nlatent2, output_channels=_CONF.NcatsB, input_channels=_CONF.Nlatent, 
                postproc_channels=_CONF.Nlatent2, dilations=_CONF.timeseries_channels, filter_width=_CONF.filter_width, dropout_rate=0.3)
 
    sensor_enc =sensor_module3(x_r,training = True)
    logit_activations = timeseries_model(sensor_enc,training=True)
        # Adam optimizer
    if _CONF.use_lr_decay:
        learning_rate = tf.train.exponential_decay(_CONF.learning_rate, global_step, 
                                           1000, 0.96, staircase=True)
    else:
        learning_rate = _CONF.learning_rate
    wc = loss_classifier()   
    model = tf.keras.Model(inputs = [x_r], outputs = [logit_activations], name='supervised')
   
    f.write("model summary is \n" +str(model.summary()) +"\n\n")
    epochs = _CONF.epochs
    train_loss_results = []
    train_accuracy_results = []
    mask_var = tf.Variable(np.zeros((_CONF.frames_per_sample)))
    weight_var =  tf.Variable(np.zeros((_CONF.frames_per_sample)))
   
    batch_size1 = _CONF.frames_per_sample
    batches_list = np.arange(len(DG_train)) 
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate) 
    
    best_epoch = 0
    for e in range(epochs):
        loss_epoch = 0.0
        acc_epoch = 0.0 
        
        batches_shuffled = shuffle(batches_list)      
        for s in batches_shuffled:
          
           x,output_var,weight_var1,bmask_ = DG_train[s]
           x = tf.squeeze(x)
           
           x = tf.transpose(x, perm=[0,2,1]) 
           output_var = tf.squeeze(output_var)
      
           weight_var1 = tf.squeeze(weight_var1)
           bmask_ = tf.squeeze(bmask_)
           if tf.math.reduce_sum(bmask_) > 0.95*batch_size1:
                    continue
      
           mask_var.assign(bmask_)
           weight_var.assign(weight_var1)
           bmask = tf.cast(1-mask_var,tf.bool)
           wc.updateMaskWeight(weight_var,bmask)
       
           with tf.GradientTape() as tape:
                preds =(model)(x,training = True) 
                loss_value = wc.loss_classifier1(output_var,preds)
          
                gradients = tape.gradient(loss_value ,model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0) 
                
                opt.apply_gradients(zip(gradients, model.trainable_variables))

           batchLoss,batchAccuracy = wc.getLossAccuracy() 
           if not tf.math.is_nan(batchLoss) :
                 loss_epoch = loss_epoch + batchLoss
           if not tf.math.is_nan(batchAccuracy) :
                 acc_epoch = acc_epoch + batchAccuracy   
         
        loss_epoch = loss_epoch/len(DG_train)
        train_loss_results.append(loss_epoch)

                
        acc_epoch = acc_epoch/(len(DG_train))
        train_accuracy_results.append(acc_epoch)
        tf.print("Tensors", loss_epoch,acc_epoch, output_stream=sys.stdout)
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(e,
                                                                loss_epoch,
                                                                acc_epoch))
        if e % 99 == 0:
             

              y_pred  = model.predict(DG_test)  
              predicted_categories = tf.argmax(y_pred, axis=1)
              y_true = tf.concat([y for x, y,z in DG_test], axis=0)
              y_mask = tf.concat([z for x, y,z in DG_test], axis=0)
              mask = tf.cast(1-y_mask,tf.bool)
              true_categories = tf.argmax(y_true, axis=1)
              #M1 = tf.math.confusion_matrix(predicted_categories, true_categories,num_classes= _CONF.Ncats)
              M2 = tf.math.confusion_matrix(predicted_categories[mask], true_categories[mask],num_classes= _CONF.Ncats)
             
              target_names = [i for i in np.arange(_CONF.Ncats)]
              #classification_report1 = classification_report(true_categories,predicted_categories,target_names= target_names)
              classification_report2 = classification_report(true_categories[mask],predicted_categories[mask],labels= target_names)   
              macro_avgvalues  = precision_recall_fscore_support(true_categories[mask],predicted_categories[mask],average = 'macro')
              f.write("Macro average prec,rec,fsore,None is " + str(macro_avgvalues))
              macro_avgvalue = macro_avgvalues[2]
              print("macro_avgvalue is ",  macro_avgvalue , "  and the actual macro_avgvalues is ",macro_avgvalues) 
              if e == 0:
                   macro_avg[k] = macro_avgvalue
              else:
                   if macro_avg[k] < macro_avgvalue: 
                          macro_avg[k] = macro_avgvalue
                          best_epoch = e  
                          model.save(join(model_dir, 'supervised_kfold_' + str(k) +'_.h5'))
              
              #f.write("The confusion matrix is for epoch "+str(e)+ " is "+"\n" + str(M1)+"\n")
              f.write("The confusion matrix 2 with mask for epoch "+str(e)+ " is "+"\n" + str(M2)+"\n")
              #f.write("classification_report :\n"+classification_report1+"\n")
              f.write("classification_report 2 with mask :\n"+classification_report2+"\n\n\n")
              
    io.savemat(model_dir+'train_loss_results'+str(k)+'.mat', {'train_loss_results': train_loss_results})
    io.savemat(model_dir+'train_accuracy_results'+str(k)+'.mat', {'train_accuracy_results': train_accuracy_results})


    model.save(join(model_dir, 'Finalsupervised_fold_'+str(k)+'.h5'))
    
    f.write("\n\nThe best performing model for fold "+ str(k) + "is at epoch "+ str(best_epoch))
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
  
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()
    plt.savefig(model_dir + 'fold_'+str(k)+'_TraininLossAccuracy.png')
   print(macro_avg, " is F-scores of foldss and average is ", np.mean(macro_avg), "and SD is :",np.std(macro_avg))   
   idx_k = macro_avg.index(max(macro_avg))
   final_testbabies = test_babydict.get(idx_k)
   f.write("The best performing model is index of kfold," + str(idx_k) + " and the tets babies used are " + str(final_testbabies) + 'Finalsupervised_fold_'+str(idx_k)+'.h5' )
   f.close()
   model1 = keras.models.load_model(join(model_dir, 'Finalsupervised_fold_'+str(idx_k)+'.h5'),custom_objects=dependencies)
   model1.save(join(model_dir, 'supervised.h5'))   
   np.save(model_dir+'Supervised_testbabies.npy', final_testbabies)
   print("SUCCESS")
   return
       
if __name__ == "__main__":
    result_dir = 'supervised_results/half_num_baby/'
    for i in _CONF.Ntrack:
        res_dir = result_dir + i + '/'
        if not os.path.isdir(res_dir):
              os.makedirs(res_dir,exist_ok=True)
                
        train_model(res_dir)
