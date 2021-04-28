
#!/usr/bin/env python3
from scipy import io
from sklearn.utils import shuffle
import classifier_dataGeneratorV2 as dg
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

dependencies = {
    'SENSOR_MODULE3' : modelcpc.SENSOR_MODULE3,
    'loss_classifier1': train.loss_classifier().loss_classifier1,
    'sensorEncoderModule':modelcpc.sensorEncoderModule,
    'WaveNet' : modelcpc.WaveNet,
    'Resblock' : modelcpc.Resblock
}

f = open("TestFile.txt", 'w')
result_dir = 'test-results2/'
model_dir = result_dir + _CONF.Ntrack[0] + '/'
test_babies = np.load('Supervised_testbabies.npy')
print(test_babies)
model = keras.models.load_model(join(model_dir, 'supervised.h5'),custom_objects=dependencies)
fileName_test=la.getlabeledFileNames(test_babies,folder = 'Test_Data/',data_folder = 'labeled_DATA/',test = True)
file_names2 = io.loadmat(fileName_test)['Input_fileNames']
filenames_shuffled2 = shuffle(file_names2)
DG_test = dg.My_ClassifierCustom_Generator(filenames_shuffled2,_CONF.batch_size,training = False)
y_pred  = model.predict(DG_test)

y_true = tf.concat([y for x, y,z in DG_test], axis=0)
y_mask = tf.concat([z for x, y,z in DG_test], axis=0)
predicted_categories = tf.argmax(y_pred, axis=1)
mask = tf.cast(1-y_mask,tf.bool)
true_categories = tf.argmax(y_true, axis=1)
M1 = tf.math.confusion_matrix(predicted_categories, true_categories,num_classes= _CONF.Ncats)
M2 = tf.math.confusion_matrix(predicted_categories[mask], true_categories[mask],num_classes= _CONF.Ncats)


target_names = ['class '+str(i) for i in np.arange(_CONF.Ncats)]

classification_report1 = classification_report(true_categories,predicted_categories,target_names= target_names)
classification_report2 = classification_report(true_categories[mask],predicted_categories[mask],target_names= target_names)
macro_avgvalues  = precision_recall_fscore_support(true_categories[mask],predicted_categories[mask],average = 'macro')

f.write("The confusion matrix is "+"\n" + str(M1)+"\n")
f.write("The confusion matrix 2 with mask is "+"\n" + str(M2)+"\n")
f.write("classification_report :\n"+classification_report1+"\n")
f.write("classification_report 2 with mask :\n"+classification_report2+"\n\n\n")
f.close()

XX = model.input 
YY = model.layers[1].output
model1 = tf.keras.Model(XX, YY)
del model

tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=250)

x_enc = model1.predict(DG_test)

del model1
tsne_cpc_train_results = tsne.fit_transform(x_enc[:1000])
tsne_1 = tsne_cpc_train_results[:,0]
tsne_2 = tsne_cpc_train_results[:,1]  
fig = plt.figure() 
ax1 = fig.add_subplot(121)
divider = make_axes_locatable(ax1)  
sc1 = ax1.scatter(tsne_1, tsne_2, c=true_categories[:1000], cmap=plt.cm.get_cmap("jet", _CONF.NcatsB))
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(sc1,ticks = range(_CONF.NcatsB),cax = cax,orientation='vertical')
ax2 = fig.add_subplot(122) 
divider = make_axes_locatable(ax2)
sc2 = ax2.scatter(tsne_1, tsne_2, c=true_categories[:1000], cmap=plt.cm.get_cmap("jet", _CONF.NcatsA))
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(sc2,ticks = range(_CONF.NcatsA),cax = cax,orientation='vertical')
plt.savefig('TSNE.png')
plt.show()
plt.close()

