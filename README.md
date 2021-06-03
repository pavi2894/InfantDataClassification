# InfantDataClassification
Self supervised(CPC) and Supervised learning code for infant data classification task


Supervised training : 
 TrainV2.py  for training 
 
 
CPC
python selfsupervisedtrainV2.py Selfsupervised/self*.yaml   for CPC training 

python SS_classifier.py  SVMyamls/SVMyaml*.yaml  for CNN and wavenet supervised classification 
python SS_classifier_NonTemporal.py  SVMyamls/SVMyaml*.yaml  for softmax,deepdense,Dense2Softmax classifiers
python SS_classifier__Enc_AR_freeze.py SVMyamls/SVMyaml*.yaml  for Enc+AR model freezed together to learn the encoded model and softmax,deepdense,Dense2Softmax classifiers

python CPC_TSNEExperiment.py TSNEyamls/tsne*.yaml for generating the TSNe plots
//TODO
