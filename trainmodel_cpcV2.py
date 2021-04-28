

import keras
import tensorflow as tf
#import tensorflow.keras.backend as K
from keras import backend as K
import cleaned_config as _CONF

import numpy as np
#def checkTensor(x):
   #  if dtype(x) == 'float64':
    #	x = tf.cast(x, 'float32') 
     #return x
def CONVNET1D_(code_size =256):
     convnet_model = keras.Sequential([
     keras.Input(shape=(_CONF.winlen,_CONF.channels)),
     tf.keras.layers.Conv1D(filters = 32, kernel_size = 5,strides = 2,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Conv1D(filters  =32, kernel_size = 5,strides = 2,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Conv1D(filters = 64, kernel_size = 5,strides = 2,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Conv1D(filters= 64, kernel_size = 5,strides = 2,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     keras.layers.Flatten(),
     #keras.layers.Dense(units=256, activation='sigmoid'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     keras.layers.Dense(units=code_size, activation='sigmoid', name='encoder_embedding1')],name = 'enc_model_convnet1D')

     return convnet_model



def CONVNET1D_2(code_size =256):
     convnet_model = keras.Sequential([
     keras.Input(shape=(_CONF.winlen,_CONF.channels)),
     
     tf.keras.layers.Conv1D(filters = 64, kernel_size = 3,strides = 1,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Conv1D(filters  =64, kernel_size = 3,strides = 1,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Conv1D(filters = 128, kernel_size = 3,strides = 1,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Conv1D(filters= 128, kernel_size = 3,strides = 1,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     keras.layers.Flatten(),
     #keras.layers.Dense(units=256, activation='sigmoid'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     keras.layers.Dense(units=code_size, activation='sigmoid', name='encoder_embedding1')],name = 'enc_model_convnet1D')

     return convnet_model


def CONV2D_1a():
     
     convnet_model = keras.Sequential([
     keras.Input(shape=(_CONF.winlen,_CONF.channels//2,1)),
    
     tf.keras.layers.Conv2D(filters = 80, kernel_size = (11,3),strides = (5,3),padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Conv2D(filters  = 80,kernel_size = (5,4),strides = (2,1),padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.AveragePooling2D(pool_size=(12,1),padding="valid"),
     tf.keras.layers.Dropout(rate = 0.3)])
    
     return convnet_model


def CONV2D_1(code_size =256):
     
     convnet_model1 = CONV2D_1a()
     convnet_model2 = CONV2D_1a()
     return convnet_model1,convnet_model2

def CONV2D_1b(code_size = 256):

     #print("input.shape[1],input.shape[2]",input.shape)
     convnet_model = keras.Sequential([#keras.Input(shape=(input.shape)),
     keras.Input(shape=(8,80)),
     tf.keras.layers.Conv1D(code_size, 4, strides = 1),
     tf.keras.layers.AveragePooling1D(pool_size=4,strides =4),
     #tf.keras.layers.Conv1D(code_size, 4, strides = 1,activation='relu',padding='VALID',data_format = 'channels_first'),
     tf.keras.layers.Dropout(rate = 0.3),
     keras.layers.Flatten(),
     keras.layers.Dense(units=code_size, activation='sigmoid', name='encoder_embedding1')])
     return convnet_model



def CONV2D_SIa(code_size = 256):
     
     convnet_model = keras.Sequential([
     keras.Input(shape=(_CONF.winlen,3,1)),
    
     tf.keras.layers.Conv2D(filters = 80, kernel_size = (11,3),strides = (5,3),padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Lambda(lambda x: K.squeeze(x, -2)),
     tf.keras.layers.Conv1D(filters  = 80,kernel_size = 5,strides = 2,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.AveragePooling1D(pool_size=3,padding="valid")])

     return convnet_model

def CONV2D_SIb(code_size = 256):
     
     convnet_model = keras.Sequential([
     keras.Input(shape=(_CONF.winlen,6,1)),
    
     tf.keras.layers.Conv2D(filters = 80, kernel_size = (11,3),strides = (5,6),padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.Lambda(lambda x: K.squeeze(x, -2)),
     tf.keras.layers.Conv1D(filters  = 80,kernel_size = 5,strides = 2,padding= 'SAME'),
     #keras.layers.BatchNormalization(),
     #keras.layers.LeakyReLU(),
     tf.keras.layers.AveragePooling1D(pool_size=3,padding="valid")])

     return convnet_model

def Conv2D_SI(input,code_size = 64):
     convnet_model1 = CONV2D_SIa(code_size = code_size)
     output1 = tf.keras.layers.TimeDistributed(convnet_model1)(input[:,:,:,:3,:])
     convnet_model2 = CONV2D_SIa(code_size = code_size)
     output2 = tf.keras.layers.TimeDistributed(convnet_model2)(input[:,:,:,3:,:])
     convnet_model3 = CONV2D_SIb(code_size = code_size)
     output3 = tf.keras.layers.TimeDistributed(convnet_model3)(input)
     print(output1.shape,output2.shape,output3.shape,"SHAPE S ARE AS")
     output4 = tf.keras.layers.Concatenate(axis=2)([output1,output2,output3])
     print(output4.shape)
     return output4

def Conv2D_SIlsyer(code_size = 256) : 
     convlayer =  keras.Sequential([tf.keras.layers.Conv1D(filters = 64, kernel_size = 3,strides = 1,padding= 'SAME'),
     keras.layers.Flatten(),
     keras.layers.Dense(units=code_size)])
     return convlayer


class Resblock(tf.keras.layers.Layer):
     def __init__(self, channels, kernel_size, dilation_rate,res_op = True, **kwargs):
          super(Resblock, self).__init__(name='')
        
          self.conv1da = keras.layers.Conv1D(filters = channels, kernel_size = kernel_size,dilation_rate = dilation_rate,padding= 'SAME')
          self.conv1db = keras.layers.Conv1D(filters = channels, kernel_size = kernel_size,dilation_rate = dilation_rate,padding= 'SAME')
          self.channels = channels
          self.res_op = res_op
          self.kernel_size = kernel_size
          self.dilation_rate = dilation_rate
          self.out1 = keras.layers.Conv1D(filters = channels, kernel_size = 1)
          self.out2 = keras.layers.Conv1D(filters = channels, kernel_size = 1)
     def get_config(self):
        config = super().get_config().copy()
        config.update({
        'channels' : self.channels,
        'dilation_rate' : self.dilation_rate,
        'kernel_size': self.kernel_size,
        'res_op' :self.res_op
        })
        return config     
     def call(self,x_input,training= False):
          x_1 = self.conv1da(x_input)
          #x_1 = keras.layers.BatchNormalization()(x_1)
          x_1 = tf.keras.activations.tanh(x_1)
          x_2 = self.conv1db(x_input)
          x_2 = tf.keras.activations.sigmoid(x_2)
          #print("x_1,x_2",x_1.shape,x_2.shape)
          x   = tf.math.multiply(x_1 , x_2)
          skip_out = self.out2(x)
          #x_res = self.out1(x)
          if self.res_op == True:
              x_res = self.out1(x)
              res_out = x_res + x_input
              return res_out,skip_out
          else :
              return skip_out 



class WaveNet(tf.keras.layers.Layer): # Input: (Nframes, input_channels), Output: (Nframes, output_channels)
    def __init__(self,name, residual_channels=128, output_channels=_CONF.NcatsB, input_channels=128, 
                postproc_channels=64, dilations=_CONF.timeseries_channels, filter_width=5, dropout_rate=0.3, **kwargs):
        super(WaveNet, self).__init__(name=name)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_channels = residual_channels
        self.postproc_channels = postproc_channels
        self.filter_width = filter_width
        self.dilations = dilations
        #self._name = name
        self.dropout_rate = dropout_rate
        #self.conv1d = tf.keras.layers.Conv1D(filters = residual_channels, kernel_size = 1,padding= 'SAME')
        self.conv1d = tf.keras.layers.Conv1D(filters = residual_channels, kernel_size = 1,padding= 'SAME')
      
        self.conv1da = tf.keras.layers.Conv1D(filters = postproc_channels, kernel_size = filter_width,padding= 'SAME',activation='relu')

        self.conv1db = tf.keras.layers.Conv1D(filters = output_channels, kernel_size = 1,padding= 'SAME')
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.res1 = Resblock(channels=self.residual_channels, kernel_size  = self.filter_width, dilation_rate = 1)
        self.res2 =  Resblock(channels=self.residual_channels, kernel_size  = self.filter_width, dilation_rate =2)
        self.res3 =  Resblock(channels=self.residual_channels, kernel_size  = self.filter_width, dilation_rate =4, res_op = False)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'input_channels' : self.input_channels,
        'output_channels' : self.output_channels,
        'residual_channels' : self.residual_channels ,
        'postproc_channels': self.postproc_channels,
        'filter_width': self.filter_width ,
        'dilations':self.dilations,
        'dropout_rate' : self.dropout_rate
        })
        return config    

    def call(self, X_input, training=False):
        skip_outputs = []
        X_input = self.dropout_layer(X_input,training=training)
        
        X = tf.expand_dims(X_input,axis=0)
        X = self.conv1d(X)
        X = tf.keras.activations.tanh(X)
        R = X
        X, skip = self.res1(X,training = True)
        skip_outputs.append(skip)
        X, skip = self.res2(X,training = True)
        skip_outputs.append(skip)
        skip = self.res3(X,training = True) 
        skip_outputs.append(skip)

        Y = tf.zeros_like(X)
        for S in skip_outputs:
                Y += S

        enc = self.conv1da(Y)
        Z = self.conv1db(enc)
            #Z = tf.squeeze(Z)
        enc = tf.reshape(enc, [-1, self.postproc_channels])
        Z = tf.reshape(Z,[-1,self.output_channels])
        return Z#, enc


class sensorEncoderModule(tf.keras.layers.Layer):
     def __init__(self,latent_channels,input_channels, **kwargs):
        super(sensorEncoderModule, self).__init__(name='')
        self.latent_channels = latent_channels
        self.input_channels = input_channels  
        self.conv2da_acc = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (3,11), padding='VALID', strides=[3,5])
        self.conv2db_acc = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (1,5), padding='SAME', strides=[1,2])  
        self.conv2da_gyro = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (3,11), padding='VALID', strides=[3,5])
        self.conv2db_gyro = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (1,5), padding='SAME', strides=[1,2])
        self.conv2da_ag = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (3,11), padding='VALID', strides=[3,5])
        self.conv2db_ag = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (1,5), padding='SAME', strides=[1,2])
        self.relu  = tf.keras.layers.LeakyReLU()
        self.r_dims = np.int32(np.ceil(input_channels/10)) -1
     def get_config(self):
        config = super().get_config().copy()
        config.update({
        'input_channels' : self.input_channels,
        'latent_channels' : self.latent_channels 
        })
        return config 
     def call(self,X1,X2,X3,training=False):
          #print(X1.shape,"X1 shape is ")
          X1 = tf.cast(X1, 'float32')
          X2 = tf.cast(X2, 'float32')
          X3 = tf.cast(X3, 'float32')
          X1 =self.conv2da_acc(X1)
          X1 = tf.keras.activations.tanh(X1)
          #print(X1.shape,"X1 shape is ")
          X1 = self.conv2db_acc(X1)
          #print(X1.shape,"X1 shape is ")
          X1 = self.relu(X1)
          
          X1 = tf.nn.pool(X1, window_shape=[1, self.r_dims],pooling_type='AVG',padding='VALID')
          X1 = tf.squeeze(X1,axis = -2) 

          X2 = self.conv2da_gyro(X2)
          X2 = tf.keras.activations.tanh(X2)
          X2 = self.conv2db_gyro(X2)
          X2 = self.relu(X2)
          X2 = tf.nn.pool(X2, window_shape=[1, self.r_dims],pooling_type='AVG',padding='VALID')          
#X2 =  tf:.keras.layers.AveragePooling2D( pool_size=[1, self.r_dims],padding='valid')(X2)# Output: (Nframes, x, 1, l)
          X2 = tf.squeeze(X2,axis = -2) 
          X3 = self.conv2da_ag(X3)
          X3 = tf.keras.activations.tanh(X3)
          X3 = self.conv2db_ag(X3)
          X3 = self.relu(X3)
          X3 = tf.nn.pool(X3, window_shape=[1, self.r_dims],pooling_type='AVG',padding='VALID') # Output: (Nframes, x, 1, l)
          X3 = tf.squeeze(X3,axis = -2) 
          return X1,X2,X3


class sensorEncoderModule1(tf.keras.layers.Layer):
     def __init__(self,latent_channels,input_channels, **kwargs):
        super(sensorEncoderModule1, self).__init__(name='')
        self.latent_channels = latent_channels
        self.input_channels = input_channels
        self.conv2da_acc = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (3,11), padding='VALID', strides=[3,5])
        self.conv2db_acc = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (4,5), padding='SAME', strides=[1,2])
        self.conv2da_gyro = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (3,11), padding='VALID', strides=[3,5])
        self.conv2db_gyro = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (4,5), padding='SAME', strides=[1,2])
        self.conv2da_ag = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (3,11), padding='VALID', strides=[3,5])
        self.conv2db_ag = tf.keras.layers.Conv2D(filters = self.latent_channels,kernel_size= (4,5), padding='SAME', strides=[1,2])
        self.relu  = tf.keras.layers.LeakyReLU()
        self.r_dims = np.int32(np.ceil(input_channels/10)) -1
     
     def get_config(self):
        config = super().get_config().copy()
        config.update({
        'input_channels' : self.input_channels,
        'latent_channels' : self.latent_channels
        })
        return config
     def call(self,X1,X2,X3,training=False):
         
          X1 = tf.cast(X1, 'float32')
          X2 = tf.cast(X2, 'float32')
          X3 = tf.cast(X3, 'float32')
          X1 =self.conv2da_acc(X1)
          #X1 = keras.layers.BatchNormalization()(X1)
          X1 = tf.keras.activations.tanh(X1)
          
          X1 = self.conv2db_acc(X1)
          #X1 = keras.layers.BatchNormalization()(X1)
          X1 = self.relu(X1)

          X1 = tf.nn.pool(X1, window_shape=[1, self.r_dims],pooling_type='AVG',padding='VALID')
          
          
          X1 = tf.squeeze(X1,axis = -2)
         
          X2 = self.conv2da_gyro(X2)
          #X2 = keras.layers.BatchNormalization()(X2)
          X2 = tf.keras.activations.tanh(X2)
          X2 = self.conv2db_gyro(X2)
          #X2 = keras.layers.BatchNormalization()(X2)
          X2 = self.relu(X2)
          X2 = tf.nn.pool(X2, window_shape=[1, self.r_dims],pooling_type='AVG',padding='VALID')

          X2 = tf.squeeze(X2,axis = -2)
          X3 = self.conv2da_ag(X3)
          #X3 = keras.layers.BatchNormalization()(X3)
          X3 = tf.keras.activations.tanh(X3)
          X3 = self.conv2db_ag(X3)
          #X3 = keras.layers.BatchNormalization()(X3)
          X3 = self.relu(X3)
          
          X3 = tf.nn.pool(X3, window_shape=[1, self.r_dims],pooling_type='AVG',padding='VALID') 
          X3 = tf.squeeze(X3,axis = -2)
          
          return X1,X2,X3


class SENSOR_MODULE1(tf.keras.layers.Layer):  

    def __init__(self,name, s_channels=24, latent_channels=16, output_channels=128, input_channels=256, dropout_rate=0.3, **kwargs ):
        super(SENSOR_MODULE1, self).__init__(name='')
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.s_channels =  s_channels
        self.latent_channels = latent_channels
        self.r = np.int32(s_channels/2)
        self.r_dims = np.int32(np.ceil(self.input_channels/10)) -1
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.sensormoduleLayer = sensorEncoderModule1(latent_channels,input_channels)
        self.conv1d = tf.keras.layers.Conv1D(filters = output_channels,kernel_size= 4,padding='SAME') 
        self.conv1db = tf.keras.layers.Conv1D(filters = output_channels,kernel_size= 4,padding='VALID')
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'input_channels' : self.input_channels,
        'output_channels' : self.output_channels,
        'latent_channels' : self.latent_channels ,
        'dropout_rate' : self.dropout_rate,
        's_channels' : self.s_channels
        })
        return config
  
    def call(self, X_input, training=False):
        X_input = self.dropout_layer(X_input,training=training)
        X_input = tf.expand_dims(X_input, axis=-1)
        X1 = X_input[:,:self.r,:,:] # Acc data
        X2 = X_input[:,self.r:,:,:] # Gyro data
        X1,X2,X3 = self.sensormoduleLayer(X1,X2,X_input,training )
        X_f = tf.concat([X1,X2,X3],axis=1)
        X_f = self.conv1d(X_f) 
        Y = tf.keras.layers.LeakyReLU(alpha=0.3)(X_f)
        Y = tf.nn.pool(Y, window_shape=[4],strides=[4],pooling_type='AVG',padding='VALID')
        Y = self.conv1db(Y)
        Y = tf.keras.layers.LeakyReLU(alpha=0.3)(Y)
        
        Y = tf.reshape(Y,[-1,self.output_channels])
        return Y





class SENSOR_MODULE3(tf.keras.layers.Layer):  #output : (150, 160)
    def __init__(self,name, s_channels=24, latent_channels=16, output_channels=128, input_channels=256, dropout_rate=0.3, **kwargs ):
        super(SENSOR_MODULE3, self).__init__(name=name)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_channels = latent_channels
        self.r = np.int32(s_channels/2)
        self.r_dims = np.int32(np.ceil(self.input_channels/10)) -1
        #self._name = name
        self.dropout_rate = dropout_rate
        self.s_channels = s_channels
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.sensormoduleLayer = sensorEncoderModule(latent_channels,input_channels)
        self.conv1d = tf.keras.layers.Conv1D(filters = output_channels//4,kernel_size= 4,padding='VALID',strides=4) 
        self.conv1db = tf.keras.layers.Conv1D(filters = output_channels//4,kernel_size= 4,padding='SAME')
        self.out1 = tf.keras.layers.Conv1D(filters= output_channels, kernel_size = 5,strides = 1,padding= 'SAME')
        self.out2 = tf.keras.layers.Conv1D(filters= output_channels, kernel_size = 5,strides = 1,padding= 'SAME')
                
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'input_channels' : self.input_channels,
        'output_channels' : self.output_channels, 
        'latent_channels' : self.latent_channels ,
        'dropout_rate' : self.dropout_rate,
        's_channels' : self.s_channels 
        })
        return config        
    def call(self, X_input, training=False):
        X_input = self.dropout_layer(X_input,training=training)
      
        X_input = tf.expand_dims(X_input, axis=-1)
        X1 = X_input[:,:self.r,:,:] # Acc data
        X2 = X_input[:,self.r:,:,:] # Gyro data
        X1,X2,X3 = self.sensormoduleLayer(X1,X2,X_input,training )
        X_f = tf.concat([X1,X2,X3],axis=1) # -> (Nframes, 16, l)
        idx = (0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15)           
        o = self.output_channels    
        X_f = tf.gather(X_f,idx,axis = 1)
        X_f = self.conv1d(X_f)
        Y = tf.keras.layers.LeakyReLU(alpha=0.3)(X_f)
        Y = self.conv1db(Y)
        Y = tf.keras.layers.LeakyReLU(alpha=0.3)(Y)
        Y = tf.reshape(Y,[-1,self.output_channels])
        return Y
