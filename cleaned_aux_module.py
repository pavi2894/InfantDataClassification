import numpy as np
import os
import tensorflow as tf

import cleaned_config as _CONF

from scipy import stats

class Recording():
    def __init__(self,name, sigdata, labdata_A, labdata_B, 
                train_mask_a, train_mask_b, test_mask_b=None,
                test_mask_a=None,timestamp = None):

        self.X = sigdata
        self.T = timestamp
        self.logits_A = labdata_A
        self.logits_B = labdata_B
        self.TrainMask_A = train_mask_a
        self.TrainMask_B = train_mask_b


        if test_mask_b is None:
            self.TestMaskB = train_mask_b
        else:
            self.TestMaskB = test_mask_b
    
        if test_mask_a is None:
            self.TestMaskA = train_mask_a
        else:
            self.TestMaskA = test_mask_a

        weight_type = _CONF.weight_type
        self.TrainWeightA = get_train_weights(self.logits_A,'A',weight_type=weight_type)
        self.TrainWeightB = get_train_weights(self.logits_B,'B',weight_type=weight_type)

        self.winlen = sigdata.shape[1]
        self.hop = 60
        self.NcatsA = self.logits_A.shape[1]
        self.NcatsB = self.logits_B.shape[1]
        self.Nframes = self.X.shape[0]
        self.name = name


        self.set_track('B')

    def set_track(self,track='A'):
        if track == 'A':
            self.Y = self.logits_A
            self.Y_alt = self.logits_B
            self.TrainWeight = self.TrainWeightA
            self.TestMask = self.TestMaskA
            self.TrainMask = self.TrainMask_A

        elif track == 'B':
            self.Y = self.logits_B
            self.Y_alt = self.logits_B
            self.TrainWeight = self.TrainWeightB
            self.TestMask = self.TestMaskB
            self.TrainMask = self.TrainMask_B


def get_train_weights( onehot_mat, cat_weights=None, weight_type='sum_norm'):
    
        W = np.ones([onehot_mat.shape[0]],dtype=np.float32)
      
        weight_type = 'sum_norm' 
        if weight_type == 'ones':
            A_prior = np.asarray([1, 1, 1, 1, 1],dtype=np.float32)
            B_prior = np.asarray([1, 1, 1, 1, 1, 1, 1],dtype=np.float32)
        else: # Prior distribution probabilities
            # A: p, s, l, r, c
            #A_prior = np.asarray([0.6251, 0.2671, 0.0286,0.286, 0.0506],dtype=np.float32) # 
            # B: still, t_l, t_r, p_l, p_r, c_c, c_p]
            #B_prior = np.asarray([0.5956, 0.01415, 0.01415, 0.0294, 0.0294, 0.0313, 0.2860],dtype=np.float32)
            
            B_prior = np.asarray([0.5144, 0.0082, 0.0077, 0.0124, 0.0136, 0.2601, 0.0713, 0.0616, 0.0508],dtype=np.float32)

            A_prior = np.asarray([0.316200554295792,0.116301335348954,0.0148652053413958,0.0140589569160998,0.119198790627362,0.236470143613001,0.182905013857395],dtype=np.float32)


        if cat_weights=='A':
            cat_weights = 1.0/A_prior
        elif cat_weights == 'B':
            cat_weights = 1.0/B_prior
        elif cat_weights == 'IAR':
            cat_weights = np.asarray([1.0, 1.0],dtype=np.float32)

        if weight_type == 'sum_norm':
            cat_weights /= np.sum(cat_weights)
        elif weight_type == 'mean_norm':
            cat_weights /= np.mean(cat_weights)

        #import ipdb;ipdb.set_trace()
        for i in range(W.shape[0]):
            category_weight = np.sum(onehot_mat[i,:]*cat_weights)
            W[i] *= category_weight

        return W
        
class UnlabeledRecording():
    def __init__(self,name, sigdata):
        self.X = sigdata
        self.name = name


def preprocess_data(data):
                    frames_per_sample = _CONF.frames_per_sample
            
                    Nframe = data.shape[0]
          
                    data_sample_len = Nframe//frames_per_sample
                    data_samples = np.empty(shape=(0,_CONF.frames_per_sample,_CONF.winlen,_CONF.channels))
                    for i in range(data_sample_len):
                        if i != data_sample_len-1:
                            ds = np.transpose(np.expand_dims(np.array(data[i*frames_per_sample:(i+1)*frames_per_sample,:,:]),axis=0),[0,1,3,2])
                            data_samples = np.concatenate((data_samples, ds),axis = 0)
                        #else:
                        #ds = np.transpose(np.expand_dims(np.array(data[i*frames_per_sample:,:,:]),axis=0),[0,1,3,2])
                        #    data_samples = np.concatenate((data_samples,ds),axis=0)
                          
                    return data_samples

def preprocess_dataY(data,cat = 'A'):
                    if(cat== 'A'):
                        size_ = _CONF.NcatsA
                    else:
                        size_ = _CONF.NcatsB 
                    frames_per_sample = _CONF.frames_per_sample
            
                    Nframe = data.shape[0]
          
                    data_sample_len = Nframe//frames_per_sample
                    data_labels = np.empty(shape=(0,frames_per_sample,size_))
                    for i in range(data_sample_len):
                        if i != data_sample_len-1:
                            dl = np.expand_dims(np.array(data[i*frames_per_sample:(i+1)*frames_per_sample,:]),axis=0)
                            
                            data_labels = np.concatenate((data_labels, dl),axis = 0)
                           
                        #else:
                        #ds = np.transpose(np.expand_dims(np.array(data[i*frames_per_sample:,:,:]),axis=0),[0,1,3,2])
                        #    data_samples = np.concatenate((data_samples,ds),axis=0)
                          
                    return data_labels




def preprocess_dataweight(data):
                    frames_per_sample = _CONF.frames_per_sample

                    Nframe = data.shape[0]

                    data_sample_len = Nframe//frames_per_sample
                    data_weights = np.empty(shape=(0,frames_per_sample))
                    for i in range(data_sample_len):
                        if i != data_sample_len-1:
                            dl = np.expand_dims(np.array(data[i*frames_per_sample:(i+1)*frames_per_sample]),axis=0)

                            data_weights = np.concatenate((data_weights, dl),axis = 0)

                        #else:
                        #ds = np.transpose(np.expand_dims(np.array(data[i*frames_per_sample:,:,:]),axis=0),[0,1,3,2])
                        #    data_samples = np.concatenate((data_samples,ds),axis=0)

                    return data_weights

def preprocess_datamask(data):
                    frames_per_sample = _CONF.frames_per_sample

                    Nframe = data.shape[0]

                    data_sample_len = Nframe//frames_per_sample
                    data_mask = np.empty(shape=(frames_per_sample,0))
                    for i in range(data_sample_len):
                        if i != data_sample_len-1:
                            dl = np.array(data[i*frames_per_sample:(i+1)*frames_per_sample])
                            #print("dl shape is ", dl.shape)
                            data_mask = np.concatenate((data_mask, dl),axis = 1)

                        #else:
                        #ds = np.transpose(np.expand_dims(np.array(data[i*frames_per_sample:,:,:]),axis=0),[0,1,3,2])
                                        #    data_samples = np.concatenate((data_samples,ds),axis=0)

                    return data_mask

def read_sensor_data(fname, numChansPerSensor=6, numSensors=4,timestamp=False,removenans=True,dtype=np.float32):
    D = np.fromfile(fname,dtype=dtype)
    if timestamp == False:
        D = np.reshape(D,[numChansPerSensor*numSensors,-1])
    else:
        D = np.reshape(D,[numChansPerSensor*numSensors + 1,-1])

    D = np.transpose(D)
    if removenans:
        D = D[:,~np.isnan(D[0,:])]

    if timestamp == False:
        return D
    else:
        t = D[:,-1]
        D = D[:,:-1]

        return D.astype(np.float32), t


def discretize_categories(Y):
    Y_d = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        vec = Y[i,:] + 0.01*np.random.rand(Y.shape[1])
        ind = np.argmax(vec)
        Y_d[i, ind] = 1.0
    return Y_d


def cat2onehot(Y,Ncats=None,use_unk=False):
    Nframes = np.int(Y.shape[0])
    if Ncats is None:
        Ncats = np.int(np.max(Y))
        
    if use_unk == False:
        Ncats -= 1

    onehot_mat = np.zeros([Nframes,Ncats],dtype=np.float32)

    for i in range(Nframes):
        val = np.int(Y[i])
        if val <= Ncats:
            onehot_mat[i,np.int(Y[i]-1)] = 1.0
    
    return onehot_mat

def c2oh(Y,Ncats):
    Nframes = np.int(Y.shape[0])
    Ncats = np.int(Ncats)
    onehot_mat = np.zeros([Nframes,Ncats],dtype=np.float32)
    #import ipdb;ipdb.set_trace()
    for i in range(Nframes):
        onehot_mat[i,np.int(Y[i])] = 1.0
    
    return onehot_mat

def onehot2cat(Y):
    Nframes = np.int(Y.shape[0])
    Ncats = np.int(Y.shape[1])

    cat_mat = np.zeros([Nframes,],dtype=np.float32)
    
    for i in range(Nframes):
        val = np.max(Y[i,:])
        if val > 0:
            cat_mat[i] = np.argmax(Y[i,:])
        else:
            cat_mat[i] = Ncats+1
    
    return cat_mat

def make_batches(X,Y,batch_size):
    Nbatches = X.shape[0] // batch_size

    X_batches = np.zeros([Nbatches,batch_size,X.shape[1],X.shape[2]],dtype=np.float32)
    Y_batches = np.zeros([Nbatches,batch_size,Y.shape[1]],dtype=np.float32)
    for i in range(Nbatches):
        x_tmp = X[(i)*batch_size:(i+1)*batch_size,:,:]
        x_tmp = np.expand_dims(x_tmp,axis=0)
        X_batches[i,:,:,:] = x_tmp
        y_tmp = np.expand_dims(Y[(i)*batch_size:(i+1)*batch_size,:],axis=0)
        Y_batches[i,:,:] = y_tmp

    return X_batches, Y_batches

def frame_sig(X, winlen, hop):
    Nframes = np.floor(((X.shape[0]-winlen)/hop)+1).astype(np.int)
    numchans = np.int32(X.shape[1])
    X_framed = np.zeros([Nframes,numchans,winlen],dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0,Nframes):
        start = np.int((i)*hop)
        stop = np.int(start+winlen)
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed

               

def split_train_test(Data, test_inds):
    D_train = []
    D_test = []

    # Select test subject(s)
    for i in range(len(Data)):
        if i in test_inds:
            D_test.append(Data[i])
        else:
            D_train.append(Data[i])

    return D_train, D_test

def frame_lab(Y, winlen, hop):
    if Y is None:
        return None

    Nframes = np.floor(((Y.shape[0]-winlen)/hop)+1).astype(np.int)
    Nannot = Y.shape[-1] # number of annotations
    Yf = np.zeros([Nframes,Nannot],dtype=np.float32)

    for i in range(0,Nframes):
        start = np.int((i)*hop)
        stop = np.int(start+winlen)

        val, _ = stats.mode(Y[start:stop,:])
        Yf[i,:] = val[0,:]

            
    return Yf
