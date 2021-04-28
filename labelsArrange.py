import cleaned_config as _CONF
from scipy import io
import cleaned_aux_module as sp
import numpy as np
import homedata_augmentation as aug

def getlabeledFileNames(babies,folder,data_folder,test = True,supervised = True):
  
  fileNames = []
  for iBaby in np.arange(len(babies)):

    acc_data = io.loadmat(data_folder + babies[iBaby] + "/acc_data.mat")['acc_data']
    
    gyro_data = io.loadmat(data_folder + babies[iBaby] + "/gyro_data.mat")['gyro_data']
    x_r = np.concatenate((acc_data/10.0, gyro_data/100.0),axis=1)
    x_r = sp.frame_sig(x_r,_CONF.winlen,_CONF.hop)
    if _CONF.use_augmentation:
        x_r = aug.augment(x_r) 
    #x_r = sp.preprocess_data(x_r)
    if supervised == True:
            movement_oh = io.loadmat(data_folder + babies[iBaby] + "/movement_oh.mat")['movement_oh'] 
            posture_oh = io.loadmat(data_folder + babies[iBaby] + "/posture_oh.mat")['posture_oh']
            data_mask = io.loadmat(data_folder + babies[iBaby] + "/mask.mat")['mask']
            data_mask = np.squeeze(data_mask)
            mask = (1-np.squeeze(data_mask)).astype(bool) 
            y_A = sp.discretize_categories(posture_oh)
            y_A = np.squeeze(y_A)[mask]
            y_B = sp.discretize_categories(movement_oh)
            train_weights = sp.get_train_weights(y_B,'B',weight_type = _CONF.weight_type) 
            y_B = np.squeeze(y_B)[mask]
            #train_weights = sp.get_train_weights(y_B,'B',weight_type = _CONF.weight_type)
            train_weights = np.squeeze(train_weights)[mask]
            x_r = np.squeeze(x_r)[mask]
            y_A = sp.preprocess_dataY(y_A,'A')
            y_B = sp.preprocess_dataY(y_B,'B')
            proc_train_weights = sp.preprocess_dataweight(train_weights)
            proc_data_mask = sp.preprocess_dataweight(data_mask) #np.reshape(data_mask,(-1,_CONF.frames_per_sample)) #sp.preprocess_dataweight(data_mask)
    
    x_r = sp.preprocess_data(x_r)       
    for i in range(x_r.shape[0]):
        x_r1 = x_r[i,:,:,:]   
        io.savemat(folder +str(i)+ babies[iBaby] + 'proc_inp.mat', {'x': x_r1})
        if supervised == True:
                y_B1 = y_B[i,:,:]
                io.savemat(folder +str(i)+ babies[iBaby] + 'Proc_mov_oh.mat', {'mov_oh': y_B1})
                y_A1 = y_A[i,:,:]
                io.savemat(folder +str(i)+ babies[iBaby]  + 'Proc_pos_oh.mat', {'pos_oh': y_A1})
                proc_train_weights1 = proc_train_weights[i,:]
                io.savemat(folder + str(i)+babies[iBaby]  + 'Proc_data_weight.mat',{'train_weights':proc_train_weights1})
                proc_data_mask1 = proc_data_mask[i,:]
                io.savemat(folder + str(i)+babies[iBaby]  + 'Proc_data_mask.mat',{'mask_weights':proc_data_mask1})
                print("Shapes are ",x_r1.shape,y_B1.shape,proc_train_weights1.shape,proc_data_mask1.shape)
        fileNames.append(folder +str(i)+ babies[iBaby] )
    print(len(fileNames))
  if supervised == True:  
        if test == True:  
            io.savemat('TestlabeledInputfileNames.mat',{'Input_fileNames':fileNames})
            return 'TestlabeledInputfileNames.mat'
        else:
            io.savemat('TrainlabeledInputfileNames.mat',{'Input_fileNames':fileNames})
            return 'TrainlabeledInputfileNames.mat'
  else:
        if test == True: 
              io.savemat('TestUnlabeledInputfileNames.mat',{'Input_fileNames':fileNames})  
              return 'TestUnlabeledInputfileNames.mat' 
        else:
              io.savemat('TrainUnlabeledInputfileNames.mat',{'Input_fileNames':fileNames})
              return 'TrainUnlabeledInputfileNames.mat'      
  

