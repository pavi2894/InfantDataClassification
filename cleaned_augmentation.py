import numpy as np 

import cleaned_aux_module as sp
import cleaned_config as _CONF

# Empirical roation matrices for sides:
Hltr = np.asarray([[-0.6536, -0.3446, 0.3172],
                   [0.1933, 0.6131, -0.0504],
                   [-0.3240, 0.2318, 0.3371]],dtype=np.float32)
Hrtl = np.asarray([[-0.6654, 0.2982, -0.3210],
                    [-0.1835, 0.6253, 0.1453],
                    [0.2493, 0.0849, 0.3586]],dtype=np.float32)
Lltr = np.asarray([[-0.8484, -0.0982, 0.3760],
                    [0.0167, 0.9384, 0.0535],
                    [0.0504, 0.1832, -0.4790]],dtype=np.float32)
Lrtl = np.asarray([[-0.7939, 0.1771, -0.3136],
                    [-0.0702, 0.9383, 0.0421],
                    [-0.0352, 0.2043, -0.4785]],dtype=np.float32)


def mirror_sensors(acc, gyro):
    return

def dropout_noise(data, p):
    mask = np.random.binomial(1,1.0-p,data.shape)
    return data * mask

def additive_noise(data, snr_range=[5, 10]):
    # get snr from range
    snr = np.random.random_sample()*(snr_range[1] - snr_range[0]) + snr_range[0]
    
    # Generate noise for each channel
    for i in range(data.shape[1]):
        noise = np.random.randn(data.shape[0])
        noise *= ( 10.0 ** (-snr / 20.0) * np.linalg.norm(data[:,i]) ) / ( np.linalg.norm(noise) )
        #import ipdb;ipdb.set_trace()
        data[:,i] += noise
  
    return data

def rotationMatrix(a_x, a_y, a_z, angle_type='deg'):
    if angle_type == 'deg':
        a_x *= np.pi / 180.0
        a_y *= np.pi / 180.0
        a_z *= np.pi / 180.0

    M = np.array([
        [np.cos(a_y)*np.cos(a_z), 
        -np.cos(a_x)*np.sin(a_z) + np.sin(a_x)*np.sin(a_y)*np.cos(a_z), 
         np.sin(a_x)*np.sin(a_z) + np.cos(a_x)*np.sin(a_y)*np.cos(a_z)],
        [np.cos(a_y)*np.sin(a_z), 
        np.cos(a_x)*np.cos(a_z) + np.sin(a_x)*np.sin(a_y)*np.sin(a_z),
        -np.sin(a_x)*np.cos(a_z) + np.cos(a_x)*np.sin(a_y)*np.sin(a_z)],
        [-np.sin(a_y),
        np.sin(a_x)*np.cos(a_y),
        np.cos(a_x)*np.cos(a_y)]], dtype=np.float32)

    return M

def time_warping(data, p=0.2):
    basevec = np.arange(_CONF.winlen,dtype=np.float32)+1.0
    Nframes = np.floor(((data.shape[0]-_CONF.winlen)/_CONF.winlen)+1).astype(np.int)
    for iFrame in range(Nframes):
        # Randomly warp p*100% of frames
        if np.random.random_sample() <= p:
        # Random sinusoid with random phase, amplitude [0.5, 1.5], frequency
            freq = np.random.random_sample()*basevec/np.float(basevec.shape[0])
            phase = 2*np.pi*np.random.random_sample()
            amplitude = np.random.random_sample()
            sinusoid = amplitude * np.sin(2*np.pi*freq + phase) + 2
            sinusoid /= np.mean(sinusoid)

            newbase = np.cumsum(sinusoid)
            start = iFrame*_CONF.winlen
            stop = start+_CONF.winlen
            for iChan in range(data.shape[1]):
                data[start:stop,iChan] = np.interp(newbase,basevec, data[start:stop, iChan])

    return data



def random_rotation(data, range_x=[-10, 10], range_y=[-10, 10], range_z=[-10, 10]):
    # Get rotation matrix
    #Random rotation for each sensor
    Nsens = data.shape[1] // 6
    n = data.shape[-1]//2
    acc = data[:,:n]
    gyro = data[:,n:]
    for i in range(Nsens):
        a_x = np.random.random_sample()*(range_x[1] - range_x[0])+range_x[0]
        a_y = np.random.random_sample()*(range_y[1] - range_y[0])+range_y[0]
        a_z = np.random.random_sample()*(range_z[1] - range_z[0])+range_z[0]
        M = rotationMatrix(a_x, a_y, a_z)
        
        acc[:,i*3:(i+1)*3] = np.matmul(acc[:,i*3:(i+1)*3],M)
        gyro[:,i*3:(i+1)*3] = np.matmul(gyro[:,i*3:(i+1)*3],M)

    data = np.concatenate([acc, gyro], axis=-1)

    #import ipdb;ipdb.set_trace()
    # Rotate axes

    return data

def channel_dropout(data,num_chans=1,tot_chans=4):
    chans_to_drop = np.random.permutation(tot_chans)
    chans_to_drop = chans_to_drop[:num_chans]
    N = data.shape[-1]//2
    for i in chans_to_drop:
        data[:,(3*i):(3*i+3)] *= 0.0 # Accelerometer signals
        data[:,(N+3*i):(N+3*i+3)] *= 0.0 # Gyroscope signals


    return data

def frame_sig(X, winlen, hop):
    Nframes = np.floor(((X.shape[0]-winlen)/hop)+1).astype(np.int)
    numchans = np.int32(X.shape[1])
    X_framed = np.zeros([Nframes,numchans,winlen],dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0,Nframes):
        start = np.int((i)*hop)
        stop = np.int(start+winlen)
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed


def augment(data):
    # Augmentation to frames


    #print('Data shape in: %i, %i, %i' % (data.shape[0], data.shape[1], data.shape[2]))
    # Assume data is 50% overlapped
    N = data.shape[-1]//2
    #data = np.concatenate([np.reshape(data[:,:N],[-1]), data[-1,N:]], axis=0)
    #data = np.concatenate(
    #    [np.reshape(np.transpose(data[:,:,:N],[0,2,1]),[-1,data.shape[1]]), 
    #    np.transpose(data[-1,:,N:])], 
    #    axis=0)
    data = np.concatenate(
        [np.reshape(data[:,:,:,:N],[-1,data.shape[1]]), 
        np.transpose(data[-1,:,:,N:])], 
        axis=0)




    # Time warping
    if np.random.random_sample() < _CONF.aug_p_time_warping:
        data = time_warping(data, p=1.0)

    # Random rotation
    if np.random.random_sample() < _CONF.aug_p_rotation:
        data = random_rotation(data)

    # Additive noise augmentation
    if np.random.random_sample() < _CONF.aug_p_noise:
        #data = additive_noise(data,snr_range=[10,30])    
        data = dropout_noise(data,_CONF.aug_p_dropout)

    # Sensor dropout:
    if np.random.random_sample() < _CONF.aug_p_chandropout:
        data = channel_dropout(data,num_chans=1)

    # Retain framed format
    data = sp.frame_sig(data, _CONF.winlen, _CONF.hop)

    #print('Data shape out: %i, %i, %i' % (data.shape[0], data.shape[1], data.shape[2]))
    return data



# Test script:
if __name__ == "__main__":
    data = np.zeros([123,24,120])
    data[:,0,:] = 1.0
    data[:,1,:] = 2.0
    data[:,2,:] = 3.0
    N = 60
    a = np.reshape(np.transpose(data[:,:,:N],[0,2,1]),[-1,data.shape[1]])
    b = np.transpose(data[-1,:,N:])
    data2 = np.concatenate([np.reshape(np.transpose(data[:,:,:N],[0,2,1]),[-1,data.shape[1]]), np.transpose(data[-1,:,N:])], axis=0)




