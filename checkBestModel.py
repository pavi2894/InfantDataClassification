
import numpy as np

import operator
import collections
folder = 'SM1FinalKfold_F1Dense_TrainigLossModels/'
dense_softmax = {}
for i in [64,128,140,160,256,512,32]:
    dense_softmax_dict = np.load(folder+'Final_dict'+str(i)+'.npy',allow_pickle=True)
    dense_softmax_dict = eval(str(dense_softmax_dict))
    dense_softmax.update(dense_softmax_dict)

#wavenet = np.load('WavenetFinal_dict.npy',allow_pickle=True)
#wavenet = eval(str(wavenet))

dense_best_F1 = sorted(dense_softmax.items(), key=lambda v : v[1][0], reverse=True)[0]
dense_best_Acc = sorted(dense_softmax.items(), key=lambda v : v[1][1], reverse=True)[0]
print(dense_best_Acc)
print(dense_best_F1)
print("\n\n\n")
#wavenet_best = max(wavenet.items(), key=operator.itemgetter(1))[0]
print("Best Dense+softmax later is ", dense_best_Acc[0],"  and accuracy value is ", dense_softmax[dense_best_Acc[0]][1])
print("Best Dense+softmax later is ", dense_best_F1[0],"  and accuracy value is ", dense_softmax[dense_best_F1[0]][0])

#print("Best wavenet model is ", wavenet_best, "  and accuracy value is ", wavenet[wavenet_best])
#wavenetsorted =  dict(sorted(wavenet.items(), key=lambda item: item[1]))
#print(wavenetsorted)
print("\n\n\n Sorted based on F1 score")
dense_softmaxsorted = sorted(dense_softmax.items(), key=lambda v : v[1][0], reverse=True)
print(dense_softmaxsorted)
