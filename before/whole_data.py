import tensorflow
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image


from glob import glob

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model=load_model('c:/data/modelcheckpoint/project.hdf5')

img_list=glob('c:/data/dataset/mask/testimg/*.jpg')

test=list()

datagen=ImageDataGenerator()

for i in img_list:
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=np.array(img)
        test.append(img)
    except:
        pass

test=np.array(test)
testflow=datagen.flow(test)

# pred=model.predict(test)
pred=model.predict_generator(testflow)
pred=np.argmax(pred, axis=-1)


print('전체 : ', len(img_list)) # 1632
print('마스크사람 : ', len(img_list)-np.count_nonzero(pred))
print('마스크비율 : ', 1-np.count_nonzero(pred)/len(img_list))

i=0
for i in range(len(test)):
    if pred[i]==0:
        np.save('c:/dataset/train_face/true_mask/mask' + str(i) + '.npy', arr=test[i])
        img=np.load('c:/dataset/train_face/true_mask/mask' + str(i) + '.npy')
        img=Image.fromarray((img*255).astype(np.uint8))
        img.save('c:/dataset/train_face/true_mask/mask' + str(i) + '.jpg')
        print(str(i+1) + '번째는 mask 입니다')
        i+=1
    elif pred[i]==1:
        np.save('c:/dataset/train_face/true_nomask/face' + str(i) + '.npy', arr=test[i])
        img=np.load('c:/dataset/train_face/true_nomask/face' + str(i) + '.npy')
        img=Image.fromarray((img*255).astype(np.uint8))
        img.save('c:/dataset/train_face/true_nomask/face' + str(i) + '.jpg')
        print(str(i+1) + '번째는 face 입니다')
        i+=1
    elif pred[i]==2:
        np.save('c:/dataset/train_face/true_pareidolia/pareidolia' + str(i) + '.npy', arr=test[i])
        img=np.load('c:/dataset/train_face/true_pareidolia/pareidolia' + str(i) + '.npy')
        img=Image.fromarray((img*255).astype(np.uint8))
        img.save('c:/dataset/train_face/true_pareidolia/pareidolia' + str(i) + '.jpg')
        print(str(i+1) + '번째는 pareidolia 입니다')
        i+=1

import matplotlib.pyplot as plt

# results
# best_project.hdf5
# 전체 :  1009
# 마스크사람 :  -2018
# 마스크비율 :  3.0
# [0.37070063 0.5508767  0.07842268]

# best__project.hdf5
