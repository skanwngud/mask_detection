import tensorflow
import glob
import cv2
import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dropout, Dense, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop, Adamax, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from PIL import Image

from sklearn.model_selection import train_test_split

img_list=glob.glob('c:/dataset/train_face/edit/mask/*.jpg')
img_list_2=glob.glob('c:/dataset/train_face/edit/nomask/*.jpg')
par_img=glob.glob('c:/dataset/train_face/edit/pareidolia/*.jpg')
# all_list=glob.glob('c:/dataset/export/images/*.jpg')

data=list()
label=list()
test=list()

count=1

for i in img_list: # mask
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('e:/datasets/train_face/train_mask/edit_mask'+str(count)+'.jpg', img)
        # img=np.array(img)/255.
        data.append(img)
        label.append(0)
        count+=1
    except:
        pass


for i in img_list_2: # nomask
    try:
        img=tf.keras.preprocessing.image.load_img(i,
        color_mode='rgb',
        interpolation='nearest')
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('e:/datasets/train_face/train_nomask/edit_nomask'+str(count)+'.jpg', img)
        # img=np.array(img)/255.
        data.append(img)
        label.append(1)
        count+=1
    except:
        pass


for i in par_img:
    try:
        img=tf.keras.preprocessing.image.load_img(i,
        color_mode='rgb',
        interpolation='nearest')
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('e:/datasets/train_face/pareidolia/edit_pareidolia'+str(count)+'.jpg', img)
        # img=np.array(img)/255.
        data.append(img)
        label.append(2)
        count+=1
    except:
        pass

# for i in all_list:
#     try:
#         img=cv2.imread(i)
#         img=cv2.resize(img, (256, 256))
#         img=np.array(img)/255.
#         test.append(img)
#     except:
#         pass

data=np.array(data)
# label=np.array(label)
# test=np.array(test)

datagen=ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    rescale=1./255
)

datagen2=ImageDataGenerator()

x_train, x_test, y_train, y_test=train_test_split(
    data, label,
    train_size=0.9,
    random_state=23
)

x_train, x_val, y_train, y_val=train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=23
)

y_train=to_categorical(y_train)
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)

trainset=datagen.flow_from_directory(
    'c:/dataset/train_face/edit/nomask/'
)

print(len(data))
print(data.shape)

i=1
for i in range(13):
    plt.imshow(data[i])
# plt.imshow(trainset[0])
plt.show()