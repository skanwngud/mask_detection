# 필요 라이브러리 임포트
from glob import glob # 다량의 이미지를 불러오기 위한 라이브러리
from selenium import webdriver # chrome 사용하기 위한 webdriver
from selenium.webdriver.common.keys import Keys
import time # 페이지 로드를 위해 딜레이를 주기 위함
import urllib.request # url 이동

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # 이미지 로드를 위함

import cv2

import tensorflow
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, \
    BatchNormalization, Activation, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    horizontal_flip=True
)

datagen2=ImageDataGenerator()

kernel=np.array([[0, -1, 0], [-1, 5, 0], [0, -1, 0]])

data=list()
label=list()
img_list=glob('c:/datasets/face_train/human/*.jpg')
par_list=glob('c:/datasets/face_train/no_human/*.jpg') # 유사 사람 얼굴
rectangle=(0, 56, 256, 150)

for i in img_list:
    try:
        img=tf.keras.preprocessing.image.load_img(i,
        color_mode='grayscale',
        target_size=(256, 256),
        interpolation='nearest')
        img=np.array(img)/255.
        img=img.reshape(256, 256, 1)
        data.append(img)
        label.append(0)
    except:
        pass

for i in par_list:
    try:
        par=tf.keras.preprocessing.image.load_img(i,
        color_mode='garyscale',
        target_size=(256, 256),
        interpolation='nearest')
        par=np.array(par)/255.
        par=par.reshape(256, 256, 1)
        data.append(par)
        label.append(1)
    except:
        pass

data=np.array(data)
label=np.array(label)

label=to_categorical(label)

x_train, x_test, y_train, y_test=train_test_split(
    data, label,
    train_size=0.95,
    random_state=23
)

x_train, x_val, y_train, y_val=train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=23
)

# train_set=datagen.flow(
#     x_train, y_train,
#     batch_size=4
# )

# test_set=datagen2.flow(
#     x_test, y_test
# )

# val_set=datagen2.flow(
#     x_val, y_val
# )

# np.save('c:/datasets/face_train/x_train.npy', arr=x_train) # x
# np.save('c:/datasets/face_train/y_train.npy', arr=y_train) # y
# np.save('c:/datasets/face_train/x_val.npy', arr=x_val) # x
# np.save('c:/datasets/face_train/y_val.npy', arr=y_val) # y
# np.save('c:/datasets/face_train/x_test.npy', arr=x_test)
# np.save('c:/datasets/face_train/y_test.npy', arr=y_test)


es=EarlyStopping(
    monitor='val_loss',
    patience=150,
    verbose=1
)

rl=ReduceLROnPlateau(
    monitor='val_loss',
    patience=20,
    verbose=1,
    factor=0.1
)

mc=ModelCheckpoint(
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    filepath='c:/data/modelcheckpoint/project_{val_loss:.4f}_{val_acc:.4f}.hdf5'
)

model=Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))

model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))

model.add(Conv2D(256, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Flatten())

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, activation='softmax'))

# 컴파일, 훈련
model.compile(
    optimizer=RMSprop(
        learning_rate=0.01,
        epsilon=None),
    loss='categorical_crossentropy',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=12,
    callbacks=[es, rl, mc]
)

# model.fit_generator(
#     train_set,
#     validation_data=val_set,
#     steps_per_epoch=137,
#     epochs=1000,
#     callbacks=[es, rl, mc]
# )

# model.fit(
#     train_set,
#     validation_data=val_set,
#     steps_per_epoch=273,
#     epochs=50,
#     callbacks=[es, rl, mc]
# )

loss=model.evaluate(
    x_test, y_test
)

# loss=model.evaluate_generator(
#     test_set
# )

# pred=model.predict(
#     x_test
# )

pred=model.predict_generator(
    x_test
)

pred=np.where(pred>0.5, 1, 0)
pred1=pred[:5]

print(loss)
print(pred1)


# results
# [0.5822412967681885, 0.8181818127632141]
# [1]

# results
# [1.1617838144302368, 0.6818181872367859]

