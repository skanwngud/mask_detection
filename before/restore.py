import tensorflow
import numpy as np
import tensorflow as tf
import glob
import cv2
import datetime
import matplotlib.pyplot as plt
# import cvlib as cv

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dropout, Dense, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop, Adamax, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB4

from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

# img_list=glob('c:/datasets/face_train/human/mask/*.jpg')
# img_list_2=glob('c:/datasets/face_train/human/nomask/*.jpg')

# img_list=glob.glob('e:/datasets/train_face/edit/mask/*.jpg')
# img_list_2=glob.glob('e:/datasets/train_face/edit/nomask/*.jpg')
# par_img=glob.glob('e:/datasets/train_face/edit/pareidolia/*.jpg')

# all_list=glob.glob('f:/datasets/train_face/true_all/*.jpg')
# all_list=glob.glob('c:/dataset/mask/export/images/*.jpg')
all_list=glob.glob('c:/dataset/export/images/*.jpg')

# print(len(img_list))

data=list()
label=list()
test=list()

str_time=datetime.datetime.now()
count=1

# print('preprocessing')
# for i in img_list: # mask
#     try:
#         img=cv2.imread(i)
#         # img=cv2.resize(img, (256, 256))
#         # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
#         cv2.imwrite('e:/datasets/train_face/train_mask/edit_mask'+str(count)+'.jpg', img)
#         img=np.array(img)/255.
#         data.append(img)
#         label.append(0)
#         count+=1
#     except:
#         pass
# print('mask preprocessing : ', datetime.datetime.now()-str_time)

# for i in img_list_2: # nomask
#     try:
#         img=tf.keras.preprocessing.image.load_img(i,
#         color_mode='rgb',
#         interpolation='nearest')
#         img=cv2.imread(i)
#         # img=cv2.resize(img, (256, 256))
#         # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
#         cv2.imwrite('e:/datasets/train_face/train_nomask/edit_nomask'+str(count)+'.jpg', img)
#         img=np.array(img)/255.
#         data.append(img)
#         label.append(1)
#         count+=1
#     except:
#         pass
# print('nomask preprocessing : ', datetime.datetime.now()-str_time)


# for i in par_img:
#     try:
#         img=tf.keras.preprocessing.image.load_img(i,
#         color_mode='rgb',
#         interpolation='nearest')
#         img=cv2.imread(i)
#         # img=cv2.resize(img, (256, 256))
#         # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # img=cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7, searchWindowSize=21)
#         cv2.imwrite('e:/datasets/train_face/pareidolia/edit_pareidolia'+str(count)+'.jpg', img)
#         img=np.array(img)/255.
#         data.append(img)
#         label.append(2)
#         count+=1
#     except:
#         pass
# print('pareidolia preprocessing : ', datetime.datetime.now()-str_time)

for i in all_list:
    try:
        img=cv2.imread(i)
        img=cv2.resize(img, (256, 256))
        img=np.array(img)/255.
        test.append(img)
    except:
        pass

# for i in img_list:
#     try:
#         img=cv2.imread(i)
#         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img=cv2.resize(img, (256, 256))
#         img=np.array(img)/255.
#         data.append(img)
#         label.append(0)
#     except:
#         pass

# for i in img_list_2:
#     try:
#         img=cv2.imread(i)
#         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img=cv2.resize(img, (256, 256))
#         img=np.array(img)/255.
#         data.append(img)
#         label.append(1)
#     except:
#         pass

# for i in par_img:
#     try:
#         img=cv2.imread(i)
#         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img=cv2.resize(img, (256, 256))
#         img=np.array(img)/255.
#         data.append(img)
#         label.append(2)
#     except:
#         pass

# data=np.array(data)
# label=np.array(label)
test=np.array(test)

# print(data.shape) # (1328, 256, 256, 3)
# print(label.shape) # (1328, )
# print(test.shape) # (1628, 256, 256, 3)

datagen=ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

datagen2=ImageDataGenerator()



# x_train, x_test, y_train, y_test=train_test_split(
#     data, label,
#     train_size=0.9,
#     random_state=23
# )

# x_train, x_val, y_train, y_val=train_test_split(
#     x_train, y_train,
#     train_size=0.8,
#     random_state=23
# )

# x_train=x_train.reshape(-1, 256, 256, 1)
# x_val=x_val.reshape(-1, 256, 256, 1)
# x_test=x_test.reshape(-1, 256, 256, 1)

# y_train=to_categorical(y_train)
# y_val=to_categorical(y_val)
# y_test=to_categorical(y_test)

# np.save('e:/data/npy/pro_x_train.npy', arr=x_train)
# np.save('e:/data/npy/pro_x_test.npy', arr=x_test)
# np.save('e:/data/npy/pro_x_val.npy', arr=x_val)
# np.save('e:/data/npy/pro_y_train.npy', arr=y_train)
# np.save('e:/data/npy/pro_y_test.npy', arr=y_test)
# np.save('e:/data/npy/pro_y_val.npy', arr=y_val)

batch_size=8

x_train=np.load('c:/data/npy/pro_x_train.npy')
y_train=np.load('c:/data/npy/pro_y_train.npy')
x_test=np.load('c:/data/npy/pro_x_test.npy')
y_test=np.load('c:/data/npy/pro_y_test.npy')
x_val=np.load('c:/data/npy/pro_x_val.npy')
y_val=np.load('c:/data/npy/pro_y_val.npy')


# np.save('c:/data/npy/pro_trainset.npy', arr=trainset)
# np.save('c:/data/npy/pro_valset.npy', arr=valset)
# np.save('c:/data/npy/pro_testset.npy', arr=testset)

es=EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=1
)

rl=ReduceLROnPlateau(
    monitor='val_loss',
    patience=30,
    verbose=1,
    factor=0.1
)

mc=ModelCheckpoint(
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    filepath='c:/data/modelcheckpoint/project.hdf5'
)


trainset=datagen.flow(x_train, y_train, batch_size=batch_size)
valset=datagen2.flow(x_val, y_val)
testset=datagen2.flow(x_test, y_test)

testflow=datagen2.flow(test)

model=Sequential()
model.add(Conv2D(256, 3, padding='same', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.4))

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
model.add(Dropout(0.4))

model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(3, activation='softmax'))






model.compile(
    optimizer=Adam(
        learning_rate=0.001,
        epsilon=None),
    loss='sparse_categorical_crossentropy',
    metrics='acc'
)

epoch=len(x_train)//batch_size

model.fit_generator(
    trainset,
    validation_data=valset,
    epochs=1000,
    steps_per_epoch=epoch,
    callbacks=[es, rl, mc]
)

model.load_weights('c:/data/modelcheckpoint/project.hdf5')
pred=model.predict_generator(testflow)
pred=np.argmax(pred, axis=-1)


for i in range(len(test)):
    if pred[i]==0:
        np.save('c:/dataset/train_face/true_mask/' + str(i) + '.npy', arr=test[i])
        img=np.load('c:/dataset/train_face/true_mask/' + str(i) + '.npy')
        img=Image.fromarray((img*255).astype(np.uint8))
        img.save('c:/dataset/train_face/true_mask/' + str(i) + '.jpg')
        print(str(i) + '번 째 파일은 mask 입니다.')
        i+=1
    elif pred[i]==1:
        np.save('c:/dataset/train_face/true_nomask/' + str(i) + '.npy', arr=test[i])
        img=np.load('c:/dataset/train_face/true_nomask/' + str(i) + '.npy')
        img=Image.fromarray((img*255).astype(np.uint8))
        img.save('c:/dataset/train_face/true_nomask/' + str(i) + '.jpg')
        print(str(i) + '번 째 파일은 face 입니다.')
        i+=1
    elif pred[i]==2:
        np.save('c:/dataset/train_face/true_pareidolia/' + str(i) + '.npy', arr=test[i])
        img=np.load('c:/dataset/train_face/true_pareidolia/' + str(i) + '.npy')
        img=Image.fromarray((img*255).astype(np.uint8))
        img.save('c:/dataset/train_face/true_pareidolia/' + str(i) + '.jpg')
        print(str(i) + '번 째 파일은 pareidolia 입니다.')
        i+=1






print(type(pred[0]))
print(type(pred))
print('전체 : ', len(all_list))
print('마스크사람 : ', len(all_list)-np.count_nonzero(pred))
print('마스크비율 : ', 1-np.count_nonzero(pred)/len(all_list))
print(pred.shape)


# results

# 전체 :  1294
# 마스크사람 :  297
# 마스크비율 :  0.22952086553323026

# 전체 :  1294
# 마스크사람 :  229
# 마스크비율 :  0.17697063369397215

# 전체 :  1294
# 마스크사람 :  54
# 마스크비율 :  0.04173106646058733

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 256, 256, 64)      832
_________________________________________________________________
batch_normalization (BatchNo (None, 256, 256, 64)      256
_________________________________________________________________
activation (Activation)      (None, 256, 256, 64)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 256, 256, 64)      16448
_________________________________________________________________
batch_normalization_1 (Batch (None, 256, 256, 64)      256
_________________________________________________________________
activation_1 (Activation)    (None, 256, 256, 64)      0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 256, 256, 64)      16448
_________________________________________________________________
batch_normalization_2 (Batch (None, 256, 256, 64)      256
_________________________________________________________________
activation_2 (Activation)    (None, 256, 256, 64)      0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 128, 128, 64)      0
_________________________________________________________________
dropout (Dropout)            (None, 128, 128, 64)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 128, 128, 128)     32896
_________________________________________________________________
batch_normalization_3 (Batch (None, 128, 128, 128)     512
_________________________________________________________________
activation_3 (Activation)    (None, 128, 128, 128)     0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 128, 128, 128)     65664
_________________________________________________________________
batch_normalization_4 (Batch (None, 128, 128, 128)     512
_________________________________________________________________
activation_4 (Activation)    (None, 128, 128, 128)     0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 128, 128, 128)     65664
_________________________________________________________________
batch_normalization_5 (Batch (None, 128, 128, 128)     512
_________________________________________________________________
activation_5 (Activation)    (None, 128, 128, 128)     0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 64, 128)       0
_________________________________________________________________
dropout_1 (Dropout)          (None, 64, 64, 128)       0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 64, 64, 256)       131328
_________________________________________________________________
batch_normalization_6 (Batch (None, 64, 64, 256)       1024
_________________________________________________________________
activation_6 (Activation)    (None, 64, 64, 256)       0
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 64, 64, 256)       262400
_________________________________________________________________
batch_normalization_7 (Batch (None, 64, 64, 256)       1024
_________________________________________________________________
activation_7 (Activation)    (None, 64, 64, 256)       0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 64, 64, 256)       262400
_________________________________________________________________
batch_normalization_8 (Batch (None, 64, 64, 256)       1024
_________________________________________________________________
activation_8 (Activation)    (None, 64, 64, 256)       0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 32, 256)       0
_________________________________________________________________
dropout_2 (Dropout)          (None, 32, 32, 256)       0
_________________________________________________________________
flatten (Flatten)            (None, 262144)            0
_________________________________________________________________
dense (Dense)                (None, 1024)              268436480
_________________________________________________________________
batch_normalization_9 (Batch (None, 1024)              4096
_________________________________________________________________
activation_9 (Activation)    (None, 1024)              0
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800
_________________________________________________________________
batch_normalization_10 (Batc (None, 512)               2048
_________________________________________________________________
activation_10 (Activation)   (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328
_________________________________________________________________
batch_normalization_11 (Batc (None, 256)               1024
_________________________________________________________________
activation_11 (Activation)   (None, 256)               0
_________________________________________________________________
dense_3 (Dense)              (None, 128)               32896
_________________________________________________________________
batch_normalization_12 (Batc (None, 128)               512       
_________________________________________________________________
activation_12 (Activation)   (None, 128)               0
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 387
=================================================================
Total params: 269,993,027
Trainable params: 269,986,499
Non-trainable params: 6,528
_________________________________________________________________
'''