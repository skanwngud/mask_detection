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

from PIL import Image

from sklearn.model_selection import train_test_split, KFold

# img_list=glob('c:/datasets/face_train/human/mask/*.jpg')
# img_list_2=glob('c:/datasets/face_train/human/nomask/*.jpg')


# 이미지 로드
# img_list=glob.glob('c:/datasets/train_face/train_mask/*.jpg')
# img_list_2=glob.glob('c:/datasets/train_face/train_nomask/*.jpg')
# par_img=glob.glob('c:/datasets/train_face/pareidolia/*.jpg')

# img_list=glob.glob('f:/datasets/train_face/edit/mask/*.jpg')
# img_list_2=glob.glob('f:/datasets/train_face/edit/nomask/*.jpg')
# par_img=glob.glob('f:/datasets/train_face/edit/pareidolia/*.jpg')


all_list=glob.glob('f:/datasets/train_face/true_all/*.jpg')


# data=list()
# label=list()

# print(len(img_list)) # 525
# print(len(img_list_2)) # 307
# print(len(par_img)) # 496

data=list()
label=list()
test=list()

str_time=datetime.datetime.now()
count=1

# print('preprocessing')
# for i in img_list: # mask
#     try:
#         img=tf.keras.preprocessing.image.load_img(i,
#         color_mode='rgb',
#         interpolation='nearest')
#         # img=cv2.imread(i)
#         img=cv2.resize(img, (256, 256))
#         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
#         # img=cv2.imread(i)
#         img=cv2.resize(img, (256, 256))
#         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
#         # img=cv2.imread(i)
#         img=cv2.resize(img, (256, 256))
#         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        img=tf.keras.preprocessing.image.load_img(i,
        color_mode='rgb',
        interpolation='nearest')
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

data=np.array(data)
label=np.array(label)
test=np.array(test)

tf.convert_to_tensor(test)

print(type(test))
print(test.shape)

# print(data.shape) # (1328, 256, 256, 3)
# print(label.shape) # (1328, )
# print(test.shape) # (2129, 256, 256, 3)
# print(type(test)) 

datagen=ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2
)


datagen2=ImageDataGenerator()


# x_train, x_test, y_train, y_test=train_test_split(data, label, train_size=0.9, random_state=23)

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

# np.save('c:/data/npy/pro_x_train.npy', arr=x_train)
# np.save('c:/data/npy/pro_x_test.npy', arr=x_test)
# np.save('c:/data/npy/pro_x_val.npy', arr=x_val)
# np.save('c:/data/npy/pro_y_train.npy', arr=y_train)
# np.save('c:/data/npy/pro_y_test.npy', arr=y_test)
# np.save('c:/data/npy/pro_y_val.npy', arr=y_val)

batch_size=8

x_train=np.load('c:/data/npy/pro_x_train.npy')
y_train=np.load('c:/data/npy/pro_y_train.npy')
x_test=np.load('c:/data/npy/pro_x_test.npy')
y_test=np.load('c:/data/npy/pro_y_test.npy')
x_val=np.load('c:/data/npy/pro_x_val.npy')
y_val=np.load('c:/data/npy/pro_y_val.npy')

trainset=datagen.flow(x_train, y_train, batch_size=batch_size)
valset=datagen2.flow(x_val, y_val)
testset=datagen2.flow(x_test, y_test)

es=EarlyStopping(
    monitor='val_loss',
    patience=150,
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

model=Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))

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
model.add(Dropout(0.2))

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
model.add(Dropout(0.2))
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

model.add(Dense(3, activation='softmax'))


# 컴파일, 훈련
model.compile(
    optimizer=Adam(
        learning_rate=0.001,
        epsilon=None),
    loss='sparse_categorical_crossentropy',
    metrics='acc'
)

epoch=len(x_train)//batch_size

# model.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     epochs=1000,
#     batch_size=8,
#     callbacks=[es, rl, mc]
# )

model.fit_generator(
    trainset,
    validation_data=valset,
    epochs=1,
    steps_per_epoch=epoch,
    callbacks=[es, rl, mc]
)

model.load_weights('c:/data/modelcheckpoint/project.hdf5')
pred=model.predict(test)
pred=np.argmax(pred, axis=-1)


print(type(pred[0]))
print('전체 : ', len(all_list)) # 1632
print('마스크사람 : ', len(all_list)-np.count_nonzero(pred))
print('마스크비율 : ', np.count_nonzero(pred)/len(all_list))
print(pred[0])

plt.imshow(pred[0])
plt.show()

# imgnum=0
# for image in pred:
#     if image==0:
#         cv2.imwrite('e:/datasets/train_face/true_mask/' + str(imgnum) + '.jpg', image)
#     elif image==1:
#         cv2.imwrite('e:/datasets/train_face/true_nomask/' + str(imgnum) + '.jpg', image)
#     elif image==2:
#         cv2.imwrite('e:/datasets/train_face/true_pareidolia/' + str(imgnum) + '.jpg', image)

# results
# no_imagedatagenerator
# 
