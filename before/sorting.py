import numpy as np
import keras
import glob
import datetime
import cv2

from PIL import Image

from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization,\
    Activation, GaussianDropout, GlobalAveragePooling2D, Input, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split, KFold

eff = EfficientNetB2(
    include_top=False,
    input_shape=(256, 256, 3)
)

eff.trainable = True

es = EarlyStopping(
    monitor='val_loss',
    patience=30,
    verbose=1
)

rl = ReduceLROnPlateau(
    monitor='val_loss',
    patience=10,
    verbose=1
)

mc = ModelCheckpoint(
    monitor='val_loss',
    save_best_only=True,
    filepath='c:/data/modelcheckpoint/project_final.hdf5',
    verbose=1
)

datagen = ImageDataGenerator(
    width_shift_range=(-1, 1),
    height_shift_range=(-1, 1),
    rotation_range=40,
    preprocessing_function=preprocess_input
)

datagen2 = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_images = glob.glob(
    'c:/data/dataset/export/images/*.jpg'
)

print(len(test_images)) # 1009

test = list()
for i in test_images:
    temp = cv2.imread(i)
    temp = cv2.resize(
        temp, (256, 256))
    temp = np.array(temp)/255.
    test.append(temp)

test = np.array(test)

print(type(test))

x_train = np.load(
    'c:/data/npy/pro_x_train.npy'
)

x_val = np.load(
    'c:/data/npy/pro_x_val.npy'
)

x_test = np.load(
    'c:/data/npy/pro_x_test.npy'
)

y_train = np.load(
    'c:/data/npy/pro_y_train.npy'
)

y_val = np.load(
    'c:/data/npy/pro_y_val.npy'
)

y_test= np.load(
    'c:/data/npy/pro_y_test.npy'
)

print(x_train.shape) # (956, 256, 256, 3)
print(x_val.shape) # (239, 256, 256, 3)
print(x_test.shape) # (133, 256, 256, 3)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# print(x_train[0]) # 1/255 완료

# batch_size = 32
# epochs = len(x_train)//batch_size

# train_set = datagen.flow(
#     x_train, y_train,
#     batch_size=batch_size,
#     seed=23
# )

# val_set = datagen2.flow(
#     x_val, y_val,
#     batch_size=batch_size,
#     seed=23
# )

# test_set = datagen2.flow(
#     x_test, y_test,
#     batch_size=batch_size,
#     seed=23
# )

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
x_test = preprocess_input(x_test)

test = preprocess_input(test)

print(test.shape)

# 모델

x = eff.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = GaussianDropout(0.4)(x)
output = Dense(3)(x)
model = Model(eff.input, output)

# 컴파일, 훈련
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics='acc'
)

# model.fit_generator(
#     train_set,
#     validation_data=val_set,
#     epochs = 1,
#     steps_per_epoch = epochs,
#     callbacks = [es, rl, mc]
# )

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs = 1000,
    callbacks = [es, rl, mc]
)

# 평가, 예측
# loss = model.evaluate_generator(
#     test_set
# )

model.load_weights(
    'c:/data/modelcheckpoint/project_final.hdf5'
)

loss = model.evaluate(
    x_test, y_test
)

pred = model.predict(
    test
)

pred = np.argmax(pred, axis = -1)

print(loss)
print(type(pred))
print(pred.shape)

for i in range(len(test_images)):
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

print('전체 : ', len(test_images)) # 1632
print('마스크사람 : ', len(test_images)-np.count_nonzero(pred))
print('마스크비율 : ', 1-np.count_nonzero(pred)/len(test_images))
