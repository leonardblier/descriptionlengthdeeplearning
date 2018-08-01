'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Flatten, SpatialDropout2D, BatchNormalization, Input,
    Conv2D, MaxPooling2D, ZeroPadding2D, Activation)
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

import pickle as pkl
import pdb

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

batch_size = 32
num_classes = 10
epochs = 10000000

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets

def customload_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = customload_data("mnist.npz")

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
class MyNormalisation():
    
    def __init__(self):
        pass
    
    def tooltransform(self, imagedata):
        imagedatanormed = imagedata / 255
        yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                        [-0.14714119, -0.28886916,  0.43601035 ],
                        [ 0.61497538, -0.51496512, -0.10001026 ]])
                        
        #imagedatanormed = np.moveaxis(imagedatanormed, 3, 2)
        #imagedatanormed = np.dot(yuv_from_rgb, imagedatanormed)
        #imagedatanormed = np.moveaxis(imagedatanormed, 0, 3)
            
        return imagedatanormed
        
        
    def fit_transform(self, imagedata):
        imagedatanormed = self.tooltransform(imagedata)
            
        self.mean = imagedatanormed.mean()
        imagedatanormed -= self.mean
        
        self.std = imagedatanormed.std()
        imagedatanormed /= self.std
        
        return imagedatanormed
    
    def transform(self, imagedata):
        imagedatanormed = self.tooltransform(imagedata)
        imagedatanormed -= self.mean
        imagedatanormed /= self.std
        return imagedatanormed
        
        
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mynormalisation = MyNormalisation()
x_train = mynormalisation.fit_transform(x_train)
x_test = mynormalisation.transform(x_test)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def make_model_vgg():
    input_ = Input(shape=(28,28,1))
    x = input_
    
    def convbnrelu(nfilters):
        def fun(input_):
            x = ZeroPadding2D((1, 1))(input_)
            x = Conv2D(nfilters, kernel_size=(3,3))(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            return x
        return fun
                 
    def vgglayer(nlayers, nfilters, dropout=0.4):     
        def fun(input_):
            x = input_
            for _ in range(nlayers - 1):
                x = convbnrelu(nfilters)(x)
                x = SpatialDropout2D(dropout)(x)
            x = convbnrelu(nfilters)(x)
            x = MaxPooling2D(pool_size=(2,2))(x)
            return x
        return fun
    
    x = vgglayer(2, 32, dropout=0.3)(x) 
    x = vgglayer(2, 64)(x)
    x = vgglayer(2, 128)(x)
    x = vgglayer(2, 256)(x)
    #x = vgglayer(3, 256)(x)
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    optim = Adam(lr=0.001)
    
    model = Model(inputs=[input_], output=[output])
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=optim,
                metrics=['accuracy'])
    return model



def make_model_mlp():
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
    return model    


    
loss_train = []
loss_test = []
acc_train = []
acc_test = []
histlist = []


modelsscoreslist = []
        
v = 0


cb1 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=500, verbose=1, mode='auto')
cb2 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=50, verbose=1, mode='auto')

datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

class MyImageGenerator():
    def __init__(self, datagen, imgs, labels, batch_size):
        self.datagen = datagen.flow(imgs, labels, batch_size=batch_size)
        self.labels = labels
        self.batch_size = batch_size
        
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    
    def next(self):
        x, y = next(self.datagen)
        #x = x[2:-2,2:-2]
        return (x,y)
    
datagen.fit(x_train)

        
def computescores(modelgenerator, description, shortdescription, 
                  x_train, indexes, **kwargs):
    model = modelgenerator()
    model.summary()
    v = 0
    validation_data = None
    #loss_test = []
    #acc_test = []
    #ltrain = []
    #atrain = []
    
    for k, idx in enumerate(indexes):
        print("===> Training with %d training samples."%(idx))
        model = modelgenerator()
        x_reduced_train = x_train[:idx]
        y_reduced_train = y_train[:idx]
        
        mygen = MyImageGenerator(datagen, x_reduced_train, 
                                 y_reduced_train, batch_size)

        
        if k == len(indexes) - 1:
            x_valid = x_train[idx:]
            y_valid = y_train[idx:]
        else:
            x_valid = x_train[idx:indexes[k+1]]
            y_valid = y_train[idx:indexes[k+1]]
        
        #v=1
        if idx > 10000:
            v = 1
            cb = cb2
        else:
            v=2
            cb = cb1
        
        validation_data = (x_valid, y_valid)
        
        
        
        steps_per_epoch = int(np.ceil(idx/batch_size ))
        hist = model.fit_generator(mygen, steps_per_epoch, 
            validation_data=validation_data,
            verbose=v, callbacks=[cb],
            **kwargs)
        histlist.append(hist.history)
        
        score = model.evaluate(x_valid, y_valid, verbose=0)
    
        loss_test.append(score[0])
        acc_test.append(score[1])
        ltrain = hist.history["loss"][-1]
        atrain = hist.history["acc"][-1]
        
        acc_train.append(atrain)
        loss_train.append(ltrain)
        
        print("Loss : %.3f  Accuracy : %.2f  Loss train : %.3f  Accuracy train %.2f" % (loss_test[-1], acc_test[-1], loss_train[-1], acc_train[-1]))

    rdict = {"description":description, "shortdescription":shortdescription,
        "indexes":indexes, "histories":histlist}
    return rdict



minidx = num_classes
geomparam = 2
maxk = int(np.floor( (np.log(x_train.shape[0]) - np.log(num_classes)) / np.log(geomparam)))
indexes = [int(np.floor(num_classes * geomparam ** k)) for k in range(maxk + 1 )]
#indexes = indexes[-1:]
print("Indexes : ", indexes)

modelscores = computescores(make_model_vgg, 
    "VGG : Same VGG than for CIFAR", 
    "VGG", x_train, indexes, epochs=epochs)

