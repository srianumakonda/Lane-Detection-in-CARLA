# Credit goes to https://github.com/zhixuhao/unet/blob/master/model.py. I took this script and made minor modifications to accomedate for my needs. Credit goes to that github script,
# user name is zhixuhao, https://github.com/zhixuhao

import numpy as np 
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras import backend as keras
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

class UNet_Model:

    def __init__(self):
        self.model = None
        self.loaded_model = None
        self.model_loaded = False

    def load_model(self, filepath):
        self.loaded_model = load_model(filepath=filepath)
        self.model_loaded = True
        return self.loaded_model

    def unet(self, input_size, loss, metrics):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        
        self.model = Model(inputs = inputs, outputs = conv10)
        self.model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics =[metrics])
        return self.model

    def model_summary(self):
        return self.model.summary()
        
    def train_model(self, filepath, X_train, y_train, X_val, y_val, epochs, display_callback):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                                        save_weights_only=True,
                                                                        monitor='val_loss',
                                                                        mode='max',
                                                                        save_best_only=True)
        reducelr = ReduceLROnPlateau(monitor="val_loss",
                                     factor=0.2,
                                     patience=3,
                                     min_lr=1e-7,
                                     verbose=1)
                                                    
            
        if self.model_loaded:
            self.loaded_model.fit(x=X_train, y=y_train, validation_data=(X_val,y_val), steps_per_epoch=14610//32, epochs=epochs, callbacks=[model_checkpoint_callback, reducelr, display_callback], verbose=1)      
        else:
            self.model.fit(x=X_train, y=y_train, validation_data=(X_val,y_val), steps_per_epoch=14610//32, epochs=epochs, callbacks=[model_checkpoint_callback, reducelr, display_callback], verbose=1)           

    def test_predict(self, X_test, y_test, idx, model_filepath=None):
        if self.model_loaded:
            fig = plt.figure(figsize=(10,10))
            plt.subplot(3, 3, 1)
            plt.imshow(np.array(X_test[idx]), cmap="gray")
            plt.subplot(3, 3, 2)
            plt.imshow(np.squeeze(self.loaded_model.predict(np.array(X_test[idx]).reshape(1,128,128,3))),cmap="gray")
            plt.subplot(3, 3, 3)
            plt.imshow(np.array(y_test[idx]), cmap="gray")
        else:
            fig = plt.figure(figsize=(10,10))
            plt.subplot(3, 3, 1)
            plt.imshow(np.array(X_test[idx]), cmap="gray")
            plt.subplot(3, 3, 2)
            plt.imshow(np.squeeze(self.loaded_model.predict(np.array(X_test[idx]).reshape(1,128,128,3))),cmap="gray")
            plt.subplot(3, 3, 3)
            plt.imshow(np.array(y_test[idx]), cmap="gray")

    def save_model(self, file_dest):
        if self.model_loaded:
            self.loaded_model.save(file_dest, save_format='tf')
        else:
            self.model.save(file_dest, save_format='tf')