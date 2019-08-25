from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from util.JGSUtil import convert_to_nparray,resource_folder
import os
import math
class FaceGenreModel:

    __categories = ["gal", "natural", "office", "street"]
    __instance = None
    __model = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        pass

    def __build_model(self, in_shape):
        if not self.__model is None:
            return self.__model

        self.__model = Sequential()

        self.__model.add(Conv2D(64, (3, 3), input_shape=in_shape))
        self.__model.add(Activation('relu'))
        self.__model.add(Conv2D(64, (3, 3)))
        self.__model.add(Activation('relu'))
        self.__model.add(Conv2D(64, (3, 3)))
        self.__model.add(Activation('relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))

        self.__model.add(Conv2D(128, (3, 3)))
        self.__model.add(Activation('relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))

        self.__model.add(Conv2D(128, (3, 3)))
        self.__model.add(Activation('relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))

        self.__model.add(Flatten())
        self.__model.add(Dense(64))
        self.__model.add(Activation('relu'))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(len(self.__categories), 
        kernel_initializer='uniform'))

        self.__model.add(Activation('softmax'))
        self.__model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        
    def get_genre(self, img_base64_data):
        np_array = convert_to_nparray(img_base64_data)
        if self.__model is None:
            self.__build_model(np_array.shape[1:])
            self.__model.load_weights(
                resource_folder('settings/face-model4.hdf5'))
    
        pre = self.__model.predict(np_array)
        for i, p in enumerate(pre):
            detail_percentage={}
            for i,v in enumerate(p):
                detail_percentage[self.__categories[i]] = round(p[i]*100,2)
            return self.__categories[p.argmax()], detail_percentage