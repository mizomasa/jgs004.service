# ----------------------------------------------------
# オートエンコーダーのモデルを構築する
# ----------------------------------------------------
from keras import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import os
import io
from PIL import Image
from util.JGSUtil import convert_to_nparray,resource_folder

class AutoEncoderModel:

    __instance = None
    __autoencoder = None
    __checkpoint_model = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
            
    def get_feature_value(self, base64_data, image_size=100):
        target = convert_to_nparray(base64_data, image_size)
        self.load_model(target)
        feature = self.__checkpoint_model.predict(target)
        return feature[0]

    def convert(self, base64_data, image_size):
        img = Image.open(io.BytesIO(base64_data)).convert("RGB").img.resize(image_size, image_size)
        data = np.asarray(img)
        return (self.convert_np_array(data), self.convert_np_array(data))

    def convert_np_array(self, data):
        v=[]
        v.append(data)
        return np.array(v)

    # モデルの構築
    def build_model(self, in_shape):
        input_img = Input(shape=in_shape)

        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same', name='encoder_layer')(x)

        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(64, 3, 3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

        self.__autoencoder: Model = Model(input_img, decoded)
        self.__autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        

    # モデルを読み込む
    def load_model(self, x_train):
        if not self.__autoencoder is None:
            return
        # モデルの構築
        self.build_model(x_train.shape[1:])
        # モデル読み込み
        self.__autoencoder.load_weights(
            resource_folder('settings/autoencoder.h5'))

        self.__checkpoint_model = Model(
            self.__autoencoder.input, 
            self.__autoencoder.get_layer(name='encoder_layer').output)
    