# ----------------------------------------------------
# オートエンコーダーのモデルを構築する
# ----------------------------------------------------
from keras import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np

import io
from PIL import Image
from util.JGSUtil import convert_to_nparray

    
def get_feature_value(base64_data, image_size=100):
    x_test = convert_to_nparray(base64_data, image_size)
    # モデルを読み込む
    autoencoder = load_model(x_test)

    # 中間層の特徴量を抽出する
    checkpoint_model = Model(autoencoder.input, 
        autoencoder.get_layer(name='encoder_layer').output)
    y = checkpoint_model.predict(x_test)
    return y[0]

def convert(base64_data, image_size):
    img = Image.open(io.BytesIO(base64_data)).convert("RGB").img.resize(image_size, image_size)
    data = np.asarray(img)
    return (convert_np_array(data), convert_np_array(data))

def convert_np_array(data):
    v=[]
    v.append(data)
    return np.array(v)

# モデルの構築
def build_model(in_shape):
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

    autoencoder: Model = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

# モデルを読み込む
def load_model(x_train):
    # モデルの構築
    autoencoder = build_model(x_train.shape[1:])
    # モデル読み込み
    autoencoder.load_weights('./resorce/settings/autoencoder.h5')
    return autoencoder




