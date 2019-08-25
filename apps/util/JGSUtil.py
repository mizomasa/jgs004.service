import io
import os
import pickle
import numpy as np
from PIL import Image
from util.base64_util import decode

def convert_to_nparray(img_as_base64, image_size=100):
    img_decoded = decode(img_as_base64)
    img = Image.open(io.BytesIO(img_decoded)).convert("RGB").resize((image_size, image_size))
    np_array = np.asarray(img)
    target_data = []
    target_data.append(np_array)
    return np.array(target_data).astype('float32') / 255

def resource_folder(path=""):
    return os.path.join(os.getcwd(),'apps/resource/',path)

class Dumper():
    def __init__(self, dump_name):
        self.dump_file_name = resource_folder(f'dump/{dump_name}.pickle')

    def serialize(self, obj):
        with open(self.dump_file_name, 'wb') as f:
            pickle.dump(obj, f) 

    def de_serialize(self):
        with open(self.dump_file_name, 'rb') as f:
            return pickle.load(f)      

    def exist(self):
        return os.path.exists(self.dump_file_name)

