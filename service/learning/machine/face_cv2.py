import os
import numpy as np
import cv2
from util.base64_util import decode, encode_data

class FaceCV2:
    _instance = None
    _cascade = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._load_cascade()

    def extract(self, img_orginal_base64):
        image_pic = self._base64_to_imgae(img_orginal_base64)
        face = self.cascade.detectMultiScale(image_pic)
        if face is None:
            print('nothing face.')
            return None
        for x, y, w, h in face:
            face_cut = image_pic[y:y + h, x:x + w]
        cv2.imwrite( "./temp/face/sample.jpg", face_cut)
        return self._image_to_base64(face_cut)

    def _load_cascade(self):
        XML_PATH = './resorce/xml/haarcascade_frontalface_alt2.xml'
        if not os.path.exists(XML_PATH):
            print("Nothing")
            ValueError("Nothing setting file.")
        self.cascade = cv2.CascadeClassifier(XML_PATH)

    def _base64_to_imgae(self, img_orginal_base64):
        img_decoded = decode(img_orginal_base64)
        img_np = np.fromstring(img_decoded, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)

    def _image_to_base64(self, img):
        result, dst_data = cv2.imencode('.jpg', img)
        if not result:
            ValueError("convert error.")

        return encode_data(dst_data)
