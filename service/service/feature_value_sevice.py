from learning.deep.autoencoder_model import get_feature_value

class FeatureValueSevice:
    IMAGE_SIZE = 100
    def __init__(self):
        pass

    def excute(self, img_face_base64):
        return get_feature_value(img_face_base64, self.IMAGE_SIZE)
