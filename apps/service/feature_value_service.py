from learning.deep.autoencoder_model import AutoEncoderModel as model

class FeatureValueService:
    IMAGE_SIZE = 100
    def __init__(self):
        pass

    def excute(self, img_face_base64):
        return model.get_instance().get_feature_value(img_face_base64, self.IMAGE_SIZE)
