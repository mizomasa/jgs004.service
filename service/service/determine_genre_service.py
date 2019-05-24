from learning.deep.face_genre_model import build_model
from util.JGSUtil import convert_to_nparray

class DetermineGenreService():
    categories = ["gal", "natural", "office", "street"]
    __model = None
    def __init__(self):
        pass

    def execute(self, img_base64_data):
        np_array = convert_to_nparray(img_base64_data)
        if DetermineGenreService.__model is None:
            DetermineGenreService.__model = build_model(np_array.shape[1:])
            #TDOD; face_genre_modelに戻っていく
            DetermineGenreService.__model.load_weights('./resorce/settings/face-model4.hdf5')
        pre = DetermineGenreService.__model.predict(np_array)
        for i, p in enumerate(pre):
            return self.categories[p.argmax()]