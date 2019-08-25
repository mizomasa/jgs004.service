from learning.deep.face_genre_model import FaceGenreModel


class DetermineGenreService():

    def __init__(self):
        pass

    def execute(self, img_base64_data):
        return FaceGenreModel.get_instance().get_genre(img_base64_data)
