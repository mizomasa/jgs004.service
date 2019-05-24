import service.extract_face_service as ext
import service.determine_genre_service as det
import service.feature_value_sevice as fb

class UserFaceImage:
    def __init__(self):
        self.path = ""
        self.data_as_base64 = ""

        self.orginal = ""
        self.data_orginal_as_base64 = ""

        self.genre = ""
        self.feature = [] #特徴量
        self.dimension = 0

    def extract_face(self):
        service = ext.ExtractFaceService()
        self.data_as_base64 = service.execute(self.data_orginal_as_base64)
        return self

    def apply_to_classifier(self):
        service = det.DetermineGenreService()
        self.genre = service.execute(self.data_as_base64)
        return self

    def extract_feature(self):
        service = fb.FeatureValueSevice()
        self.feature = service.excute(self.data_as_base64)
        return self