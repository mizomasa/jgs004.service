import os
import service.feature_value_service as fb
from util.base64_util import encode
class SuggestImage:
    def __init__(self):
        self.genre = ""
        self.feature = None
        self.original_path = ""
        self.face_path = ""
        self.exist_file = False

    def extract_feature(self):
        service = fb.FeatureValueService()
        self.exist_file = os.path.exists(self.face_path)
        if not self.exist_file:
            self.feature = None
            return self
        data_as_base64 = encode(self.face_path)
        self.feature = service.excute(data_as_base64)
        return self