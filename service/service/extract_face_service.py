import learning.machine.face_cv2 as fcv

class ExtractFaceService():
    def __init__(self):
        pass

    def execute(self, img_orginal_base64):
        return fcv.FaceCV2.get_instance().extract(img_orginal_base64)
