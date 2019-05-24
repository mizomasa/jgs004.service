import os
import numpy as np
from annoy import AnnoyIndex
from domain.suggest_image import SuggestImage
from util.JGSUtil import Dumper

class AnnoyWrapper:
    __instance = None
    __images_dic = {}
    __annoys_dic = {}

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
            cls.__instance.init()
        return cls.__instance

    def __init__(self):
        pass

    def get_ui_list(self, genre, feature_Value):
        annoy_db = self.__annoys_dic[genre]
        images = self.__images_dic[genre]
        index_list = annoy_db.get_nns_by_vector(
            self.__conver_vector(feature_Value), 
            10, search_k=-1, include_distances=False)

        ui_paths = [images[i].orginal_path for i in index_list]
        return ui_paths

    def init(self):
        ROOT = "./resorce/images"
        self.load_images(ROOT)
        self.load_annoy()

    def load_images(self, ROOT):
        all_files = os.listdir(ROOT)
        genres = ["gal", "office", "street", "natural"]
        for f in all_files:
            if (f in genres) and os.path.isdir(os.path.join(ROOT, f)):
                self.__images_dic[f] = [] 

        for d in self.__images_dic.keys():
            dumper = Dumper(d)
            images = None
            if dumper.exist():
                images = dumper.de_serialize()
            else:
                images = self.__create_images(os.path.join(ROOT, d))
                dumper.serialize(images)
            self.__images_dic[d] = images

    def load_annoy(self):
        demention = 2704
        self.__annoys_dic = {'gal':AnnoyIndex(demention, metric='angular'),
        'natural':AnnoyIndex(demention, metric='angular'),
        'office':AnnoyIndex(demention, metric='angular'),
        'street':AnnoyIndex(demention, metric='angular')}

        for key in self.__images_dic.keys():
            annoy_file_path = f'./resorce/annoy/{key}.ann'
            annoy_db = self.__annoys_dic[key]

            if os.path.exists(annoy_file_path):
                annoy_db.load(annoy_file_path)
                continue
            for i, image in enumerate(self.__images_dic[key]):
                if not image.exist_file:
                    print(f"nothing file >> {image.orginal_path}")
                    continue
                annoy_db.add_item(i, self.__conver_vector(image.feature))
            annoy_db.build(10)
            annoy_db.save(annoy_file_path)

    def __conver_vector(self, feature):
        return np.ravel(feature)

    def __create_images(self, d):
        print('load start:' + d)
        images = []
        face_path_root = os.path.join(d, 'face')
        original_path_root = os.path.join(d, 'original')
        file_paths = [f for f in os.listdir(original_path_root) if os.path.splitext(f)[1]=='.jpg']
        file_paths.sort()
        for image_name in file_paths:
            img = SuggestImage()
            img.genre = os.path.basename(d)
            img.orginal_path = os.path.join(original_path_root, image_name)
            img.face_path = os.path.join(face_path_root, image_name)
            img.extract_feature()
            images.append(img)
            print('load:' + image_name)
        return images










        