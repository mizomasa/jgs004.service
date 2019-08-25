import os
import re
import pickle
import domain.suggest_image as si
def start():
    ROOT = "./resorce/images"
    all_files = os.listdir(ROOT)
    dirs = [f for f in all_files if os.path.isdir(os.path.join(ROOT, f))]
    for d in dirs:
        do_renames(os.path.join(ROOT, d))

def do_renames(d):
    print('load start:' + d)
    face_path_root = os.path.join(d, 'face')
    #original_path_root = os.path.join(d, 'original')
    for image_name in os.listdir(face_path_root):

        full_path = os.path.join(face_path_root, image_name)
        re_full_path = re.sub("-[0-9][0-9].jpg", ".jpg", full_path)
        os.rename(full_path, re_full_path)
        print('load:' + full_path)
if __name__ == "__main__":
    start()