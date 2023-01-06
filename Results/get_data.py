from utils import map_gen as mpg
from utils import boundingbox as bb
import numpy as np
import os
import cv2
import pickle


'''
Script para obtener y guardar cada imagen en jpg y su mapa de densidad
en csv
'''

def get_data():
    # Obtener la lista de los clips anotados, almacenada en clip_list.pkl
    Clips = bb.load_clip_list()
    
    if not os.path.isdir('./images'):
        os.mkdir('./images')

    if not os.path.isdir('./gt_csv/'):
        os.mkdir('./gt_csv/')

    images_dir = './images/'
    gtcsv_dir = './gt_csv/'

    i = 1
    clip_i = 1
    for tclip in Clips:
        print('Doing clip' + str(clip_i))
        path_video = tclip.get_vdir()
        boxes = bb.boxes_from_xml(tclip.get_fpath())
        ran = tclip.get_fran()
        for x in range(ran[0], ran[1], 1):
            imgnew, hxml, frame = mpg.map_gen_art(x, boxes, path_video)
            # se guarda imgnew como csv
            fname_csv = 'IMG_' + str(i) + '.csv'
            np.savetxt(os.path.join(gtcsv_dir, fname_csv), imgnew, delimiter=",", fmt='%u')
            # guardar imagen original jpg
            fname_img = 'IMG_' + str(i) + '.jpg'
            cv2.imwrite(os.path.join(images_dir, fname_img), frame)
            i += 1
        clip_i += 1


if __name__ == '__main__':
    get_data()
