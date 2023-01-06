import json
from utils import boundingbox as bb
from utils import map_gen as mpg
import cv2
import matplotlib.pyplot as plt
import pickle

'''
Datos de las anotaciones encontradas en el mapa de densidad para los 19 
primeros clips, pasados a json por cada video
'''


def append_to_dict(nframe, h_xml, hmap, z, d):
    data['video'].append({
        'frame': nframe,
        'h_xml': h_xml,
        'h_map': hmap,
        'head_frame': z,
        'euclidean_d': d
    })


data = dict()
data['video'] = []


def generar_json(i, w):
    while (i >= w - 1800) and (i <= w):
        bframe = bb.boxes_in_frame(i, boxes)
        imgnew, hxml = mpg.map_gen_art(i, boxes, 'new.mkv')
        fname = 'map.png'
        plt.imsave(fname, imgnew, cmap='viridis')
        imgmap = cv2.imread('map.png')
        nheads, xy = mpg.head_count(imgmap, 50)
        a, z = 0, 0
        i += 1
        # si no hay cabezas guardar como 0
        if nheads == 0:
            append_to_dict(i, hxml, nheads, 0, 0)
        for bx in bframe:
            center = bb.getcenter(bx)
            # guardar centroide teorico
            headx_anotation = center[0]
            heady_anotation = center[1]
            if nheads > 0:
                for a in range(nheads):
                    # centro del mapa de densidad
                    headx_art = xy[a][1]
                    heady_art = xy[a][0]
                    # verificacion si está cercano el centroide teorico del practico
                    if abs(headx_art - headx_anotation) < 10 and abs(heady_art - heady_anotation) < 10:
                        z += 1
                        # ecuacion distancia ecuclidiana
                        d = (((headx_anotation - headx_art) ** 2) + ((heady_anotation - heady_art) ** 2)) ** (1 / 2)
                        append_to_dict(i, hxml, nheads, z, round(d, 2))

    return None


# Obtener la lista de los clips anotados, almacenada en clip_list.pkl
open_file = open("./clip_list/clip_list_1-19.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
Clips = []

for x in loaded_list:
    Clips.append(mpg.Clip(x[0], x[1], x[2]))


nclip = 1
for clip in Clips:
    vdir = clip.get_vdir()
    mpg.get_video()
    archivo = clip.get_fpath()
    boxes = bb.boxes_from_xml()
    print('clip '+str(nclip)+' processing')
    fran = clip.get_fran()
    generar_json(fran[0], fran[1])
    nclip += 1

# Set the json filename
jsonFile = 'Resultados2.json'
# Open a json file for writing json data
with open(jsonFile, 'w') as fileHandler1:
    json.dump(data, fileHandler1, indent=2)
