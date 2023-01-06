from utils import map_gen as mpg
from utils import boundingbox as bb
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

def stats_xml():

    # Obtener la lista de los clips anotados, almacenada en clip_list.pkl
    Clips = bb.load_clip_list()

    for clip in Clips:

        filename = clip.get_fpath()
        vid_data = []
        root_node = ET.parse(filename).getroot()

        name_lst = list((root_node.iter('name')))
        vid_data.append(name_lst[0].text)
        boxes = bb.boxes_from_xml(filename)
        frame_range = clip.get_fran()

        vid_data.append(frame_range[0])
        vid_data.append(frame_range[1])

        nheads = []
        for frame in range(int(frame_range[0]), int(frame_range[1]+1), 1):
            bframe = bb.boxes_in_frame(frame, boxes)
            nheads.append(len(bframe))

        nheads = np.array(nheads)
        nheads_mean = nheads.mean()
        total_heads = nheads.sum()
        print('next clip')
        vid_data.append(round(nheads_mean, 2))
        vid_data.append(total_heads)

        data_total.append(vid_data)

    data_total = pd.DataFrame(data_total)
    data_total.to_csv('annotations_data_1-40.csv', index=True, header=['Video', 'Frame inicio', 'Frame final',
                                                                       'Cabezas promedio', 'Total cabezas'])


if __name__ == '__main__':
    stats_xml()
