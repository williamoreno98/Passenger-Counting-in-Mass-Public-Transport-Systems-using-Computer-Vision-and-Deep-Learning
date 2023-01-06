from utils import make_density_map as dmap
from utils import boundingbox as bb
from utils.dotrect import drawrect
import cv2
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm as CM

Clips = bb.load_clip_list()

tclip = Clips[37]

path_video = tclip.get_vdir()
boxes = bb.boxes_from_xml(tclip.get_fpath())
ran = tclip.get_fran()
first = True
for x in range(ran[0]+300, ran[1]-1200, 1):
    frame_number = x
    # print("Doing clip {n} frame {s}".format(n=clip_i, s=frame_number))
    print(frame_number)
    bframe = bb.boxes_in_frame(frame_number, boxes)
    pointList = dmap.get_pointlist(bframe)
    an_cnt = len(pointList)

    cap = cv2.VideoCapture(path_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = cap.read()
    im = frame.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    denmap = dmap.get_density_map_fixed_gaussian(im, pointList)
    denmap1 = dmap.get_density_map_knearest(im, pointList)
    plt.imsave('currentmap.png', denmap, cmap=CM.jet)
    plt.imsave('currentmap1.png', denmap1, cmap=CM.jet)
    mapimg = cv2.imread('/data/estudiantes/william/PdG-Code/data_prep/currentmap.png')
    mapimg1 = cv2.imread('/data/estudiantes/william/PdG-Code/data_prep/currentmap1.png')
    cnt_csv = denmap.sum()
    cnt_csv1 = denmap1.sum()
    gt_3c = mapimg
    gt_3c1 = mapimg1
    img_3c = frame.copy()
    drawrect(img_3c, (25, 40), (615, 465), (0, 0, 255), 2)
    #img_3c = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    #print(gt_3c.shape)
    #print(img_3c.shape)
    scale_percent = 70  # percent of img_3c size
    width = int(img_3c.shape[1] * scale_percent / 100)
    height = int(img_3c.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    og_resized = cv2.resize(img_3c, dim, interpolation=cv2.INTER_AREA)
    gt_resized = cv2.resize(gt_3c, dim, interpolation=cv2.INTER_AREA)
    gt_resized1 = cv2.resize(gt_3c1, dim, interpolation=cv2.INTER_AREA)

    # print nheads in map
    textmap = 'GT: '+str(round(cnt_csv))
    textmap1 = 'GT: '+str(round(cnt_csv1))
    textim = 'AN: '+str(an_cnt)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (200, 40)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    gt_resized = cv2.putText(gt_resized, textmap, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
    gt_resized1 = cv2.putText(gt_resized1, textmap1, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
    og_resized = cv2.putText(og_resized, textim, org, font, fontScale,
                            (255,0,0), thickness, cv2.LINE_AA, False)
    

    # put images together
    h_img = cv2.hconcat([gt_resized, og_resized, gt_resized1])

    if first:
        out = cv2.VideoWriter('map_clip37.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (h_img.shape[1], h_img.shape[0]))
        first = False
    
    out.write(h_img)

out.release()
