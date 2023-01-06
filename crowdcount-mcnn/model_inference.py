import os
import torch
import numpy as np
from skimage.feature import peak_local_max
from src.crowd_count import CrowdCounter
from src.network import load_net
from src.data_loader import ImageDataLoader
from src import utils
import cv2

from utils import make_density_map as mdm
from utils import boundingbox as bb
import matplotlib.pyplot as plt
from matplotlib import cm as CM

#Esto solo es para hacer pruebas

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
vis = False
save_output = False


model_path0 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mcnn_shtechA_660.h5'
model_path2 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapas_fijos_1.1_2520.h5'
model_path = model_path2
model_name = os.path.basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')

net = CrowdCounter()
trained_model = os.path.join(model_path)
load_net(trained_model, net)
net.cuda()
net.eval()


def model_map(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = (ht/4)*4
    wd_1 = (wd/4)*4
    img = cv2.resize(img,(int(wd_1),int(ht_1)))
    img = img.reshape((1,1,img.shape[0],img.shape[1]))
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    density_map = net(img)
    density_map = density_map.data.cpu().numpy()
    dmap = 255 * density_map / np.max(density_map)
    dmap = dmap[0][0]
    et_count_integral = round(density_map.sum())

    cv2.imwrite('mapnowfn.png', dmap)
    imagen = cv2.imread('mapnowfn.png', 0)
    otsu_threshold, image_result = cv2.threshold(
                                imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    nhead = peak_local_max(imagen, min_distance=5,threshold_abs=otsu_threshold)
    et_count_otsu = len(nhead)

    return et_count_integral, et_count_otsu


Clips = bb.load_clip_list()

tclip = Clips[5]

path_video = tclip.get_vdir()
boxes = bb.boxes_from_xml(tclip.get_fpath())
ran = tclip.get_fran()

frame_number = 6503
bframe = bb.boxes_in_frame(frame_number, boxes)
pointList = mdm.get_pointlist(bframe)
an_cnt = len(pointList)


cap = cv2.VideoCapture(path_video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
res, frame = cap.read()
img = frame.copy()

denmap = mdm.get_density_map_fixed_gaussian(img, pointList)
plt.imsave('currentmap.png', denmap, cmap=CM.jet)
mapimg = cv2.imread('/data/estudiantes/william/PdG-Code/data_prep/currentmap.png')
cnt_csv = round(denmap.sum())

et_count_integral, et_count_otsu = model_map(img)
modmap= cv2.imread('mapnowfn.png', 0)
print(modmap.shape)
modmap = np.array(modmap)
countnz = np.rint(np.count_nonzero(modmap)/60)
headcount = 0
for x in range(0, modmap.shape[1]+1, 9):
    for y in range(0, modmap.shape[0]+1, 9):
        checkwindow = modmap[y:y+9,x:x+9]
        if np.count_nonzero(checkwindow) > 20:
            headcount+=1
    
     
        
#print('count nz: ', countnz)


if(et_count_integral<=7):
    et_count=et_count_integral
elif (et_count_otsu>=8):
    et_count=et_count_otsu
print('GT: ', cnt_csv)
print('count nz: ', countnz)
print('count w: ', headcount)
print('Sum: ', et_count_integral)
print('Peak count: ', et_count_otsu)
print('Decision: ', et_count)



