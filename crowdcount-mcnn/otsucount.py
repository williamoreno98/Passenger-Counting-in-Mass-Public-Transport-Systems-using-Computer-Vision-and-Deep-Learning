import os
import torch
import numpy as np
from skimage.feature import peak_local_max
from src.crowd_count import CrowdCounter
from src.network import load_net
from src.data_loader import ImageDataLoader
from src import utils
import cv2 
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from utils import make_density_map as mdm
from utils import boundingbox as bb
'''
Para sacar videos con los modelos generados
'''
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
vis = False
save_output = False

model_path0 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mcnn_shtechA_660.h5'
model_path1 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mcnn_shtechA_9350.h5'
model_path2 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapas_fijos_1.1_2520.h5'
model_path3 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapa_knn_2380.h5'
model_path = model_path2
model_name = os.path.basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')

net = CrowdCounter()
trained_model = os.path.join(model_path)
load_net(trained_model, net)
net.cuda()
net.eval()

Clips = bb.load_clip_list()

tclip = Clips[16]

path_video = tclip.get_vdir()
boxes = bb.boxes_from_xml(tclip.get_fpath())
ran = tclip.get_fran()

frame_number = 15500
bframe = bb.boxes_in_frame(frame_number, boxes)
pointList = mdm.get_pointlist(bframe)
an_cnt = len(pointList)
print("hxml: ", an_cnt)

cap = cv2.VideoCapture(path_video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
res, frame = cap.read()
img = frame.copy()
#cv2.imwrite('currentimg.jpg', img)

#GT Map
denmap = mdm.get_density_map_fixed_gaussian(img, pointList)
#plt.imsave('currentmap.png', denmap, cmap=CM.jet)
#mapimg = cv2.imread('/data/estudiantes/william/PdG-Code/data_prep/currentmap.png')
cnt_csv = round(denmap.sum())
print("gt sum: ", cnt_csv)

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
et_count = round(density_map.sum())
print('et sum: ', et_count)
#print(dmap.shape)

cv2.imwrite('mapnow.png', dmap)
mapimg = cv2.imread('mapnow.png', 0)
# Optimal threshold value is determined automatically.
#th2 = cv2.adaptiveThreshold(dmap,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
 #           cv2.THRESH_BINARY_INV,15,2)
otsu_threshold, image_result = cv2.threshold(
    mapimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
print("Obtained threshold: ", otsu_threshold)
coordinates = peak_local_max(mapimg, min_distance=5, threshold_abs=otsu_threshold)
print("plm: ", len(coordinates))




