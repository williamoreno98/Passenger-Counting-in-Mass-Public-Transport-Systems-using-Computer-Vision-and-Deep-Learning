import os
import torch
import numpy as np
from skimage.feature import peak_local_max
from src.crowd_count import CrowdCounter
from src.network import load_net
from src.data_loader import ImageDataLoader
from src import utils
import cv2 
import time
import glob


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
vis = False
save_output = False

model_path = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapas_fijos_1.1_2520.h5'
model_name = os.path.basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')

net = CrowdCounter()
trained_model = os.path.join(model_path)
load_net(trained_model, net)
net.cuda()
net.eval()


img_path = glob.glob('/data/estudiantes/william/PdG-Code/data_prep/data_class/test_data/image_test/89/*.jpg')
timing_list = []

for path in img_path:

    img = cv2.imread(path)

    start = time.process_time() 

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
    cv2.imwrite('mapnow.png', dmap)
    timing = time.process_time() - start
    print(timing)
    timing_list.append(timing)

timing_list = np.array(timing_list)
mean_time = timing_list.mean() 

print('El promedio es ',mean_time)
