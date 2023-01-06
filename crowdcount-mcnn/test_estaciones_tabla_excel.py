import os
import torch
import numpy as np
from skimage.feature import peak_local_max
from src.crowd_count import CrowdCounter
from src.network import load_net
from src.data_loader import ImageDataLoader
from src import utils
from skimage.feature import peak_local_max
import scipy.misc
import cv2 as cv
from utils.countmethods import count_exp, count_exp_nz,make_plots
import re
import csv


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
vis = False
save_output = False

model_path1 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapa_fijo_2520.h5'
model_path2 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/saved_models/station_1.1_re_2__300.h5'
model_path = model_path2
output_dir = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/output'
out='/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

pmae1a,pmae2a,pmae3a,pmae4a,pmae_totala = [], [], [], [], []
pmse1a,pmse2a,pmse3a,pmse4a, pmse_totala= [], [], [], [], []
pmae1b,pmae2b,pmae3b,pmae4b,pmae_totalb = [], [], [], [], []
pmse1b,pmse2b,pmse3b,pmse4b, pmse_totalb= [], [], [], [], []
pmae1c,pmae2c,pmae3c,pmae4c,pmae_totalc = [], [], [], [], []
pmse1c,pmse2c,pmse3c,pmse4c, pmse_totalc= [], [], [], [], []
pmae1d,pmae2d,pmae3d,pmae4d,pmae_totald= [], [], [], [], []
pmse1d,pmse2d,pmse3d,pmse4d, pmse_totald= [], [], [], [], []

maea,mrea,maeb,mreb,maec,mrec,maed,mred= [],[],[],[],[],[],[],[]
maea1,mrea1,maeb1,mreb1,maec1,mrec1,maed1,mred1= [],[],[],[],[],[],[],[]
maea2,mrea2,maeb2,mreb2,maec2,mrec2,maed2,mred2= [],[],[],[],[],[],[],[]
maea3,mrea3,maeb3,mreb3,maec3,mrec3,maed3,mred3= [],[],[],[],[],[],[],[]
maea4,mrea4,maeb4,mreb4,maec4,mrec4,maed4,mred4= [],[],[],[],[],[],[],[]
a,b,c,d,order=0,0,0,0,0
frame_number,clip_number=[],[]
count_estimado,count_integral,cat=[],[],[]
cate=0
for nfol in range(1,101):
    print(nfol)
    data_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/test_data/image_test/station/'+str(nfol)
    gt_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/test_data/gtfx_test/station/'+str(nfol)

    net = CrowdCounter()
        
    trained_model = os.path.join(model_path)
    load_net(trained_model, net)
    net.cuda(device="cuda:3")
    net.eval()
    maetotal_fin=0.0
    mae1a,mae2a,mae3a,mae4a, mae_totala= 0.0, 0.0, 0.0, 0.0, 0.0
    mse1a,mse2a,mse3a,mse4a, mse_totala = 0.0, 0.0, 0.0, 0.0, 0.0
    mae1b,mae2b,mae3b,mae4b, mae_totalb= 0.0, 0.0, 0.0, 0.0, 0.0
    mse1b,mse2b,mse3b,mse4b, mse_totalb = 0.0, 0.0, 0.0, 0.0, 0.0
    mae1c,mae2c,mae3c,mae4c, mae_totalc= 0.0, 0.0, 0.0, 0.0, 0.0
    mse1c,mse2c,mse3c,mse4c, mse_totalc = 0.0, 0.0, 0.0, 0.0, 0.0
    mae1d,mae2d,mae3d,mae4d, mae_totald= 0.0, 0.0, 0.0, 0.0, 0.0
    mse1d,mse2d,mse3d,mse4d, mse_totald = 0.0, 0.0, 0.0, 0.0, 0.0
    count1, count2, count3, count4 , count_total=0,0,0,0,0
    et_count_otsu=0.0
    et_count_integral=0.0
    et_count1, et_count2, et_count3, et_count4 =0.0,0.0,0.0,0.0

    #load test data
    data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    for blob in data_loader:        
        count_total+=1              
        im_data = blob['data']
        gt_data = blob['gt_density']
        fname = blob['fname']
        start = 'IMG_'
        end = '.jpg'
        order=(fname.split(start))[1].split(end)[0]
        order=int(order)

        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        utils.save_density_map(density_map, out, 'image1.png')
        imagen=cv.imread('/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/image1.png', 0)

        otsu_threshold, image_result = cv.threshold(
                        imagen, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,)
        nhead = peak_local_max(imagen, min_distance=5,threshold_abs=otsu_threshold)
        gt_count = round(np.sum(gt_data))

        et_count_otsu= len(nhead)
        et_count_integral=round(np.sum(density_map))
        modmap = np.array(imagen)
        countnz = np.rint(np.count_nonzero(modmap)/65)
        if(gt_count==0 and et_count_integral==0):
            et_count_integral=0
        if(gt_count>=0 and gt_count<=3):
            cate=1
        if(gt_count>=4 and gt_count<=9):
            cate=2
        if(gt_count>=10 and gt_count<=25):
            cate=3
        if(gt_count>=26):
            cate=4
        count_estimado.append(gt_count)
        count_integral.append(et_count_integral)
        cat.append(cate)
        frame_number.append(order)
        if(order>=0 and order<=1799):
            clip_number.append(1)
        if(order>=1800 and order<=3599):
            clip_number.append(2)
        if(order>=3600 and order<=5399):
            clip_number.append(3)
        if(order>=5400 and order<=7199):
            clip_number.append(4)
        if(order>=7200 and order<=8999):
            clip_number.append(5)
        if(order>=9000 and order<=10799):
            clip_number.append(6)
        if(order>=10800 and order<=12599):
            clip_number.append(7)
        if(order>=12600 and order<=14399):
            clip_number.append(8)
        if(order>=14400 and order<=16199):
            clip_number.append(9)
        if(order>=16200 and order<=17999):
            clip_number.append(10)
        if(order>=18000 and order<=19799):
            clip_number.append(11)
        if(order>=19800 and order<=21599):
            clip_number.append(12)
        if(order>=21600 and order<=23399):
            clip_number.append(13)
        if(order>=23400 and order<=25199):
            clip_number.append(14)
        if(order>=25200 and order<=26999):
            clip_number.append(15)
        if(order>=27000 and order<=28799):
            clip_number.append(16)
        if(order>=28800 and order<=30599):
            clip_number.append(17)
        if(order>=30600 and order<=32399):
            clip_number.append(18)
        if(order>=32400 and order<=34199):
            clip_number.append(19)
        if(order>=34200 and order<=36000):
            clip_number.append(20)


with open("datos_Test_Conteos_ordenados.csv", "w") as csv_file:   
    writer = csv.writer(csv_file, delimiter=',')
    level_counter = 0
    max_levels = len(count_estimado)
    while level_counter < max_levels:
        if(level_counter==0):
            writer.writerow(("Conteo GT","Conteo Final","Nivel Frame según GT","# Frame (0-35999)","# clip(1-20)")) 
        else:
            writer.writerow((round(count_estimado[level_counter],2),round(count_integral[level_counter],2),round(cat[level_counter],2),frame_number[level_counter],clip_number[level_counter]))
        level_counter = level_counter +1 

