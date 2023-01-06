import os
import torch
import numpy as np
from skimage.feature import peak_local_max
from src.crowd_count import CrowdCounter
from src.network import load_net
from src.data_loader import ImageDataLoader
from src import utils
from skimage.feature import peak_local_max
from math import sqrt
import scipy.misc
import cv2 as cv
from utils.countmethods import count_exp, count_exp_nz,make_plots

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
vis = False
save_output = False

model_path1 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapa_fijo_2520.h5'
model_path2 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mcnn_shtechA_660.h5'
model_path = model_path1
output_dir = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/output'
out='/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

mae_totala,mse_totala,mae_totalb,mse_totalb,mae_totalc,mse_totalc,mae_totald,mse_totald=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0   

pmae_totala,pmse_totala,pmae_totalb,pmse_totalb,pmae_totalc,pmse_totalc,pmae_totald,pmse_totald=[],[],[],[],[],[],[],[]

for nfol in range(1,10):

    data_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/test_data/image_test/'+str(nfol)
    gt_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/test_data/gtfx_test/'+str(nfol)

    net = CrowdCounter()
        
    trained_model = os.path.join(model_path)
    load_net(trained_model, net)
    net.cuda()
    net.eval()
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
        #et_count = blob['head']
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

        if(et_count_integral<=7):
            et_count1=et_count_integral
        elif (et_count_otsu>=8):
            et_count1=et_count_otsu
        if(et_count1>=4 and et_count1<=17):
            et_count1=countnz

        if(et_count_integral<=7):
            et_count2=et_count_integral
        elif (et_count_otsu>=8):
            et_count2=et_count_otsu

        et_count3 = et_count_integral
        et_count4 = et_count_otsu
       
        mae_totala += abs(gt_count-et_count1)
        mse_totala += ((gt_count-et_count1)*(gt_count-et_count1))

        mae_totalb += abs(gt_count-et_count2)
        mse_totalb += ((gt_count-et_count2)*(gt_count-et_count2))

        mae_totalc += abs(gt_count-et_count3)
        mse_totalc += ((gt_count-et_count3)*(gt_count-et_count3))
        
        mae_totald += abs(gt_count-et_count4)
        mse_totald += ((gt_count-et_count4)*(gt_count-et_count4))
        
        if vis:
            utils.display_results(im_data, gt_data, density_map)
        if save_output:
            utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

    print(nfol)

    mae_totala = mae_totala/count_total
    mse_totala = sqrt(mse_totala/count_total)
    mae_totalb = mae_totalb/count_total
    mse_totalb = sqrt(mse_totalb/count_total)

    mae_totalc = mae_totalc/count_total
    mse_totalc = sqrt(mse_totalc/count_total)

    mae_totald = mae_totald/count_total
    mse_totald = sqrt(mse_totald/count_total)

    pmae_totala.append(mae_totala)
    pmse_totala.append(mse_totala)

    pmae_totalb.append(mae_totalb)
    pmse_totalb.append(mse_totalb)

    pmae_totalc.append(mae_totalc)
    pmse_totalc.append(mse_totalc)

    pmae_totald.append(mae_totald)
    pmse_totald.append(mse_totald)


pmae_totala = np.mean(np.array(pmae_totala))
pmse_totala = np.mean(np.array(pmse_totala))

pmae_totalb = np.mean(np.array(pmae_totalb))
pmse_totalb = np.mean(np.array(pmse_totalb))

pmae_totalc = np.mean(np.array(pmae_totalc))
pmse_totalc = np.mean(np.array(pmse_totalc))

pmae_totald = np.mean(np.array(pmae_totald))
pmse_totald = np.mean(np.array(pmse_totald))


f = open(file_results, 'w') 
print('\n PMAE_nz: %0.2f, PMSE_nz: %0.2f' % (pmae_totala,pmse_totala))
print('\n PMAE_experimental: %0.2f, PMSE_experimental: %0.2f' % (pmae_totalb,pmse_totalb))
print('\n PMAE_integral: %0.2f, PMSE_integral: %0.2f' % (pmae_totalc,pmse_totalc))
print('\n PMAE_picos: %0.2f, PMSE_picos: %0.2f' % (pmae_totald,pmse_totald))


f.write('PMAE: %0.2f, PMSE: %0.2f' % (pmae_totala,pmse_totala))
f.close()
