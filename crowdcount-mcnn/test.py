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

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
vis = False
save_output = False

model_path1 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapa_fijo_2520.h5'
model_path2 = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mapas_fijos_1.1_2520.h5'
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


for nfol in range(1,101):

    data_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/test_data/image_test/'+str(nfol)
    gt_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/test_data/gtfx_test/'+str(nfol)

    net = CrowdCounter()
        
    trained_model = os.path.join(model_path)
    load_net(trained_model, net)
    net.cuda()
    net.eval()
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
        #fname = blob['fname']

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
            et_count1 = countnz

        if(et_count_integral<=7):
            et_count2=et_count_integral
        elif (et_count_otsu>=8):
            et_count2=et_count_otsu
        
        et_count3 = et_count_integral
        et_count4 = et_count_otsu
       
        if(gt_count==0 and et_count1==0):
            et_count1=1
        if(gt_count==0 and et_count2==0):
            et_count2=1
        if(gt_count==0 and et_count3==0):
            et_count3=1
        if(gt_count==0 and et_count4==0):
            et_count3=1
        if(gt_count==0):
           gt_count=1

        #print('gt_count: '+str(gt_count)+' '+'et_count: '+str(et_count))

        mae_totala += abs(gt_count-et_count1)
        #mse_totala += ((gt_count-et_count1)*(gt_count-et_count1))
        mse_totala += abs((gt_count-et_count1)/gt_count)

        mae_totalb += abs(gt_count-et_count2)
        #mse_totalb += ((gt_count-et_count2)*(gt_count-et_count2))
        mse_totalb += abs((gt_count-et_count2)/gt_count)

        mae_totalc += abs(gt_count-et_count3)
        #mse_totalc += ((gt_count-et_count3)*(gt_count-et_count3))
        mse_totalc += abs((gt_count-et_count3)/gt_count)

        mae_totald += abs(gt_count-et_count4)
        #mse_totald += ((gt_count-et_count4)*(gt_count-et_count4))
        mse_totald += abs((gt_count-et_count4)/gt_count)

        if (gt_count<=3):
            mae1a += abs(gt_count-et_count1)
            #mse1a += ((gt_count-et_count1)*(gt_count-et_count1))
            mse1a += abs((gt_count-et_count1)/gt_count)

            mae1b += abs(gt_count-et_count2)
            #mse1b += ((gt_count-et_count2)*(gt_count-et_count2))
            mse1b += abs((gt_count-et_count2)/gt_count)

            mae1c += abs(gt_count-et_count3)
            #mse1c += ((gt_count-et_count3)*(gt_count-et_count3))
            mse1c += abs((gt_count-et_count3)/gt_count)

            mae1d += abs(gt_count-et_count4)
            #mse1d += ((gt_count-et_count4)*(gt_count-et_count4))
            mse1d += abs((gt_count-et_count4)/gt_count)

            if(count1<=2):
                maea.append(abs(gt_count-et_count1))
                mrea.append(100*abs((gt_count-et_count1)/gt_count))

                maeb.append(abs(gt_count-et_count2))
                mreb.append(100*abs((gt_count-et_count2)/gt_count))

                maec.append(abs(gt_count-et_count3))
                mrec.append(100*abs((gt_count-et_count3)/gt_count))

                maed.append(abs(gt_count-et_count4))
                mred.append(100*abs((gt_count-et_count4)/gt_count))

                maea1.append(abs(gt_count-et_count1))
                mrea1.append(100*abs((gt_count-et_count1)/gt_count))

                maeb1.append(abs(gt_count-et_count2))
                mreb1.append(100*abs((gt_count-et_count2)/gt_count))

                maec1.append(abs(gt_count-et_count3))
                mrec1.append(100*abs((gt_count-et_count3)/gt_count))

                maed1.append(abs(gt_count-et_count4))
                mred1.append(100*abs((gt_count-et_count4)/gt_count))
            count1 +=1
        if (gt_count>=4 and gt_count<=9):
            mae2a += abs(gt_count-et_count1)
            #mse2a += ((gt_count-et_count1)*(gt_count-et_count1))
            mse2a += abs((gt_count-et_count1)/gt_count)

            mae2b += abs(gt_count-et_count2)
            #mse2b += ((gt_count-et_count2)*(gt_count-et_count2))
            mse2b += abs((gt_count-et_count2)/gt_count)

            mae2c += abs(gt_count-et_count3)
            #mse2c += ((gt_count-et_count3)*(gt_count-et_count3))
            mse2c += abs((gt_count-et_count3)/gt_count)

            mae2d += abs(gt_count-et_count4)
            #mse2d += ((gt_count-et_count4)*(gt_count-et_count4))
            mse2d += abs((gt_count-et_count4)/gt_count)
            
            if(count2<=2):
                maea.append(abs(gt_count-et_count1))
                mrea.append(100*abs((gt_count-et_count1)/gt_count))

                maeb.append(abs(gt_count-et_count2))
                mreb.append(100*abs((gt_count-et_count2)/gt_count))

                maec.append(abs(gt_count-et_count3))
                mrec.append(100*abs((gt_count-et_count3)/gt_count))

                maed.append(abs(gt_count-et_count4))
                mred.append(100*abs((gt_count-et_count4)/gt_count))

                maea2.append(abs(gt_count-et_count1))
                mrea2.append(100*abs((gt_count-et_count1)/gt_count))

                maeb2.append(abs(gt_count-et_count2))
                mreb2.append(100*abs((gt_count-et_count2)/gt_count))

                maec2.append(abs(gt_count-et_count3))
                mrec2.append(100*abs((gt_count-et_count3)/gt_count))

                maed2.append(abs(gt_count-et_count4))
                mred2.append(100*abs((gt_count-et_count4)/gt_count))

            count2 +=1
        if (gt_count>=10 and gt_count<=25):
            mae3a += abs(gt_count-et_count1)
            #mse3a += ((gt_count-et_count1)*(gt_count-et_count1))
            mse3a += abs((gt_count-et_count1)/gt_count)

            mae3b += abs(gt_count-et_count2)
            #mse3b += ((gt_count-et_count2)*(gt_count-et_count2))
            mse3b += abs((gt_count-et_count2)/gt_count)

            mae3c += abs(gt_count-et_count3)
            #mse3c += ((gt_count-et_count3)*(gt_count-et_count3))
            mse3c += abs((gt_count-et_count3)/gt_count)

            mae3d += abs(gt_count-et_count4)
            #mse3d += ((gt_count-et_count4)*(gt_count-et_count4))
            mse3d += abs((gt_count-et_count4)/gt_count)

            if(count3<=2):
                maea.append(abs(gt_count-et_count1))
                mrea.append(100*abs((gt_count-et_count1)/gt_count))

                maeb.append(abs(gt_count-et_count2))
                mreb.append(100*abs((gt_count-et_count2)/gt_count))

                maec.append(abs(gt_count-et_count3))
                mrec.append(100*abs((gt_count-et_count3)/gt_count))

                maed.append(abs(gt_count-et_count4))
                mred.append(100*abs((gt_count-et_count4)/gt_count))

                maea3.append(abs(gt_count-et_count1))
                mrea3.append(100*abs((gt_count-et_count1)/gt_count))

                maeb3.append(abs(gt_count-et_count2))
                mreb3.append(100*abs((gt_count-et_count2)/gt_count))

                maec3.append(abs(gt_count-et_count3))
                mrec3.append(100*abs((gt_count-et_count3)/gt_count))

                maed3.append(abs(gt_count-et_count4))
                mred3.append(100*abs((gt_count-et_count4)/gt_count))
            count3 +=1
        if (gt_count>=26):
            mae4a += abs(gt_count-et_count1)
            #mse4a += ((gt_count-et_count1)*(gt_count-et_count1))
            mse4a += abs((gt_count-et_count1)/gt_count)

            mae4b += abs(gt_count-et_count2)
            #mse4b += ((gt_count-et_count2)*(gt_count-et_count2))
            mse4b += abs((gt_count-et_count2)/gt_count)

            mae4c += abs(gt_count-et_count3)
            #mse4c += ((gt_count-et_count3)*(gt_count-et_count3))
            mse4c += abs((gt_count-et_count3)/gt_count)

            mae4d += abs(gt_count-et_count4)
            #mse4d += ((gt_count-et_count4)*(gt_count-et_count4))
            mse4d += abs((gt_count-et_count4)/gt_count)


            if(count4<=2):
                maea.append(abs(gt_count-et_count1))
                mrea.append(100*abs((gt_count-et_count1)/gt_count))

                maeb.append(abs(gt_count-et_count2))
                mreb.append(100*abs((gt_count-et_count2)/gt_count))

                maec.append(abs(gt_count-et_count3))
                mrec.append(100*abs((gt_count-et_count3)/gt_count))

                maed.append(abs(gt_count-et_count4))
                mred.append(100*abs((gt_count-et_count4)/gt_count))

                maea4.append(abs(gt_count-et_count1))
                mrea4.append(100*abs((gt_count-et_count1)/gt_count))

                maeb4.append(abs(gt_count-et_count2))
                mreb4.append(100*abs((gt_count-et_count2)/gt_count))

                maec4.append(abs(gt_count-et_count3))
                mrec4.append(100*abs((gt_count-et_count3)/gt_count))

                maed4.append(abs(gt_count-et_count4))
                mred4.append(100*abs((gt_count-et_count4)/gt_count))
            count4 +=1
        if vis:
            utils.display_results(im_data, gt_data, density_map)
        if save_output:
            utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

    print(nfol)

    mae1a = mae1a/count1
    mse1a = (mse1a/count1)*100
    mae1b = mae1b/count1
    mse1b = (mse1b/count1)*100
    mae1c = mae1c/count1
    mse1c = (mse1c/count1)*100
    mae1d = mae1d/count1
    mse1d = (mse1d/count1)*100
    
    mae2a = mae2a/count2
    mse2a = (mse2a/count2)*100
    mae2b = mae2b/count2
    mse2b = (mse2b/count2)*100
    mae2c = mae2c/count2
    mse2c = (mse2c/count2)*100
    mae2d = mae2d/count2
    mse2d = (mse2d/count2)*100

    mae3a = mae3a/count3
    mse3a = (mse3a/count3)*100
    mae3b = mae3b/count3
    mse3b = (mse3b/count3)*100
    mae3c = mae3c/count3
    mse3c = (mse3c/count3)*100
    mae3d = mae3d/count3
    mse3d = (mse3d/count3)*100

    mae4a = mae4a/count4
    mse4a = (mse4a/count4)*100
    mae4b = mae4b/count4
    mse4b = (mse4b/count4)*100
    mae4c = mae4c/count4
    mse4c = (mse4c/count4)*100
    mae4d = mae4d/count4
    mse4d = (mse4d/count4)*100

    mae_totala = mae_totala/count_total
    mse_totala = (mse_totala/count_total)*100
    mae_totalb = mae_totalb/count_total
    mse_totalb = (mse_totalb/count_total)*100

    mae_totalc = mae_totalc/count_total
    mse_totalc = (mse_totalc/count_total)*100

    mae_totald = mae_totald/count_total
    mse_totald = (mse_totald/count_total)*100

    pmae1a.append(mae1a)
    pmse1a.append(mse1a)

    pmae1b.append(mae1b)
    pmse1b.append(mse1b)

    pmae1c.append(mae1c)
    pmse1c.append(mse1c)

    pmae1d.append(mae1d)
    pmse1d.append(mse1d)

    pmae2a.append(mae2a)
    pmse2a.append(mse2a)

    pmae2b.append(mae2b)
    pmse2b.append(mse2b)

    pmae2c.append(mae2c)
    pmse2c.append(mse2c)

    pmae2d.append(mae2d)
    pmse2d.append(mse2d)

    pmae3a.append(mae3a)
    pmse3a.append(mse3a)

    pmae3b.append(mae3b)
    pmse3b.append(mse3b)

    pmae3c.append(mae3c)
    pmse3c.append(mse3c)

    pmae3d.append(mae3d)
    pmse3d.append(mse3d)

    pmae4a.append(mae4a)
    pmse4a.append(mse4a)

    pmae4b.append(mae4b)
    pmse4b.append(mse4b)

    pmae4c.append(mae4c)
    pmse4c.append(mse4c)

    pmae4d.append(mae4d)
    pmse4d.append(mse4d)

    pmae_totala.append(mae_totala)
    pmse_totala.append(mse_totala)

    pmae_totalb.append(mae_totalb)
    pmse_totalb.append(mse_totalb)

    pmae_totalc.append(mae_totalc)
    pmse_totalc.append(mse_totalc)

    pmae_totald.append(mae_totald)
    pmse_totald.append(mse_totald)

        
    make_plots(pmae1a, pmse1a, pmae1b, pmse1b, pmae1c, pmse1c, pmae1d, pmse1d,
               pmae2a, pmse2a, pmae2b, pmse2b, pmae2c, pmse2c, pmae2d, pmse2d,
               pmae3a, pmse3a, pmae3b, pmse3b, pmae3c, pmse3c, pmae3d, pmse3d,
               pmae4a, pmse4a, pmae4b, pmse4b, pmae4c, pmse4c, pmae4d, pmse4d,
               pmae_totala, pmse_totala, pmae_totalb, pmse_totalb, pmae_totalc, pmse_totalc, pmae_totald, pmse_totald,
               maea, mrea, maeb, mreb, maec, mrec, maed, mred,
               maea1, mrea1, maeb1, mreb1, maec1, mrec1, maed1, mred1,
               maea2, mrea2, maeb2, mreb2, maec2, mrec2, maed2, mred2,
               maea3, mrea3, maeb3, mreb3, maec3, mrec3, maed3, mred3,
               maea4, mrea4, maeb4, mreb4, maec4, mrec4, maed4, mred4)

pmae1a = np.mean(np.array(pmae1a))
pmse1a = np.mean(np.array(pmse1a))

pmae1b = np.mean(np.array(pmae1b))
pmse1b = np.mean(np.array(pmse1b))

pmae1c = np.mean(np.array(pmae1c))
pmse1c = np.mean(np.array(pmse1c))

pmae1d = np.mean(np.array(pmae1d))
pmse1d = np.mean(np.array(pmse1d))

pmae2a = np.mean(np.array(pmae2a))
pmse2a = np.mean(np.array(pmse2a))

pmae2b = np.mean(np.array(pmae2b))
pmse2b = np.mean(np.array(pmse2b))

pmae2c = np.mean(np.array(pmae2c))
pmse2c = np.mean(np.array(pmse2c))

pmae2d = np.mean(np.array(pmae2d))
pmse2d = np.mean(np.array(pmse2d))

pmae3a = np.mean(np.array(pmae3a))
pmse3a = np.mean(np.array(pmse3a))

pmae3b = np.mean(np.array(pmae3b))
pmse3b = np.mean(np.array(pmse3b))

pmae3c = np.mean(np.array(pmae3c))
pmse3c = np.mean(np.array(pmse3c))

pmae3d = np.mean(np.array(pmae3d))
pmse3d = np.mean(np.array(pmse3d))

pmae4a = np.mean(np.array(pmae4a))
pmse4a = np.mean(np.array(pmse4a))

pmae4b = np.mean(np.array(pmae4b))
pmse4b = np.mean(np.array(pmse4b))

pmae4c = np.mean(np.array(pmae4c))
pmse4c = np.mean(np.array(pmse4c))

pmae4d = np.mean(np.array(pmae4d))
pmse4d = np.mean(np.array(pmse4d))

pmae_totala = np.mean(np.array(pmae_totala))
pmse_totala = np.mean(np.array(pmse_totala))

pmae_totalb = np.mean(np.array(pmae_totalb))
pmse_totalb = np.mean(np.array(pmse_totalb))

pmae_totalc = np.mean(np.array(pmae_totalc))
pmse_totalc = np.mean(np.array(pmse_totalc))

pmae_totald = np.mean(np.array(pmae_totald))
pmse_totald = np.mean(np.array(pmse_totald))


f = open(file_results, 'w') 
print('\n PMAE1_final: %0.2f, PMSE1_final: %0.2f' % (pmae1a,pmse1a))
print('\n PMAE1_experimental: %0.2f, PMSE1_experimental: %0.2f' % (pmae1b,pmse1b))
print('\n PMAE1_integral: %0.2f, PMSE1_integral: %0.2f' % (pmae1c,pmse1c))
print('\n PMAE1_picos: %0.2f, PMSE1_picos: %0.2f' % (pmae1d,pmse1d))

print('\n PMAE2_final: %0.2f, PMSE2_final: %0.2f' % (pmae2a,pmse2a))
print('\n PMAE2_experimental: %0.2f, PMSE2_experimental: %0.2f' % (pmae2b,pmse2b))
print('\n PMAE2_integral: %0.2f, PMSE2_integral: %0.2f' % (pmae2c,pmse2c))
print('\n PMAE2_picos: %0.2f, PMSE2_picos: %0.2f' % (pmae2d,pmse2d))

print('\n PMAE3_final: %0.2f, PMSE3_final: %0.2f' % (pmae3a,pmse3a))
print('\n PMAE3_experimental: %0.2f, PMSE3_experimental: %0.2f' % (pmae3b,pmse3b))
print('\n PMAE3_integral: %0.2f, PMSE3_integral: %0.2f' % (pmae3c,pmse3c))
print('\n PMAE3_picos: %0.2f, PMSE3_picos: %0.2f' % (pmae3d,pmse3d))

print('\n PMAE4_final: %0.2f, PMSE4_final: %0.2f' % (pmae4a,pmse4a))
print('\n PMAE4_experimental: %0.2f, PMSE4_experimental: %0.2f' % (pmae4b,pmse4b))
print('\n PMAE4_integral: %0.2f, PMSE4_integral: %0.2f' % (pmae4c,pmse4c))
print('\n PMAE4_picos: %0.2f, PMSE4_picos: %0.2f' % (pmae4d,pmse4d))

print('\n PMAE_total_final: %0.2f, PMSE_total_final: %0.2f' % (pmae_totala,pmse_totala))
print('\n PMAE_total_experimental: %0.2f, PMSE_total_experimental: %0.2f' % (pmae_totalb,pmse_totalb))
print('\n PMAE_total_integral: %0.2f, PMSE_total_integral: %0.2f' % (pmae_totalc,pmse_totalc))
print('\n PMAE_total_picos: %0.2f, PMSE_total_picos: %0.2f' % (pmae_totald,pmse_totald))

f.write('PMAE: %0.2f, PMSE: %0.2f' % (pmae_totala,pmse_totala))
f.close()
