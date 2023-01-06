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
from utils.dotrect import drawrect
import statistics as stat
import face_detection
import csv

save_estimado, save_GT=[],[]

def draw_faces_blur(src2, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        #cv2.rectangle(src2, (x0, y0), (x1, y1), (0, 0, 255), 2)
        x,y= x0,y0
        w, h = x1-x0,y1-y0
        ROI = src2[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(ROI, (13,13), 0) 
        # Insert ROI back into image
        src2[y:y+h, x:x+w] = blur

'''
Para sacar videos con los modelos generados
'''

detector = face_detection.build_detector("DSFDDetector")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
vis = False
save_output = False

model_path = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/station_1.1_re_2__300.h5'
model_name = os.path.basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')

net = CrowdCounter()
trained_model = os.path.join(model_path)
load_net(trained_model, net)
net.cuda(device="cuda:3")
net.eval()


# frame_number = 100
first = True
alpha=0.8
Clips = bb.load_clip_list()

nvideo = 19
Clips = [Clips[nvideo]]

for tclip in Clips:
    window = []
    path_video = tclip.get_vdir()
    boxes = bb.boxes_from_xml(tclip.get_fpath())
    ran = tclip.get_fran()
    fr_id = 1
    first = True
    for frame_number in range(ran[0]+200, ran[1]-400, 1):
        print("Doing frame {n} video {s}".format(n=fr_id, s=nvideo))
        fr_id += 1
        bframe = bb.boxes_in_frame(frame_number, boxes)
        pointList = mdm.get_pointlist(bframe)
        an_cnt = len(pointList)
        
        cap = cv2.VideoCapture(path_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        res, frame = cap.read()
        img = frame.copy()
        cv2.imwrite('currentimg2.jpg', img)
        src2 = img
        #GT Map
        denmap = mdm.get_density_map_fixed_gaussian(img, pointList)
        plt.imsave('currentmap2.png', denmap, cmap=CM.jet)
        cnt_csv = round(denmap.sum())
        save_GT.append(cnt_csv)
        #print(img.shape)
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
        et_count = et_count_integral
        cv2.imwrite('mapnow2.png', dmap)
        save_estimado.append(et_count)

        '''
        imagen = cv2.imread('mapnow2.png', 0)
        otsu_threshold, image_result = cv2.threshold(
                                    imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
        nhead = peak_local_max(imagen, min_distance=5,threshold_abs=otsu_threshold)
        et_count_otsu = len(nhead)
        countnz = np.rint(np.count_nonzero(modmap)/65)

        et_count = count_exp_nz(et_count_integral, et_count_otsu, countnz)             
        '''
        
        if len(window) < 19:
            window.append(et_count)
            usemedian = False
        if len(window) >= 19:
            movep = window[0:17]
            window[1:18] = movep
            window[0] = et_count
            window_median = stat.median(window)
            usemedian = True
        
        cv2.imwrite('mapnow2.png', dmap)
        original = cv2.imread('mapnow2.png')
        height, width = original.shape[:2]
        output = cv2.resize(original, (640,480), interpolation = cv2.INTER_AREA)

        src1= output #map
        # src2 = cv2.imread('currentimg2.jpg') #img
        cmap = cv2.imread('currentmap2.png')

        #blur faces 
        detections = detector.detect(src2[:, :, ::-1])[:, :4]
        print(len(detections))
        draw_faces_blur(src2, detections)

        cv2.imwrite('blur_prueba2.png', src2)

        # [blend_images]
        beta = (1.0 - alpha)
        #blended img
        dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)

        drawrect(src2, (25, 40), (615, 465), (0, 0, 255), 2)
        drawrect(cmap, (25, 40), (615, 465), (0, 0, 255), 2)
        drawrect(dst, (25, 40), (615, 465), (0, 0, 255), 2)

        scale_percent = 70  # percent of img_3c size
        width = int(src1.shape[1] * scale_percent / 100)
        height = int(src1.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        og_resized = cv2.resize(src2, dim, interpolation=cv2.INTER_AREA)
        gt_resized = cv2.resize(cmap, dim, interpolation=cv2.INTER_AREA)
        blend_resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)

        if usemedian:
            estimado = window_median
        else: 
            estimado = et_count

        textmap = 'GT: '+str(cnt_csv)
        textet = 'ET: '+str(estimado)
        textim = 'AN: '+str(an_cnt)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (200, 40)
        fontScale = 1
        color = (0, 255, 0)
        thickness = 2
        gt_resized = cv2.putText(gt_resized, textmap, org, font, fontScale,
                                color, thickness, cv2.LINE_AA, False)
        blend_resized = cv2.putText(blend_resized, textet, org, font, fontScale,
                                color, thickness, cv2.LINE_AA, False)
        og_resized = cv2.putText(og_resized, textim, org, font, fontScale,
                                color, thickness, cv2.LINE_AA, False)

        h_img = cv2.hconcat([og_resized, gt_resized, blend_resized])

        if first:
            out = cv2.VideoWriter('gtresultsmodel_station'+str(nvideo)+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (h_img.shape[1], h_img.shape[0]))
            first = False

        out.write(h_img)
    
    nvideo += 1
    out.release()

datos_espaciados=np.round(np.linspace(0,60,num=len(save_estimado)),2)
with open("datos_clip_19.csv", "w") as csv_file:   
    writer = csv.writer(csv_file, delimiter=',')
    level_counter = 0
    max_levels = len(save_estimado)
    while level_counter < max_levels:
        if(level_counter==0):
            writer.writerow(("GT","Estimado","Intervalo")) 
        else:
            writer.writerow((save_GT[level_counter],save_estimado[level_counter],datos_espaciados[level_counter]))
        level_counter = level_counter +1 