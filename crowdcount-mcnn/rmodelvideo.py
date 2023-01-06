import os
import torch
import numpy as np
import pickle
from skimage.feature import peak_local_max
from src.crowd_count import CrowdCounter
from src.network import load_net
from src.data_loader import ImageDataLoader
from src import utils
import cv2 
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from utils.dotrect import drawrect
from utils.boundingbox import Clip
from utils.countmethods import count_exp, count_exp_nz
import statistics as stat
import face_detection
'''
Para sacar videos con los modelos generados
'''
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

detector = face_detection.build_detector("DSFDDetector")

open_file = open("/data/estudiantes/william/PdG-Code/data_prep/clip_list/random_clip_list_1-40.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
Clips = []

for x in loaded_list:
    Clips.append(Clip(x[0], 'none', x[1]))
    
nvideo = 13
Clips = [Clips[nvideo]]

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
vis = False
save_output = False

model_path = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/station_1.1_re_2__300.h5'
model_name = os.path.basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
out= '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn'
net = CrowdCounter()
trained_model = os.path.join(model_path)
load_net(trained_model, net)
net.cuda(device="cuda:4")
net.eval()


for clip in Clips:
    
    vidpath = clip.get_vdir()
    ran = clip.get_fran()

    window = []

    first = True
    alpha=0.8
    
    fr_id = 1
    for frame_number in range (ran[0],ran[1]+1,1):
        
        print("Doing frame {n} video {s}".format(n=fr_id, s=nvideo))
        fr_id += 1
        cap = cv2.VideoCapture(vidpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        res, frame = cap.read()
        img = frame.copy()
        detections = detector.detect(img[:, :, ::-1])[:, :4]
        cv2.imwrite('currentimg.jpg', img)
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
        cv2.imwrite('mapnow.png', dmap)
        imagen = cv2.imread('mapnow.png', 0)
        '''
        otsu_threshold, image_result = cv2.threshold(
                                    imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
        nhead = peak_local_max(imagen, min_distance=5,threshold_abs=otsu_threshold)
        et_count_otsu = len(nhead)

        modmap = np.array(imagen)
        countnz = np.rint(np.count_nonzero(modmap)/65)
        et_count = count_exp_nz(et_count_integral,et_count_otsu, countnz) 
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

        cv2.imwrite('mapnow.png', dmap)
        original = cv2.imread('mapnow.png')
        height, width = original.shape[:2]
        output = cv2.resize(original, (640,480), interpolation = cv2.INTER_AREA)
        dmap = cv2.resize(dmap, (640,480), interpolation = cv2.INTER_AREA)
        plt.imsave('currentmap1.png', dmap, cmap=CM.jet)
        #plt.imsave('currentmap1.png', output, cmap=CM.jet)
        cmap = cv2.imread('currentmap1.png')
        src1= output #map
        src2 = cv2.imread('currentimg.jpg') #img
        
        detections = detector.detect(src2[:, :, ::-1])[:, :4]
        print(len(detections))
        draw_faces_blur(src2, detections)
        cv2.imwrite('blur_prueba.png', src2)

        scale_percent = 70  # percent of img_3c size
        width = int(src1.shape[1] * scale_percent / 100)
        height = int(src1.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        drawrect(src2, (25, 40), (615, 465), (0, 0, 255), 2)
        drawrect(cmap, (25, 40), (615, 465), (0, 0, 255), 2)
        
        # resize image
        og_resized = cv2.resize(src2, dim, interpolation=cv2.INTER_AREA)
        gt_resized = cv2.resize(cmap, dim, interpolation=cv2.INTER_AREA)


        # [blend_images]
        beta = (1.0 - alpha)
        #blended img
        dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)

        drawrect(dst, (25, 40), (615, 465), (0, 0, 255), 2)
        blend_resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
        #usemedian=False
        if usemedian:
            estimado = window_median
        else: 
            estimado = et_count
        #print(estimado)
        
        textmap = 'ET: '+str(estimado)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (200, 40)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        blend_resized = cv2.putText(blend_resized, textmap, org, font, fontScale,
                                color, thickness, cv2.LINE_AA, False)
        
        
            # put images together
        h_img = cv2.hconcat([og_resized, blend_resized, gt_resized])

        if first:
            out = cv2.VideoWriter('rdmvideo_station'+str(nvideo)+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (h_img.shape[1], h_img.shape[0]))
            first = False

        out.write(h_img)
    
    nvideo += 1
    out.release()
        