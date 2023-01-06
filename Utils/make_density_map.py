import numpy as np
import math
from utils import boundingbox as bb
import scipy
import scipy.io as io
import scipy.io as scio
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree

def get_pointlist(bframe): 
    xpoint = []
    ypoint = []

    for box in bframe:
        center = box.get_center()
        if 25 < center[0] < 615 and 40 < center[1] < 465:
            xpoint.append(center[0])
            ypoint.append(center[1])

    xpoint = np.array(xpoint)
    ypoint = np.array(ypoint)
    pointList = np.column_stack((xpoint, ypoint))
    return pointList


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_density_map_fixed_gaussian(im, points):
    '''
    Adapted from  
    https://github.com/svishwa/crowdcount-mcnn/blob/master/data_preparation/get_density_map_gaussian.m
    '''
    im_density = np.zeros((im.shape[0], im.shape[1]))
    h, w = np.shape(im_density)

    if len(points) == 0:
        return im_density

    for j in range(0, len(points[:, 0]+1)):
        f_sz = 29 #número impar
        sigma = 8
        H = matlab_style_gauss2D((f_sz, f_sz), sigma)
        x = min(w, max(1, abs(int(math.floor(points[j][0])))))
        y = min(h, max(1, abs(int(math.floor(points[j][1])))))
        if (x > w or y > h):
            pass

        x1 = x - int(math.floor(f_sz / 2))
        y1 = y - int(math.floor(f_sz / 2))
        x2 = x + int(math.floor(f_sz / 2))+1
        y2 = y + int(math.floor(f_sz / 2))+1
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True

        if (y1 < 1):
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True

        if (x2 > w):
            dfx2 = x2 - w
            x2 = w
            change_H = True

        if (y2 > h):
            dfy2 = y2 - h
            y2 = h
            change_H = True

        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2

        if change_H:
            H = matlab_style_gauss2D((y2h - y1h + 1, x2h - x1h + 1), sigma)

        im_density[y1:y2, x1:x2] = im_density[y1:y2, x1:x2] + H

    return im_density

def get_density_map_knearest(im, points):

    #partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
    img_shape=[im.shape[0],im.shape[1]]
    im_density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return im_density
    
    leafsize = 10
    # build kdtree
    tree = KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.019
        else:
            sigma = np.average(np.array(pt.shape))/2./2. #case: 1 point
        if(sigma>1e6):
            sigma=15
        im_density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    
    return im_density
