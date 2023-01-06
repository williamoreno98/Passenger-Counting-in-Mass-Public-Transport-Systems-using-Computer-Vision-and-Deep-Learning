import cv2
# import paramiko
from utils import boundingbox as bb
from numpy import pi, exp, sqrt
import numpy as np
from skimage.feature import peak_local_max


# Class for storing clips info
class Clip:
    def __init__(self, vdir, xmlf, frame_range):
        self.__vdir = vdir
        self.__xmlf = xmlf
        self.__frame_range = frame_range

    def get_vdir(self):
        return self.__vdir

    def get_fpath(self):
        return self.__xmlf

    def get_fran(self):
        return self.__frame_range


def kernel_gaussiano(sigma, tamano):
    # Generacion kernel 2D basado en el sigma y el tamano que determina el rango entre la media(0 para este caso)
    # y 3 desviaciones estándar
    kernel1D = [exp(-z * z / (2 * (sigma ** 2))) / sqrt(2 * pi * (sigma ** 2)) for z in range(-tamano, tamano + 1)]
    # producto exterior consigo mismo para crear kernel gaussiano 2D
    kernel2D = np.outer(kernel1D, kernel1D)
    return kernel2D


# Funtion for map with filtering
def map_gen_filter(frame_number, boxes, video):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = cap.read()
    bframe = bb.boxes_in_frame(frame_number, boxes)
    img = frame.copy()
    imgnew = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgnew = imgnew / 10
    hxml = 0
    for bx in bframe:
        hxml += 1
        y, h, x, w = bb.roifrombx(bx)
        roi_of_img = imgnew[y:y + h, x:x + w] * 10
        roi_of_img = cv2.GaussianBlur(roi_of_img, (7, 7), 8)
        imgnew[y:y + h, x:x + w] = roi_of_img

    return imgnew, hxml


# Function for artificially generating density map
def map_gen_art(frame_number, boxes, video):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = cap.read()
    bframe = bb.boxes_in_frame(frame_number, boxes)
    img = frame.copy()
    imgnew = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgnew = imgnew / 100
    hxml = 0
    for bx in bframe:
        hxml += 1
        y, h, x, w = bb.roifrombx(bx)
        x = x + (int(w / 7))
        y = y + (int(h / 7))
        if w > h:
            tam = (w/10)
            if tam < 0.5:
                tam = 0
            if tam > 0.5 and tam <0.99:
                tam=1
            tam=int(tam)
        else:
            tam = (h/10)
            if tam < 0.5:
                tam = 0
            if tam > 0.5 and tam <0.99:
                tam=1
            tam=int(tam)
        if (tam>0):
            kernel = kernel_gaussiano(8, tam)
            imgnew[y:y + kernel.shape[0], x:x + kernel.shape[1]] = kernel * 50000
    return imgnew, hxml, frame

'''
def get_video(viddir):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='cratos', username='william', password='william', port=22)
    sftp_client = ssh.open_sftp()
    sftp_client.get(viddir, 'new.mkv')
    sftp_client.close()
    ssh.close()
    return None
'''

def head_count(img, thr):
    imgndgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xy = peak_local_max(imgndgray, min_distance=1, threshold_abs=thr)
    nhead = xy.shape[0]
    return nhead, xy
