from lxml import etree
import cv2
import pickle

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


def load_clip_list():
    open_file = open("/data/estudiantes/william/PdG-Code/data_prep/clip_list/clip_list_1-40.pkl", "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    Clips = []

    for x in loaded_list:
        Clips.append(Clip(x[0], x[1], x[2]))
    
    return Clips


class BBox:
    def __init__(self, frame, xtl, ytl, xbr, ybr):
        self.__frame = frame
        self.__xtl = xtl
        self.__ytl = ytl
        self.__xbr = xbr
        self.__ybr = ybr

    def get_frame(self):
        return self.__frame

    def get_xtl(self):
        return self.__xtl

    def get_ytl(self):
        return self.__ytl

    def get_xbr(self):
        return self.__xbr

    def get_ybr(self):
        return self.__ybr

    def get_center(self):
        xCenter = (self.__xtl + self.__xbr) / 2
        yCenter = (self.__ytl + self.__ybr) / 2
        return [int(xCenter), int(yCenter)]

    def get_ptbl(self):
        return [self.__xtl, self.__ytl]

    def get_pttr(self):
        return [self.__xbr, self.__ybr]


def boxes_from_xml(xmlf):
    tree = etree.parse(xmlf)
    root = tree.getroot()

    boxes = []
    # Loop through tracks and through each box
    for track in root.iter('track'):
        #    tr_id = int(track.attrib['id'])
        for box in track.iter('box'):
            frame = int(box.attrib['frame'])
            xtl = int(float(box.attrib['xtl']))
            ytl = int(float(box.attrib['ytl']))
            xbr = int(float(box.attrib['xbr']))
            ybr = int(float(box.attrib['ybr']))
            boxes.append(BBox(frame, xtl, ytl, xbr, ybr))
    return boxes


#  get boxes from a frame
def boxes_in_frame(nframe, boxes):
    fr_boxes = []
    for box in boxes:
        bxframe = box.get_frame()
        if bxframe == nframe:
            fr_boxes.append(box)

    return fr_boxes


def getcenter(bx):
    center = bx.get_center()
    return center


# draw rectangle and center in image
def draw_box(bx, frame):
    center = bx.get_center()
    ptbl = bx.get_ptbl()
    pttr = bx.get_pttr()
    img = frame
    img = cv2.rectangle(img, (ptbl[0], ptbl[1]), (pttr[0], pttr[1]), (0, 255, 0), 1)
    img = cv2.rectangle(img, (center[0], center[1]), (center[0], center[1]), (0, 0, 255), 3)
    return img


def roifrombx(box):
    ptbl = box.get_ptbl()
    pttr = box.get_pttr()
    col = ptbl[1]
    height = pttr[1]-ptbl[1]
    row = ptbl[0]
    width = pttr[0]-ptbl[0]

    return col, height, row, width