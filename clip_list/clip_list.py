import pickle

'''
Script ot make clip list and storing it in .pkl
'''


def make_list():
    clip1 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_43.mkv',
             'archivos_XML/annotations_Station_DDCL_S02_43.xml', (0, 1800)]
    clip2 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_43.mkv',
             'archivos_XML/annotations_Station_DDCL_S02_43.xml', (5000, 6800)]
    clip3 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_37.mkv',
             'archivos_XML/annotations_Station_DDCL_S02_37.xml', (0, 1800)]
    clip4 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_37.mkv',
             'archivos_XML/annotations_Station_DDCL_S02_37.xml', (5000, 6800)]
    clip5 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_37.mkv',
             'archivos_XML/annotations_Station_DDCL_S02_37.xml', (25000, 26800)]
    clip6 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence2/videoTM_32.mkv',
             'archivos_XML/annotations_Station_SDM_S02_32.xml', (6000, 7800)]
    clip7 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence2/videoTM_22.mkv',
             'archivos_XML/annotations_Station_SDM_S02_22.xml', (0, 1800)]
    clip8 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_20.mkv',
             'archivos_XML/annotations_Station_DDCL_S02_20.xml', (2000, 3800)]
    clip9 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_20.mkv',
             'archivos_XML/annotations_Station_DDCL_S02_20.xml', (5000, 6800)]
    clip10 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_20.mkv',
              'archivos_XML/annotations_Station_DDCL_S02_20.xml', (9000, 10800)]
    clip11 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence1/videoTM_24.mkv',
              'archivos_XML/annotations_Station_SDM_S01_24.xml', (18000, 19800)]
    clip12 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence1/videoTM_24.mkv',
              'archivos_XML/annotations_Station_SDM_S01_24.xml', (21000, 22800)]
    clip13 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence1/videoTM_24.mkv',
              'archivos_XML/annotations_Station_SDM_S01_24.xml', (25000, 26800)]
    clip14 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_26.mkv',
              'archivos_XML/annotations_Station_DDCR_S01_26.xml', (5000, 6800)]
    clip15 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_26.mkv',
              'archivos_XML/annotations_Station_DDCR_S01_26.xml', (7000, 8800)]
    clip16 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence1/videoTM_24.mkv',
              'archivos_XML/annotations_Station_SDM_S01_24.xml', (0, 1800)]
    clip17 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence1/videoTM_24.mkv',
              'archivos_XML/annotations_Station_SDM_S01_24.xml', (15000, 16800)]
    clip18 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_26.mkv',
              'archivos_XML/annotations_Station_DDCR_S01_26.xml', (0, 1800)]
    clip19 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_26.mkv',
              'archivos_XML/annotations_Station_DDCR_S01_26.xml', (12000, 13800)]
    clip20 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_26.mkv',
              'archivos_XML/annotations_Station_DDCR_S01_26.xml', (9000, 10800)]
    clip21 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence2/videoTM_10.mkv',
              'archivos_XML/annotations_Bus_SDM_S02_10.xml', (0, 1800)]
    clip22 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence2/videoTM_10.mkv',
              'archivos_XML/annotations_Bus_SDM_S02_10.xml', (5000, 6800)]
    clip23 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence2/videoTM_10.mkv',
              'archivos_XML/annotations_Bus_SDM_S02_10.xml', (10000, 11800)]
    clip24 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence3/videoTM_29.mkv',
              'archivos_XML/annotations_Bus_DDCR_S03_29.xml', (0, 1800)]
    clip25 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence8/videoTM_18.mkv',
              'archivos_XML/annotations_Bus_DDCL_S08_18.xml', (0, 1800)]
    clip26 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence7/videoTM_14.mkv',
              'archivos_XML/annotations_Bus_DDCL_S07_14.xml', (5000, 6800)]
    clip27 = ['/data/faculty/hcarrillo/Bus/SingleDoorRear/Sequence1/videoTM_06.mkv',
              'archivos_XML/annotations_Bus_SDR_S01_06.xml', (0, 1800)]
    clip28 = ['/data/faculty/hcarrillo/Bus/SingleDoorRear/Sequence1/videoTM_06.mkv',
              'archivos_XML/annotations_Bus_SDR_S01_06.xml', (7000, 8800)]
    clip29 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_16.mkv',
              'archivos_XML/annotations_Bus_DDCR_S05_16.xml', (6000, 7800)]
    clip30 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_16.mkv',
              'archivos_XML/annotations_Bus_DDCR_S05_16.xml', (10000, 11800)]
    clip31 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_05.mkv',
              'archivos_XML/annotations_Bus_DDCR_S05_05.xml', (0, 1800)]
    clip32 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_05.mkv',
              'archivos_XML/annotations_Bus_DDCR_S05_05.xml', (8000, 9800)]
    clip33 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_05.mkv',
              'archivos_XML/annotations_Bus_DDCR_S05_05.xml', (17000, 18800)]
    clip34 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence8/videoTM_34.mkv',
              'archivos_XML/annotations_Bus_DDCR_S08_34.xml', (0, 1800)]
    clip35 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence6/videoTM_17.mkv',
              'archivos_XML/annotations_Bus_DDCL_S06_17.xml', (0, 1800)]
    clip36 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence8/videoTM_04.mkv',
              'archivos_XML/annotations_Bus_DDCR_S08_04.xml', (0, 1800)]
    clip37 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence8/videoTM_16.mkv',
              'archivos_XML/annotations_Bus_DDCL_S08_16.xml', (7000, 8800)]
    clip38 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence3/videoTM_10.mkv',
              'archivos_XML/annotations_Bus_DDCL_S03_10.xml', (7000, 8800)]
    clip39 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence4/videoTM_25.mkv',
              'archivos_XML/annotations_Bus_SDM_S04_25.xml', (7000, 8800)]
    clip40 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence8/videoTM_04.mkv',
              'archivos_XML/annotations_Bus_DDCR_S08_04.xml', (7000, 8800)]

    clip_list = [clip1, clip2, clip3, clip4, clip5, clip6, clip7, clip8,
                 clip9, clip10, clip11, clip12, clip13, clip14, clip15,
                 clip16, clip17, clip18, clip19,clip20, clip21, clip22,
                 clip23, clip24, clip25, clip26,clip27, clip28, clip29,
                 clip30, clip31, clip32, clip33,clip34, clip35, clip36,
                 clip37, clip38, clip39, clip40]

    file_name = "clip_list_1-40.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(clip_list, open_file)
    open_file.close()

    # open_file = open(file_name, "rb")
    # loaded_list = pickle.load(open_file)
    # open_file.close()
    # print(loaded_list)


if __name__ == '__main__':
    make_list()
