import pickle

def make_list():
    clip1 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_18.mkv', (1, 1200)]
    clip2 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_59.mkv', (5000, 6200)]
    clip3 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence2/videoTM_39.mkv', (0, 1200)]
    clip4 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence3/videoTM_02.mkv', (5000, 6200)]
    clip5 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence3/videoTM_03.mkv', (25000, 26200)]
    clip6 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence1/videoTM_05.mkv', (1, 1200)]
    clip7 = ['/data/faculty/hcarrillo/Station/SingleDoorMiddle/Sequence1/videoTM_29.mkv', (3000, 4200)]
    clip8 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence1/videoTM_04.mkv', (2000, 3200)]
    clip9 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraLeft/Sequence1/videoTM_50.mkv', (5000, 6200)]
    clip10 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence3/videoTM_06.mkv', (5500, 6700)]
    clip11 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence3/videoTM_03.mkv', (1800, 3000)]
    clip12 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence2/videoTM_34.mkv', (2100, 3300)]
    clip13 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence2/videoTM_18.mkv', (2500, 3700)]
    clip14 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence2/videoTM_29.mkv', (5000, 6200)]
    clip15 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_79.mkv',(7000, 8200)]
    clip16 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_29.mkv', (0, 1200)]
    clip17 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_30.mkv',(0, 1200)]
    clip18 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence1/videoTM_70.mkv', (0, 1200)]
    clip19 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence2/videoTM_22.mkv', (1200, 2400)]
    clip20 = ['/data/faculty/hcarrillo/Station/DoubleDoorCameraRight/Sequence2/videoTM_35.mkv', (9000, 10200)]
    clip21 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence2/videoTM_16.mkv', (0, 1200)]
    clip22 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence2/videoTM_13.mkv', (5000, 6200)]
    clip23 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence2/videoTM_10.mkv', (9000, 10200)]
    clip24 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence3/videoTM_09.mkv', (1000, 2200)]
    clip25 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence3/videoTM_16.mkv', (0, 1200)]
    clip26 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence5/videoTM_10.mkv', (5000, 6200)]
    clip27 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence3/videoTM_08.mkv', (0, 1200)]
    clip28 = ['/data/faculty/hcarrillo/Bus/SingleDoorRear/Sequence1/videoTM_01.mkv', (7000, 8200)]
    clip29 = ['/data/faculty/hcarrillo/Bus/SingleDoorRear/Sequence1/videoTM_08.mkv', (6000, 7200)]
    clip30 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_13.mkv',(1000, 2200)]
    clip31 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_05.mkv', (5000, 6200)]
    clip32 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_05.mkv', (10800, 12000)]
    clip33 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence5/videoTM_10.mkv', (1700, 2900)]
    clip34 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence8/videoTM_16.mkv',  (0, 1200)]
    clip35 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence4/videoTM_13.mkv', (0, 1200)]
    clip36 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraRight/Sequence8/videoTM_04.mkv', (3000, 4200)]
    clip37 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence8/videoTM_03.mkv', (500, 1700)]
    clip38 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence3/videoTM_08.mkv', (7000, 8200)]
    clip39 = ['/data/faculty/hcarrillo/Bus/SingleDoorMiddle/Sequence4/videoTM_22.mkv', (5000, 6200)]
    clip40 = ['/data/faculty/hcarrillo/Bus/DoubleDoorCameraLeft/Sequence6/videoTM_04.mkv', (7000, 8200)]

    clip_list = [clip1, clip2, clip3, clip4, clip5, clip6, clip7, clip8,
                 clip9, clip10, clip11, clip12, clip13, clip14, clip15,
                 clip16, clip17, clip18, clip19,clip20, clip21, clip22,
                 clip23, clip24, clip25, clip26,clip27, clip28, clip29,
                 clip30, clip31, clip32, clip33,clip34, clip35, clip36,
                 clip37, clip38, clip39, clip40]

    file_name = "random_clip_list_1-40.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(clip_list, open_file)
    open_file.close()

    # open_file = open(file_name, "rb")
    # loaded_list = pickle.load(open_file)
    # open_file.close()
    # print(loaded_list)


if __name__ == '__main__':
    make_list()
