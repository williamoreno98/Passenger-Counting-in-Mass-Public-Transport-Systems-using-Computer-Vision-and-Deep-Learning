from utils import boundingbox as bb
import cv2

''' 
Script para comprobar lectura de anotaciones exportadas de CVAT,
marcándolas en un video
'''


def read_annotations():

    archivo = 'annotationsTM_15.xml'
    boxes = bb.boxes_from_xml(archivo)

    cap = cv2.VideoCapture('videoTM_15.mkv')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_boxes.avi', fourcc, 30.0, (640, 480))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            bx_in_frame = bb.boxes_in_frame(i, boxes)
            img = frame
            for bx in bx_in_frame:
                img = bb.draw_box(bx, img)

            i += 1
            out.write(img)
            cv2.imshow('video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_annotations()
