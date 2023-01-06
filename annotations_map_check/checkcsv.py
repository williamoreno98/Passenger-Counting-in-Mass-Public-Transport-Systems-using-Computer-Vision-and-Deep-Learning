import numpy as np  # linear algebra
import csv
from skimage.feature import peak_local_max
import cv2

'''
Para leer y mostrar un mapa en csv
'''
# Path al csv
file_path = "./data_class/test_data/gt_test/IMG_10.csv"
results = []
with open(file_path) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
    for row in reader:  # each row is a list
        results.append(row)

groundtruth = np.asarray(results, dtype=np.uint8)
nhead = peak_local_max(groundtruth, threshold_abs=50)
print(nhead.shape[0])
# print(groundtruth.shape)
cv2.imshow('map', groundtruth)
cv2.waitKey(0)
cv2.destroyAllWindows()