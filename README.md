# Passenger-Counting-in-Mass-Public-Transport-Systems-using-Computer-Vision-and-Deep-Learning


The following is the code for the paper"Passenger Counting in Mass Public Transport Systems using Computer Vision and Deep Learning" in IEEE transactions.


This is an unofficial implementation of CVPR 2016 paper ["Single Image Crowd Counting via Multi Column Convolutional Neural Network"](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf).

During training, fine tuning was also completed by changing different hyper-parameters in the model. Using the MAE it was found that the variable kernel density maps do not reduce the loss function. The best training result was obtained with the fixed kernel density maps. To estimate the number of heads in the image, the sum of all the pixels of the density map is made.

# Train

For training you can use the code crowdcount-mcnn/train.py 

# Test

In crowdcount-mcnn/final_models/ you can find the final models to testing.

# Results
You can find the results of the neural network output with documentation and graphs

# Utils
In utils you can generate a density map

# archivos_XML
In archivos_XML you can find all the labels of the head counting (GT)

# crowdcount-mcnn

Original repository and modifications of the fine tuning with our data.

