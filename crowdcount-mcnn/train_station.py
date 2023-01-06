import os
import torch
import numpy as np
import sys
import cv2 as cv
import random

import matplotlib.pyplot as plt
from pylab import *
from src.crowd_count import CrowdCounter
from src import network
from src.network import load_net
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model
from skimage.feature import peak_local_max
from torch.utils.tensorboard import SummaryWriter

pepoch,pmae,pmae1,pmae2,pmae3,pmae4=[],[],[],[],[],[]
total2=0

writer = SummaryWriter(log_dir='train_station_1.1')
try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


print(torch.__version__)
method = 'station'
dataset_name = 'train_final_1.1'
output_dir = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/saved_models'
train_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/train_data/image_train/station/1'
train_gt_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/train_data/gtfx_train/station/1'


val_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/val_data/image_val/station/1'
val_gt_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/val_data/gtfx_val/station/1'

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
best_mae = sys.maxsize

model_path = '/data/estudiantes/william/PdG-Code/data_prep/crowdcount-mcnn/final_models/mcnn_shtechA_660.h5'

#training configuration
start_step = 0
end_step = 15000
lr = 0.00001
momentum = 0.9
disp_interval = 500
log_interval = 250


#Tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = True   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


# load net
net = CrowdCounter()
trained_model = os.path.join(model_path)
load_net(trained_model, net)
net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='localhost',port=6006)
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:    
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

a=2

while(True):

    for epoch in range(start_step, end_step+1):
        step = -1
        train_loss = 0
        for blob in data_loader:
            step = step + 1
            im_data = blob['data']
            gt_data = blob['gt_density']
            #gt_count = blob['head']
            density_map = net(im_data, gt_data*1.1)
            loss = net.loss
            train_loss += loss.data
            step_cnt += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            duration = t.toc(average=False)
            fps = step_cnt / duration
            density_map = density_map.data.cpu().numpy()
            et_count = (np.sum(density_map))
            gt_count=(np.sum(gt_data))
            writer.add_scalar("Loss/train", loss, epoch)
            #writer.flush()
            if step%100==0:
                log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f, a %4.1f' % (epoch,
                    step, 1./fps, gt_count,et_count,a-1)
                log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True

            if re_cnt:
                t.tic()
                re_cnt = False

        if (epoch%5==0):
            save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
            network.save_net(save_name, net)
            #calculate error on the validation dataset
            total2=total2+1
            mae,mae1,mae2,mae3,mae4,mse,mse1,mse2,mse3,mse4 = evaluate_model(save_name, data_loader_val)
            pmae.append(mae)
            pmae1.append(mae1)
            pmae2.append(mae2)
            pmae3.append(mae3)
            pmae4.append(mae4)
            pepoch.append(epoch)

            if mae < best_mae:
                best_mae = mae
                best_mse = mse
                best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
            log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch,mae,mse)
            log_print(log_text, color='green', attrs=['bold'])
            log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae,best_mse, best_model)
            log_print(log_text, color='green', attrs=['bold'])
            if use_tensorboard:
                exp.add_scalar_value('MAE', mae, step=epoch)
                exp.add_scalar_value('MSE', mse, step=epoch)
                exp.add_scalar_value('train_loss', train_loss/data_loader.get_num_samples(), step=epoch)

        if(epoch%25==0 and epoch!=0):
            train_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/train_data/image_train/station/%d'%a
            train_gt_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/train_data/gtfx_train/station/%d'%a
            #b=random.randint(1,9)
            val_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/val_data/image_val/station/%d'%a
            val_gt_path = '/data/estudiantes/william/PdG-Code/data_prep/data_class/val_data/gtfx_val/station/%d'%a
            best_mae = sys.maxsize
            data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
            data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
            a+=1
            if (a == 99):
                break
    if (a == 99):
        break

plt.plot(pepoch, pmae1)
plt.plot(pepoch, pmae2)
plt.plot(pepoch, pmae3)
plt.plot(pepoch, pmae4)
legend(('Nivel 1', 'Nivel 2', 'Nivel 3', 'Nivel 4'),
prop = {'size': 5}, loc='upper left')
grid()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE vs Número epochs de entrenamiento')
plt.savefig('loss_train_epochs_niveles.png')

cla()   # Clear axis
clf()   # Clear figure
close() # Close a figure window

plt.plot(pepoch, pmae)
grid()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE vs Número de epochs en entrenamiento') 
plt.savefig('loss_train_total.png')

cla()   # Clear axis
clf()   # Clear figure
close() # Close a figure window

import csv
import os

with open("datos_train_1.1.csv", "w") as csv_file:   
    writer = csv.writer(csv_file, delimiter=',')
    level_counter = 0
    max_levels = total2
    while level_counter < max_levels:
        if(level_counter==0):
            writer.writerow(("Epoch","MAE total","MAE Nivel 1","MAE Nivel 2","MAE Nivel 3","MAE Nivel 4")) 
        writer.writerow((pepoch[level_counter],round(pmae[level_counter],2),round(pmae1[level_counter],2),round(pmae2[level_counter],2),round(pmae3[level_counter],2),round(pmae4[level_counter],2))) 
        level_counter = level_counter +1 