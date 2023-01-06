from src.crowd_count import CrowdCounter
from src.network import load_net
import numpy as np


def evaluate_model(trained_model, data_loader):
    net = CrowdCounter()
    load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae,mae1,mae2,mae3,mae4 = 0.0,0.0,0.0,0.0,0.0
    mse,mse1,mse2,mse3,mse4 = 0.0,0.0,0.0,0.0,0.0
    a=0
    b=0
    c=0
    d=0
    total=0

    for blob in data_loader:      
        total=total+1                  
        im_data = blob['data']
        gt_data = blob['gt_density']
        #gt_count= blob['head']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = round(np.sum(gt_data))
        et_count = round(np.sum(density_map))

        if(gt_count<=3):
            mae1 += abs(gt_count-et_count)
            mse1 += ((gt_count-et_count)*(gt_count-et_count))
            a=a+1
        if(gt_count>=4 and gt_count<=9):
            mae2 += abs(gt_count-et_count)
            mse2 += ((gt_count-et_count)*(gt_count-et_count))
            b=b+1
        if(gt_count>=10 and gt_count<=25):
            mae3 += abs(gt_count-et_count)
            mse3 += ((gt_count-et_count)*(gt_count-et_count))
            c=c+1
        if(gt_count>=26):
            mae4 += abs(gt_count-et_count)
            mse4 += ((gt_count-et_count)*(gt_count-et_count))
            d=d+1
        
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))

    if (a==0):
        mae1=2
        a=1
    if (b==0):
        mae2=2
        b=1
    if (c==0):
        mae3=2
        c=1
    if (d==0):
        mae4=2
        d=1

    mae1 = mae1/a
    mse1 =  np.sqrt(mse1/a)

    mae2 = mae2/b
    mse2 =  np.sqrt(mse2/b)

    mae3 = mae3/c
    mse3 =  np.sqrt(mse3/c)
    
    mae4 = mae4/d
    mse4 =  np.sqrt(mse4/d)

    mae = mae/total
    mse =  np.sqrt(mse/total)

    return mae,mae1,mae2,mae3,mae4,mse,mse1,mse2,mse3,mse4