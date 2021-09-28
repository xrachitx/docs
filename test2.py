import torch
import argparse
import numpy as np
import skimage.io as sio
from scipy.ndimage import zoom
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
# from torchsummary import summary
from torchvision import datasets, transforms
from models.docs import DOCSNet
import os
from tqdm import tqdm
from data_loader import DOCS_Data

def remove_pad(a, pad):
    return a[pad[0][0]:a.shape[0]-pad[0][1],pad[1][0]:a.shape[1]-pad[1][1]]

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Object Co-Segmentation (DOCS) Demo: '
						 'Given two input images, segments the common objects within two images.')
    parser.add_argument('gpu', metavar='GPU', type=int, help='gpu-id')
    parser.add_argument('image_a_path', metavar='IMG_A_PATH', help='path to first image.')
    parser.add_argument('image_b_path', metavar='IMG_B_PATH', help='path to second image.')
    parser.add_argument('snapshot', metavar='SNAPSHOT_PATH', help='paht to model\'s snapshot.')
    return parser.parse_args()

def main():
    # args = parse_args()

    # rgb_means = [122.67892, 116.66877, 104.00699]

    # set the device
    if not torch.cuda.is_available():
        raise RuntimeError('You need gpu for running this demo.')
    device = torch.device('cuda:0')
    print('Device:', device)

    print('Setting up the network...')
    # state = torch.load(args.snapshot, map_location='cpu')
    net = DOCSNet()
    # net.load_state_dict(state['net_params'])
    net.train()
    net.to(device)
    for params in net.parameters():
        params.requires_grad = True
    lr = 0.0001
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr = lr)
    epochs = 20
    batch_size = 10
    dataset = DOCS_Data("./data/","Aeroplane")
    train_loader1 = DataLoader(dataset=dataset,batch_size=1,shuffle=True)
    # img_path = "./data/Images/Aeroplane/Partial/"
    # gt_path = "./data/Skeletons/Aeroplane/Partial/"
    # imgs = os.listdir(img_path)

    for epoch in tqdm(range(epochs)):
        for img_a_padded, gt_a_padded,black_a,white_a in train_loader1:
            # image_a_path = img_path + im1
            # img_a, img_a_padded, pad_a= load_image(image_a_path)
            # gt_a_path = gt_path + im1
            # gt_a, gt_a_padded, _,black_a,white_a = load_gt(gt_a_path)
            gt_a_padded = gt_a_padded.to(device).float()
            img_a_padded = img_a_padded.to(device)
            print(img_a_padded.shape,gt_a_padded.shape,black_a,white_a)
            exit()
            #   mask = mask.to(device)
            # mean_mask = np.zeros_like()
            for im2 in imgs:

                image_b_path = img_path + im2
                if image_a_path!=image_b_path:

                    gt_b_path = gt_path + im2
                    # load img_b
                    img_b, img_b_padded, pad_b= load_image(image_b_path)
                    # load gt_b
                    gt_b, gt_b_padded, _,black_b,white_b= load_gt(gt_b_path)

                    
                    gt_b_padded = gt_b_padded.to(device).float()

                    img_b_padded = img_b_padded.to(device)
                    out_a, out_b = net.forward(img_a_padded, img_b_padded, softmax_out=True)
                    black_out_a = out_a[:,0,:,:]
                    black_out_b = out_b[:,0,:,:]
                    white_out_a = out_a[:,1,:,:]
                    white_out_b = out_b[:,1,:,:]
                    #   print(black_out_a.shape,gt_a_padded.shape)
                    #   exit()
                    #   print("outa: ",out_a.shape,gt_a_padded.shape,"mask: ", mask.shape)
                    #   mask +=out_a
                    la_black = criterion(black_out_a, gt_a_padded[:,0,:,:])
                    lb_black = criterion(black_out_b, gt_b_padded[:,0,:,:])
                    la_white = criterion(white_out_a, gt_a_padded[:,1,:,:])
                    lb_white = criterion(white_out_b, gt_b_padded[:,1,:,:])
                    la =   white_a/(black_a+white_a)*la_black + black_a/(black_a+white_a)*la_white
                    lb =   white_b/(black_b+white_b)*lb_black + black_b/(black_b+white_b)*lb_white
                    loss = la +lb
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        print(f"Epoch: {epoch}-------Loss: {loss.item()}")
            
        #   mask /= len(imgs)-1

    torch.save(net,"model.pth")
    # result_a = remove_pad(out_a[0,1].cpu().detach().numpy(), pad_a)>0.5
    # result_b = remove_pad(out_b[0,1].cpu().detach().numpy(), pad_b)>0.5

    # filtered_img_a = img_a * np.tile(result_a,(3,1,1)).transpose((1,2,0))
    # filtered_img_b = img_b * np.tile(result_b,(3,1,1)).transpose((1,2,0))

    # plt.subplot(2,2,1)
    # plt.imshow(img_a)
    # plt.subplot(2,2,2)
    # plt.imshow(img_b)
    # plt.subplot(2,2,3)
    # plt.imshow(filtered_img_a)
    # plt.subplot(2,2,4)
    # plt.imshow(filtered_img_b)
    # plt.show()

if __name__ == '__main__':
    main()
