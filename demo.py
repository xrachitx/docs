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

def load_image(filename, input_size=512):
    im = sio.imread(filename)
    # print("img: ",im.shape)
    h, w = im.shape[:2]
    if h>=w and h>input_size:
        im=zoom(im,(input_size/h,input_size/h,1))
        h, w = im.shape[:2]
    elif w>=h and w>input_size:
        im=zoom(im,(input_size/w,input_size/w,1))
        h, w = im.shape[:2]
    
    pad_top = (input_size - h)//2
    pad_lef = (input_size - w )//2
    pad_bottom = input_size - h - pad_top
    pad_right  = input_size - w  - pad_lef
    pad = ((pad_top, pad_bottom), (pad_lef, pad_right), (0,0))
    im_padded = np.pad(im, pad, 'constant', constant_values=0)
    im_padded = im_padded.astype(np.float32)
    im_padded /= 255
    im_padded = im_padded.transpose((2,0,1))
    im_padded = torch.from_numpy(np.expand_dims(im_padded, axis=0))
    # print(im.shape,im_padded.shape)
    
    return im, im_padded, pad

def load_gt(filename, input_size=512):
    im = sio.imread(filename)
    # print("gt: ",im.shape)
    h, w = im.shape[:2]
    if h>=w and h>input_size:
        im=zoom(im,(input_size/h,input_size/h))
        h, w = im.shape[:2]
    elif w>=h and w>input_size:
        im=zoom(im,(input_size/w,input_size/w))
        h, w = im.shape[:2]
    
    pad_top = (input_size - h)//2
    pad_lef = (input_size - w )//2
    pad_bottom = input_size - h - pad_top
    pad_right  = input_size - w  - pad_lef
    pad = ((pad_top, pad_bottom), (pad_lef, pad_right))
    im_padded = np.pad(im, pad, 'constant', constant_values=0)
    im_padded = im_padded.astype(np.float32)
    im_padded /= 255
    im_padded = np.where(im_padded>0.5,1,0)
    # exit()
    black,white = np.unique(im_padded,return_counts= True)[1]
    final = np.zeros((2,im_padded.shape[0],im_padded.shape[1]),dtype=float32)
    final[0,:,:] = 1-im_padded
    final[1,:,:] = im_padded
    # print(np.unique(im_padded))
    final = torch.from_numpy(np.expand_dims(final, axis=0))
    # print(im.shape,im_padded.shape)
    # print(torch.unique(f))
    # mask = torch.from_numpy(np.zeros((1,2,final.shape[-2],final.shape[-1]),dtype=float))
    # print(torch.unique(mask))
    # print("AMSKK: ",mask.dtype)
    return im, final, pad,black,white 

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
    img_path = "./data/Images/Aeroplane/Partial/"
    gt_path = "./data/Skeletons/Aeroplane/Partial/"
    imgs = os.listdir(img_path)

    for epoch in tqdm(range(epochs)):
      for im1 in tqdm(imgs):
          image_a_path = img_path + im1
          img_a, img_a_padded, pad_a= load_image(image_a_path)
          gt_a_path = gt_path + im1
          gt_a, gt_a_padded, _,black_a,white_a = load_gt(gt_a_path)
          gt_a_padded = gt_a_padded.to(device)
          img_a_padded = img_a_padded.to(device)
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

                  
                  gt_b_padded = gt_b_padded.to(device)

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
                  print("DTYPE OF BLACK PRED AND REAL: ",black_out_a.dtype, gt_a_padded[:,0,:,:].dtype)
                  la_black = criterion(black_out_a, gt_a_padded[:,0,:,:])
                  lb_black = criterion(black_out_b, gt_b_padded[:,0,:,:])
                  la_white = criterion(white_out_a, gt_a_padded[:,1,:,:])
                  lb_white = criterion(white_out_b, gt_b_padded[:,1,:,:])
                  la =   white_a/(black_a+white_a)*la_black + black_a/(black_a+white_a)*la_white
                  lb =   black_b/(black_b+white_b)*lb_black + white_b/(black_b+white_b)*lb_white
                  loss = la +lb
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
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
