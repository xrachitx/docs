import os
import skimage.io as sio
from scipy.ndimage import zoom
import numpy as np

class DOCS_Data(Dataset):
    def __init__(self, root_dir,category, image_height = 28, image_width = 28):
        if root_dir[-1] !="/":
            root_dir+="/"
        self.img_dir = f"{root_dir}Images/{category}/"
        self.gt_dir = f"{root_dir}Skeletons/{category}/"
        self.data = os.listdir(self.img_dir)
        self.data = [[self.img_dir+d,self.gt_dir+d] for d in self.data]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path,gt_path = self.data[idx][0],self.data[idx][1]
        _, image, _ = self.load_image(img_path)
        _, image_label, _ = self.load_image(gt_path)
        return image, image_label
    
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
        im_padded = np.expand_dims(im_padded, axis=0)
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
        hello = np.unique(im_padded,return_counts= True)
        print(hello)
        exit()
        final = np.zeros((2,im_padded.shape[0],im_padded.shape[1]),dtype=int)
        final[0,:,:] = 1-im_padded
        final[1,:,:] = im_padded
        # print(np.unique(im_padded))
        final = np.expand_dims(final, axis=0)
        # print(im.shape,im_padded.shape)
        # print(torch.unique(f))
        # mask = torch.from_numpy(np.zeros((1,2,final.shape[-2],final.shape[-1]),dtype=float))
        # print(torch.unique(mask))
        # print("AMSKK: ",mask.dtype)
        return im, final, pad