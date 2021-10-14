"""
Python: 3.6.9
Author: lakj
Charset: UTF-8
"""
from src.dataloader import Coseg
from src.models import Model
import src.utils as utils
from torch.nn.functional import softplus
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
import os
torch.multiprocessing.set_sharing_strategy('file_system')

def split_i(array:list, i:int) -> (list, list):
    """ Given an array it splits the array in two parts at index i

    """
    if i==len(array)-1:
        return array[i], array[:-1]
    else:
        pre = array[0:i]
        post = array[i+1:]
        l = pre + post
        x = array[i]
        return x, l
    

def precompute_features(imgs:list, GTs:list, phi:models) -> list:
    """ Since the architecture uses features computed through the image and its mask
    we precompute them to speed up the usage at runtime (Equation (13))

    """
    G_Ts = [1 - GTn for GTn in GTs]
    IGms = [GTn * In for GTn, In in zip(GTs, imgs)] 
    I_Gms = [G_Tn * In for G_Tn, In in zip(G_Ts, imgs)] 
    features = [(phi(IGm), phi(I_Gm)) for IGm, I_Gm in zip(IGms, I_Gms)]
    return features
   
    
def Ls(GTn:torch.tensor, Mn:torch.tensor) -> torch.tensor:
    """ Cross entropy loss for individual supervision, Equation (12)
    """
    return (-(GTn * torch.log(Mn+1e-15) + (1- GTn) * torch.log((1- Mn)+1e-15))).sum()

def Ls_modified(GTn:torch.tensor, Mn:torch.tensor, Wn: torch.tensor) -> torch.tensor:
    """ Cross entropy loss for individual supervision, Equation (12)
    """
    x = (-(GTn * torch.log(Mn+1e-15) + (1- GTn) * torch.log((1- Mn)+1e-15)))
    x = Wn*x
    x = x.sum()
    return x

def L2(GTn:torch.tensor, Mn:torch.tensor, Wn: torch.tensor) -> torch.tensor:
    """ l2 loss with pixel weighting, Equation (12)
    """
    x = torch.sqrt((GTn - Mn)**2+1e-016)
    x = Wn*x
    x = x.sum()
    return x

def Lc(i:int, imgs:list, masks:list, features:list, phi:models,DEVICE:str) -> torch.tensor:
    """ Triplet loss group wise constraint Equation (14)
    """
    # phi = phi.to(DEVICE)
    Ion = phi(masks[i] * imgs[i])
    # masks[i] = masks[i].cpu()
    # imgs[i] = imgs[i].cpu
    # phi = phi.cpu()
    fi, fts = split_i(features, 1)
    cumsum = 0
    for IGm, I_Gm in fts:
        IGm = IGm.to(DEVICE)
        I_Gm = I_Gm.to(DEVICE)
        P = torch.sqrt(((Ion - IGm)**2)+1e-016)
        N = torch.sqrt(((Ion - I_Gm)**2)+1e-016)
        cumsum += softplus(P-N).sum()
        IGm = IGm.cpu()
        I_Gm = I_Gm.cpu()
    cumsum /= len(fts)
    return cumsum


################################################################################
############################# Main script starts ###############################
################################################################################

def main():

    # Params
    DEVICE = 'cuda'
    GROUP_SIZE = 55
    EPOCHS = 250
    TBOARD = False # If you have tensorboard running set it to true


    categories = os.listdir("./data/images/")
    data = {}
    vgg19_original = models.vgg19()
    phi = nn.Sequential((*(list(vgg19_original.children())[:-2])))
    for param in phi.parameters():
        param.requires_grad = False
    phi = phi.to(DEVICE)


    for category in categories:
        # Load data
        print(f"Loading Data for {category}")
        coseg = Coseg(category)
        trloader = DataLoader(coseg, batch_size=1, shuffle=False, num_workers=1)
        imgs = []
        GTs = []
        weights = []
        for i, (In, GTn,weight) in enumerate(trloader):
            if i == GROUP_SIZE:
                break
            else:

                imgs.append(In)
                GTs.append(GTn)
                weights.append(weight)
                imgs[i] = imgs[i].to(DEVICE)
                GTs[i] = GTs[i].to(DEVICE)
                weights[i] = weights[i].to(DEVICE)

        # Precompute features
        print(f"[ OK ] Data loaded: {category}")

        features = precompute_features(imgs, GTs, phi)

        print(f"[ OK ] Feature precomputed: {category}")

        data[category] = [imgs,GTs,weights,features]

    # print("[ OK ] Feature precomputed")



    # exit()
    

    # Instantiate the model
    if DEVICE == 'cuda':
        groupnet = Model((1,3, 224, 224)).cuda()
    else:
        groupnet = Model((1,3, 224, 224))
    print("[ OK ] Model instantiated")

    # Optimizer
    # [ PAPER ] suggests SGD with these parametes, but desn't work
    #optimizer = optim.SGD(groupnet.parameters(), momentum=0.99,lr=0.00005, weight_decay=0.0005)
    optimizer = optim.Adam(groupnet.parameters(), lr=0.00002)

    # Train Loop
    losses = []
    if TBOARD:
        writer = SummaryWriter()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        # lss_black = 0
        # lcs_black = 0
        # lss_white = 0
        # lcs_white = 0
        lss = 0
        lcs = 0
        for category in categories:

            imgs = data[category][0]
            GTs = data[category][1]
            weights = data[category][2]
            features = data[category][3]

            # for i in range(len(imgs)):

                # features[i] = features[i].to(DEVICE)
            # In = In.to(DEVICE)
            # GTn = GTn.to(DEVICE)
        
            masks = groupnet(imgs)
            # print(len(masks),masks[0][:,1,:,:].shape)
            # exit()
            for i in range(len(imgs)):
                lss += L2(masks[i], GTs[i],weights[i])
                # [ PAPER ] suggests to activate group loss after 100 epochs
                if epoch >= 100:
                    lcs += Lc(i, imgs, masks, features, phi,DEVICE)
        
        lss /= len(imgs)
        
        if epoch >= 100:  
            lcs /= len(imgs)
                
            # for i in range(len(imgs)):
            #     imgs[i] = imgs[i].cpu()
            #     GTs[i] = GTs[i].cpu()
            #     weights[i] = weights[i].cpu()
                # features[i] = features[i].cpu()

        # [ PAPER ] suggests 0.1, but it does not work
        loss = lss + 1.*lcs 
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f'[ ep {epoch}, cat {category} ] - Loss: {loss.item():.4f}')
        if (epoch+1)%50==0:
            for category in categories:
                fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(10,5))
                imgs = data[category][0]
                GTs = data[category][1]
                weights = data[category][2]
                features = data[category][3]
                for i in range(5):
                    axs[0,i].imshow(imgs[i].detach().cpu().numpy().squeeze(0).transpose(1,2,0))
                    axs[0,i].axis('off')
                    axs[1,i].imshow(GTs[i].detach().cpu().numpy().squeeze(0).squeeze(0))
                    axs[1,i].axis('off')
                    axs[2,i].imshow(masks[i].detach().cpu().numpy().squeeze(0).squeeze(0))
                    axs[2,i].axis('off')
                plt.savefig(f"../../outs/predictions_{category}_{epoch}.png")
                plt.close()
        if TBOARD:
            writer.add_scalar("loss", loss.item(), epoch)
            utils.tboard_imlist(masks, "masks", epoch, writer)
        losses.append(loss.item())
        # print(f'[ ep {epoch} ] - Loss: {loss.item():.4f}')

    if TBOARD:
        writer.close()


    # Plot results in the same folder 




    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax.plot(losses)
    if epoch>100:
        ax.axvline(100, c='r', ls='--', label="Activate Lc loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig("loss.png")
    plt.close()
    print("[ OK ] Plot")
    print("Before Changing Current Directories",os.getcwd())
    os.chdir("../")
    print("After Changing Current Directories",os.getcwd())
    print(os.getcwd())
    torch.save(groupnet,os.getcwd()+"model.pth")
    print("[ OK ] Saved")



if __name__ == "__main__":
    main()
