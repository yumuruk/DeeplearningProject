import sys
import os
import torch
import argparse
from datetime import datetime
import time
import random
import numpy as np

import torch.nn as nn
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from dataloader import UIEBDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    transform =[
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ]
    
    print("Training dataset loading...")
    train_dataloader = DataLoader(
        UIEBDataset(args.train_dir, transform=transform, img_size=args.image_size),
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,
    )
    print(f"{len(train_dataloader)} batches have been loaded")

    print("Testing dataset is loading...")
    test_dataloader = DataLoader(
        UIEBDataset(args.valid_dir, transform=transform, img_size=args.image_size), 
        batch_size=args.batch_size,  
        shuffle=True,
        num_workers=2,
    )
    print(f"{len(test_dataloader)} batches have been loaded")
    
    ### 0. parse_args
    
    ### 1. visdom (optional)
    
    ### 2. dataset and dataloader
    
    ### 3. network
    
    ### 4. loss
    
    ### 5. optimizer
    
    ### 6. scheduler
    
    ### 7. resume 
    
    ### 8. train
    
    ### 9. test
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Underwater_detection')
    parser.add_argument('--train_dir', type=str, default='/home/ptthuy/CODE/UNROLLED_UNDERWATER/UIEB_DATASET/train/',help='Training dataset')
    parser.add_argument('--val_dir', type=str, default='/home/ptthuy/CODE/UNROLLED_UNDERWATER/UIEB_DATASET/test/',help='Validation dataset')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--train_logs', type=str, default='./train_logs',help='Training logs and outputs')
    parser.add_argument('--seed', type=int, default=5, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='input batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='initial learning rate (default: 1e-4)')
    # parser.add_argument('--n_negative_samples', type=int, default=3, metavar='N', help='number of negative samples to train (default: 5)') # for contrastive learning
    parser.add_argument('--resume', type=str, default=None, help='Resume training from saved checkpoint(s).',)
    parser.add_argument('--image_size', type=int, default=416, help='Image size for training and testing')
    args = parser.parse_args()
    
    main(args)