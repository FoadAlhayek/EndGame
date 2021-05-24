from UNet import UNet
from RanzcrData import RanzcrDatasetSegmentation
import numpy as np
import os
import cv2
import torch
from matplotlib import pyplot as plt
from datetime import datetime
from torchsummary import summary
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Paths
mask_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train_mask_512'
img_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train_512'
labels_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train.csv'
annotations_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train_annotations.csv'

# Params
in_channels = 1
out_channels = 2
num_epochs = 8
batch_size = 5
initial_lr= 1e-4
DEBUG = False
DEBUG_DATASET_SIZE = 100
torch.manual_seed
k_folds = 4

# Setup GPU-mode if possible
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Helpers
def read_sample_ids(path, extension):
    id_set = {0}
    try:
        for filename in os.listdir(path):
            if filename.endswith(extension):
                id_set.add(filename.replace(extension,''))
        id_set.remove(0)
        return id_set
    except:
        print('Could not find mask directory with provided path, exiting...')
        sys.exit(1)

        
def train(model_, optimizer_, criterion_, trainloader_, device_):
    train_loss = []
    model_.train()
    
   
    for batch_idx, (X_batch, y_batch) in enumerate(trainloader_):
         # Zero out gradients
        optimizer_.zero_grad()
        
        # Move batch to selected device
        X_batch, y_batch = X_batch.to(device_), y_batch.to(device_)
        
        # Compute prediction
        y_pred = model_(X_batch)
        loss = criterion_(y_pred, y_batch)
        train_loss.append(loss.item())
        loss.backward()
        optimizer_.step()
        
    return np.mean(train_loss) 


def valid(model_, criterion_, validloader_, device_):
    valid_loss = []
    pred_list = []
    targ_list = []

    
    model.eval()
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(validloader_):
            
            # Move batch to selected device
            X_batch, y_batch = X_batch.to(device_), y_batch.to(device_)

            # Predict and compute loss
            y_pred = model_(X_batch)
            loss = criterion_(y_pred, y_batch)

            # Save loss, predictions and targets
            valid_loss.append(loss.item())
            pred_list.append(y_pred.cpu())
            targ_list.append(y_batch.cpu())
          
    return np.mean(valid_loss)

    
# Load ids
mask_ids = read_sample_ids(mask_path, '.png')
img_ids = read_sample_ids(img_path, '.jpg')
un_masked_images = mask_ids ^ img_ids

# For faster training when debugging
if DEBUG:
    mask_ids = list(mask_ids)[:DEBUG_DATASET_SIZE]
    num_epochs = 5
    k_folds = 2

# Load dataset
dataset = RanzcrDatasetSegmentation(img_path, mask_path, list(mask_ids))

# Loss function
criterion = torch.nn.BCEWithLogitsLoss()
kfold = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
    print(f'Starting fold: {fold}')
    
    # Setup subsamplers
    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = SubsetRandomSampler(valid_ids)
    
    # Setup current fold's loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)
    
    # Initialize model, optimizer, LR-scheduler
    model = UNet(in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=6,  verbose=True)
    
    # Setup variables for metric collection
    fold_v_loss = []
    fold_t_loss = []
    best_valid_loss = 10^5
    
    for epoch in range(0, num_epochs):
        print(f'Starting epoch: {epoch + 1}')
        
        # Train
        t_loss = train(model, optimizer, criterion, train_loader, device)
        
        # Valid
        v_loss = valid(model, criterion, valid_loader, device)
        scheduler.step(v_loss)
        
        if v_loss < best_valid_loss:
            best_valid_loss = v_loss
            print(f'NEW BEST VAL LOSS = {round(best_valid_loss,4)}, saving model...')
            save_path = f'model-fold-{fold}-segmentation.pth'
            torch.save(model.state_dict(), save_path)
            
        
        fold_v_loss.append(v_loss)
        fold_t_loss.append(t_loss)
        
        # Print epoch summary
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string + ' EPOCH: ' + str(epoch) + ' Train loss: ' + str(round(t_loss,4)) + 
              ' Val loss: ' + str(round(v_loss,4)) + 
              ' LR: ' +str(optimizer.param_groups[0]['lr']))
        print('------------------------------------------------------------------')
    np.save(f'v_loss_seg_{fold}.npy', fold_v_loss)
    np.save(f't_loss_seg_{fold}.npy', fold_t_loss)            
