import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RanzcrData import RanzcrDatasetClassification
from Model import CNN

'''
Author: Kayed Mahra
Main script for training the classification stage
'''

# Paths
mask_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train_mask_512'
img_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train_512'
labels_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train.csv'
annotations_path = r'C:\Users\Gnomechild\Desktop\ranzcr-data\Data_512\train_annotations.csv'

# Parameters
k_folds = 4
num_epochs = 30
num_classes = 11
batch_size = 10
initial_lr = 1e-4
architecture = 'efficientnet_b0'
DEBUG = False
DEBUG_DATASET_SIZE = 100
torch.manual_seed(1234)

if DEBUG:
	kfold_seed = 1234
else:
	kfold_seed = None
kfold = KFold(n_splits=k_folds, shuffle=True, random_state = kfold_seed)


# Setup GPU-mode if possible
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Helper functions
'''
Read sample IDs from mask folder to only use samples that have a binary mask annotation
Params: Path to mask folder
Returns: set of sample identifiers
'''
def read_sample_ids(mask_path):
    id_set = []
    try:
        for filename in os.listdir(mask_path):
            if filename.endswith('.png'):
                id_set.append(filename.replace('.png',''))
        return id_set
    except:
        print('Could not find mask directory with provided path, exiting...')
        sys.exit(1)
        
'''
Performs training on all batches in trainloader
Params: model, optimizer, criterion, trainloader
Returns: loss
'''        
def train(model_, optimizer_, criterion_, trainloader_, device_):
    train_loss = []
    model_.train()
    
    # Zero out gradients
    optimizer_.zero_grad()
    for batch_idx, (X_batch, y_batch) in enumerate(trainloader_):
        
        # Move batch to selected device
        X_batch, y_batch = X_batch.to(device_), y_batch.to(device_)
        
        # Compute prediction
        y_pred = model_(X_batch)
        loss = criterion_(y_pred, y_batch)
        train_loss.append(loss.item())
        loss.backward()
        optimizer_.step()
        
    return np.mean(train_loss)



'''
Performs validation on all batches in validloader
Params: model, optimizer, criterion, trainloader
Returns: loss, avg. auc over classes, auc per class
'''  
def valid(model_, criterion_, validloader_, device_, num_classes_):
    valid_loss = []
    pred_list = []
    targ_list = []
    auc_by_class = []
    
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
        
    pred_list = torch.cat(pred_list)
    pred_list[:, :] = pred_list[:,:].sigmoid()
    targ_list = torch.cat(targ_list).numpy()
    pred_list = pred_list.numpy()
    
    # Calculate ROC AUC score for each class
    for c_i in range(num_classes_):
        try:
            auc_by_class.append(roc_auc_score(targ_list[:, c_i], pred_list[:,c_i]))
        except:
            auc_by_class.append(0.5)
    
    return np.mean(valid_loss), np.mean(auc_by_class), auc_by_class

'''
Saves the fold specific metrics for analysis
params: root path, fold number, train loss, valid loss, aucs, aucs by class
Returns: Nothing
'''
def save_fold_metrics(root, fold, t_loss, v_loss, aucs, aucs_by_class):
    save_path = f'model-fold-{fold}.pth'
    torch.save(model.state_dict(), os.path.join(root,save_path))
    
    t_loss_path = f'train-loss-fold{fold}'
    v_loss_path = f'valid-loss-fold{fold}'
    aucs_path = f'aucs-fold{fold}'
    aucs_by_class_path = f'aucs-by-class-fold{fold}'
    
    np.save(os.path.join(root, t_loss_path), t_loss)
    np.save(os.path.join(root, v_loss_path), v_loss)
    np.save(os.path.join(root, aucs_path), aucs)
    np.save(os.path.join(root, aucs_by_class_path), aucs_by_class)







# Load sample ids and labels
id_set = read_sample_ids(mask_path)
labels = pd.read_csv(labels_path)

# For faster training when debugging
if DEBUG:
    id_set = id_set[:DEBUG_DATASET_SIZE]
    num_epochs = 5

# Load dataset
dataset = RanzcrDatasetClassification(labels, img_path, mask_path, id_set)

# Root folder for training session
root = datetime.now().strftime("%d%m%Y%H%M%S")
os.mkdir(root)

# Loss function
criterion = torch.nn.BCEWithLogitsLoss()

auc_over_folds = []

for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
    print(f'Starting fold: {fold}')
    
    # Setup subsamplers
    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = SubsetRandomSampler(valid_ids)
    
    # Setup current fold's loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)
    
    # Initialize model, optimizer, LR-scheduler
    model = CNN(architecture, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    
    # Setup variables for metric collection
    fold_v_loss = []
    fold_t_loss = []
    fold_aucs = []
    fold_aucs_by_class = []
    best_auc = 0.0
    
    # Training loop
    for epoch in range(0, num_epochs):
        print(f'Starting epoch: {epoch + 1}')
        
        # Train
        t_loss = train(model, optimizer, criterion, train_loader, device)
        
        # Valid
        v_loss, auc, auc_by_class = valid(model,criterion, valid_loader, device, num_classes)
        scheduler.step(v_loss)
        
        if auc > best_auc:
            print(f'NEW BEST AUC = {round(auc,4)}, saving model...')
            best_auc = auc
            save_path = f'model-fold-{fold}.pth'
            torch.save(model.state_dict(), os.path.join(root,save_path))
        
        # Save metrics for epoch
        fold_t_loss.append(t_loss)
        fold_v_loss.append(v_loss)
        fold_aucs.append(auc)
        fold_aucs_by_class.append(auc_by_class)
        
        # Print epoch summary
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string + ' EPOCH: ' + str(epoch) + ' Train loss: ' + str(round(t_loss,4)) + 
              ' Val loss: ' + str(round(v_loss,4)) + ' AUC: ' + str(round(auc,4)) + 
              ' LR: ' +str(optimizer.param_groups[0]['lr']))
        print('------------------------------------------------------------------')
        
    # Save metrics for fold
    save_fold_metrics(root, fold, np.array(fold_t_loss), np.array(fold_v_loss), 
                      np.array(fold_aucs), np.array(fold_aucs_by_class))
    # Save best auc
    auc_over_folds.append(best_auc)
    
print('------------------------------------------------------------------')
print('Finished training')
print(f'Best auc: {round(np.max(auc_over_folds),4)} from fold: {np.argmax(auc_over_folds)}')
print(f'Aucs from all folds: {[round(num, 4) for num in auc_over_folds]}')
print('------------------------------------------------------------------')