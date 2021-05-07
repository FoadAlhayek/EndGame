import cv2
import os
import shutil
import sys
import copy
import pandas as pd
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt
from PIL import Image
import ast

'''
Author: Kayed Mahra
Instructions: Make sure script is in same dir level as the required files: train_annotations.csv, train.csv, train (img dir), test (img dir)
Change DIM param and MULTI_PROCESS FLAG (Default True)
Will generate a new folder Data_DIM with subfolders: train_resized, test_resized, train_masks, train_annotations.csv (unchanged), train.csv (unchanged)
'''

# Identifiers & Params
STUDY_ID = 'StudyInstanceUID'
DIM = 1024 # Change if needed
MULTI_PROCESS = True

# Src paths
src_annotations = r'data/train_annotations.csv'
src_ground_truth = r'data/train.csv'
src_train_imgs = r'data/train'
src_test_imgs = r'data/test' 
# Dst paths
root = r'Data_' + str(DIM)
dst_train_imgs = root + '/train_' + str(DIM)
dst_test_imgs = root + '/test_' + str(DIM)
dst_train_masks = root + '/train_mask_' + str(DIM)

# Methods
def check_all_paths_valid(path1, path2, path3, path4):
	data_exists = os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3) and os.path.exists(path4)
	if data_exists:
		print('successfully found required data')
	else:
		print('Could not find required files/folders')
		sys.exit(1)


def setup_directories(root, subdirs, file1, file2):
	try:
		os.mkdir(root)

		for i in range(len(subdirs)):
			os.mkdir(subdirs[i])

		print('Directory structure created')

	except FileExistsError:
		print('Directory already exists')

	try:   
		shutil.copy(file1, root + '/')
		shutil.copy(file2, root + '/')
		print('Successfully copied files to destination')

	except IOError:
		print('IOError occured while attempting to copy files')
		sys.exit(1)


def read_csv_data(csv_path):
	try:
		DF = pd.read_csv(csv_path)
		return DF
	except IOError:
		print(f'Could not read csv with path {csv_path}')

def build_set_of_sample_IDs(sample_dir):
	id_set = {''}
	for filename in os.listdir(sample_dir):
		if filename.endswith('.jpg'):
			id_set.add(filename.replace('.jpg',''))
	id_set.remove('')
	return id_set

def build_dataset(train_ids, test_ids, DF):
	# Resizing test images
	pid = str(os.getpid())
	num_img = len(test_ids)
	counter = 0
	try:
		for t_id in test_ids:
			img_src = os.path.join(src_test_imgs, t_id + '.jpg')
			img_dst = os.path.join(dst_test_imgs, t_id + '.jpg')
			img = cv2.imread(img_src)
			img = cv2.resize(img, (DIM,DIM))
			cv2.imwrite(img_dst, img)
			counter +=1
			if counter % 100 == 0:
				print(pid + ' Resized and saved test img ' + str(counter) + ' out of ' + str(num_img))
		print(pid + ' Successfully resized ' + str(counter) + ' test images')
	except IOError:
		print('Could not read/write all test images')


	# Resizing train images and generating masks
	num_img = len(train_ids)
	counter = 0
	try:
		for t_id in train_ids:
			# Resize image
			img_src = os.path.join(src_train_imgs, t_id + '.jpg')
			img_dst = os.path.join(dst_train_imgs, t_id + '.jpg')
			img = cv2.imread(img_src)
			h, w = img.shape[0], img.shape[1]
			img = cv2.resize(img, (DIM,DIM))
			cv2.imwrite(img_dst, img)

			# Generate binary mask
			df_sample = DF.query(STUDY_ID + f'== "{t_id}"')

			if df_sample.shape[0] > 0: # Sample has annotations
				mask_dst = os.path.join(dst_train_masks, t_id + '.png')
				mask = np.zeros((h,w)).astype(np.uint8)

				for _, annot in df_sample.iterrows(): # Collect samples in same mask and interpolate points
					pixels = np.array(ast.literal_eval(annot["data"]))
					mask = cv2.polylines(mask, np.int32([pixels]), isClosed=False, color=1, thickness=15, lineType=16)

				mask = cv2.resize(mask, (DIM,DIM))
				mask = (mask > 0.5).astype(np.uint8)
				cv2.imwrite(mask_dst, mask)

			counter +=1
			if counter % 100 == 0:
				print(pid + ' Resized and saved train img ' + str(counter) + ' out of ' + str(num_img))
		print(pid + ' Successfully resized' + str(counter) + ' train images')
	except IOError:
		print('Could not read/write all train images')

if __name__ == "__main__":
	subdirs = [dst_train_imgs, dst_test_imgs, dst_train_masks]
	check_all_paths_valid(src_annotations, src_test_imgs, src_train_imgs, src_ground_truth)
	setup_directories(root, subdirs, src_annotations, src_ground_truth)
	train_ids = build_set_of_sample_IDs(src_train_imgs)
	test_ids = build_set_of_sample_IDs(src_test_imgs)
	DF = read_csv_data(src_annotations)

	if MULTI_PROCESS and multiprocessing.cpu_count() > 3:
		cores = multiprocessing.cpu_count() - 1
		train_ids = list(train_ids)
		test_ids = list(test_ids)
		split_idx_test = np.floor(len(test_ids) / cores).astype(np.uint32)
		split_idx_train = np.floor(len(train_ids) / cores).astype(np.uint32)

		batch1_test = test_ids[:split_idx_test].copy()
		batch2_test = test_ids[split_idx_test:2*split_idx_test].copy()
		batch3_test = test_ids[2*split_idx_test:].copy()

		batch1_train = train_ids[:split_idx_train].copy()
		batch2_train = train_ids[split_idx_train:2*split_idx_train].copy()
		batch3_train = train_ids[2*split_idx_train:].copy()
		
		p1 = multiprocessing.Process(target=build_dataset, args=(batch1_train, batch1_test, DF.copy(),))
		p2 = multiprocessing.Process(target=build_dataset, args=(batch2_train, batch2_test, DF.copy(),))
		p3 = multiprocessing.Process(target=build_dataset, args=(batch3_train, batch3_test, DF.copy(),))

		p1.start()
		p2.start()
		p3.start()
		p1.join()
		p2.join()
		p3.join()
		print('Finished generating new dataset')

	else:
		build_dataset(train_ids, test_ids, DF)
		print('Finished generating new dataset')
	
