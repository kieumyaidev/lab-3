from scipy.io import loadmat
import random
import os

###----------------------------------------------------------------------------------------------------------------------###
# 2. Import dataset

data_dir = []
data_path = os.path.join(os.getcwd(), 'SEED_IV', 'eeg_raw_data')
data_dir = [str(k) for k in range(1, 16)]

d = {
    '1': {},
    '2': {},
    '3': {}
}

for folder_num in [1,2,3]:
    for name in data_dir:
        name_path = os.path.join(data_path, str(folder_num), name + '.mat')   # Load EEG data from .mat file
        mat_file = loadmat(name_path)
        list_key = list(mat_file.keys())
        for i in range (24):
            d[str(folder_num)][str(i)] = mat_file[list_key[3+i]].T

# Channels and indexes (cf. Channels exploration)
channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 
            'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 
            'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

# Each session has 15 file present 15 subject. Each file contains 24 data of signal with labels list as below
SESSION_LABELS = { 
    '1': [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    '2': [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    '3': [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }
