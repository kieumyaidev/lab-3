import torch
import os

SAMPLING_FQ = 128  # Example sampling frequency, adjust as needed
FQ_MIN = 0.5
NB_FQ = 36
FQ_STEP = 0.5
RESTRICTED_FQ = [FQ_STEP*fq for fq in range(int(FQ_MIN/FQ_STEP), NB_FQ+1)]
# Warning : frequencies must be regularly spaced

###----------------------------------------------------------------------------------------------------------------------###
# Wavelet hyperparameter

win_time_wavelet = 8

WINDOW_TIME_ARRAY  = []
for i in range(1, 50):
    if (2 * 60) % i == 0:
        WINDOW_TIME_ARRAY.append(i)
OVERLAP = [8, 16, 32, 64, 128, 192, 256, 320, 384]
LEVEL_ARRAY = [4, 5, 6, 7, 8]

NUMBER_WINDOW_LENGTH = len(WINDOW_TIME_ARRAY)
NUMBER_WINDOW_SHIFT = len(OVERLAP)

WINDOW_TIME_ARRAY_STR = [str(i) for i in WINDOW_TIME_ARRAY]
OVERLAP_STR = [str(i) for i in OVERLAP] 

###----------------------------------------------------------------------------------------------------------------------###
# STFT hyperparameter
# STFT Article values
DEFAULT_STFT_PARAMETERS = {'restricted_frequencies': RESTRICTED_FQ,
                           'offset': 0,
                           'window_size': 15,
                           'window_shift': 128,
                           'window_type': "blackman",
                           'averaging': True,
                           'avg_filter_size': 15}

w_size = 4
w_shift = 130

selection_STFT_PARAMETERS = {'restricted_frequencies': RESTRICTED_FQ,
                                'offset': 0,
                                'window_size': w_size,
                                'window_shift': w_shift,
                                'window_type': "blackman",
                                'averaging': True,
                                'avg_filter_size': 15}

###----------------------------------------------------------------------------------------------------------------------###
# SS-MT hyperparameter

NO_TAPERS = [1, 2, 3, 4, 5]


alpha = 1
beta = 1
observationNoise = 100
GUESS_WINDOW_LENGTH = 150

TW = 2  # Time-bandwidth production
K = 3   # The Number of tapers

fmax = 50
NO_WINDOW_ARRAY = len(WINDOW_TIME_ARRAY)
WINDOW_TIME_ARRAY_INVERSE = [8, 10]

###----------------------------------------------------------------------------------------------------------------------###
# LSTM hyperparameter

hidden_size_LSTM = 64
num_layers_LSTM = 1
num_classes_LSTM = 3
learning_rate_LSTM = 0.001
num_epochs_LSTM = 50
batch_size_LSTM = 50

###----------------------------------------------------------------------------------------------------------------------###
# Neural Network hyperparameter using Numpy

hidden_size_nnnp = [100, 60, 30, 10]

output_size_nnnp = 3
epochs_nnnp = 100
learning_rate_nnnp = 0.001

###----------------------------------------------------------------------------------------------------------------------###
# Neural Network hyperparameter using Tensorflow

hidden_size_nntf_4 = [64, 32, 16]
hidden_size_nntf_6 = [100, 64, 32, 16, 8]

output_size_nntf = 3
epochs_nntf = 20
learning_rate_nntf = 0.001


device = (
    "cuda"
    if torch.cuda.is_available() 
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")