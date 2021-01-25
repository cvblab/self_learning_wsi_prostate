########################################################
# Imports
########################################################

import pandas as pd
from torch.utils.data import DataLoader
from utils import *
from torch import cuda
from sklearn.utils.class_weight import compute_class_weight

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0")
# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')
multi_gpu = False

random.seed(42)
torch.manual_seed(0)
np.random.seed(0)

########################################################
# Data directories and folders
########################################################

dir_images = '../PANDA/patches'
dir_dataframe = '../data/PANDA/partition.xlsx'
dir_results = '../data/models_selfAttention/'

########################################################
# Hyperparams - TEACHER MODEL TRAINING
########################################################

classes = ['isup_0', 'isup_1', 'isup_2', 'isup_3', 'isup_4', 'isup_5']
# ['gs_0', 'gs_6', 'gs_7', 'gs_8', 'gs_9', 'gs_10'] or ['isup_0', 'isup_1', 'isup_2', 'isup_3', 'isup_4', 'isup_5']
batch_size = 1
input_shape = (3, 224, 224)
n_epochs = 30
learning_rate = 1*1e-2   # max, self-attention
mode = 3
aggregation = 'attention'  # 'average', 'attention'

########################################################
# Dataset preparation
########################################################

data_frame = pd.read_excel(dir_dataframe)

train_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'train'],
                        batch_size=batch_size, classes=classes, input_shape=input_shape, data_augmentation=True)
train_loader = DataGeneratorSlideLevelLabels(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'val'],
                      batch_size=batch_size, classes=classes, input_shape=input_shape)
val_loader = DataGeneratorSlideLevelLabels(val_dataset, batch_size=batch_size, shuffle=False)

########################################################
# Model training
########################################################

# Load model
model, criterion, optimizer = get_model(mode, device, learning_rate, multi_gpu=multi_gpu, aggregation=aggregation,
                                        classes=classes)

# Initialize trainer object
trainer = CNNTrainer(n_epochs=n_epochs, criterion=criterion, optimizer=optimizer, train_on_gpu=train_on_gpu,
                     device=device, dir_out=dir_results + 'scoring_', save_best_only=True, multi_gpu=False,
                     lr_exp_decay=True,  learning_rate_esch_half=True, slice=False)

# Train
model, history = trainer.train(model, train_loader, val_loader)
# Save learning curve
history.to_excel(dir_results + 'learning_curve_scoring.xlsx')
learning_curve_plot(history, dir_results, 'lc_scoring')

########################################################
# Testing
########################################################

train_loader = []
val_loader = []

val_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'test'],
                      batch_size=batch_size, classes=classes, input_shape=input_shape)
val_loader = DataGeneratorSlideLevelLabels(val_dataset, batch_size=batch_size, shuffle=False)