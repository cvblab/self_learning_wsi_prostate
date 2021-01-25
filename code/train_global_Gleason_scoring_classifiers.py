########################################################
# Imports
########################################################

import pandas as pd
from torch.utils.data import DataLoader
from utils import *
from torch import cuda
from sklearn.utils.class_weight import compute_class_weight
from miGraph import *
from sklearn import svm

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
dir_results = '../data/models_max/'

########################################################
# Hyperparams - TEACHER MODEL TRAINING
########################################################

classes = ['isup_0', 'isup_1', 'isup_2', 'isup_3', 'isup_4', 'isup_5']
# ['gs_0', 'gs_6', 'gs_7', 'gs_8', 'gs_9', 'gs_10'] or ['isup_0', 'isup_1', 'isup_2', 'isup_3', 'isup_4', 'isup_5']
batch_size = 1
input_shape = (3, 224, 224)
n_epochs = 30
mode = 3
dist_method = 'gaussian'
C = 10

########################################################
# Dataset preparation
########################################################

data_frame = pd.read_excel(dir_dataframe)
data_frame = data_frame[0:200]

train_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'train'],
                        batch_size=batch_size, classes=classes, input_shape=input_shape, data_augmentation=False,
                        mode=2, images_on_ram=False)
train_loader = DataGeneratorSlideLevelLabels(train_dataset, batch_size=batch_size, shuffle=False)

val_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'test'],
                      batch_size=batch_size, classes=classes, input_shape=input_shape, data_augmentation=False,
                      mode=2, images_on_ram=False)
val_loader = DataGeneratorSlideLevelLabels(val_dataset, batch_size=batch_size, shuffle=False)

########################################################
# Features prediction
########################################################

# Load model
model, criterion, optimizer = get_model(mode, device, 0.1, multi_gpu=multi_gpu, aggregation='max')
model.bb.load_state_dict(torch.load(dir_results + 'student_last_model_bb'))
# Extract patch-level classifier from Student model
model = model.bb

########################################################
# Graph training and testing
########################################################


feats_train, refs_train = predict(model, train_loader, train_on_gpu, device=device, channel=1, bag_organization=True)
refs_categorical_train = [np.argmax(iRef[0, :]) for iRef in refs_train]

# building up kernels
print('GNN Training')
miKernelTrain = buildKernel_threading(feats_train, feats_train, gamma=1, delta=None, delta_method='global',
                                      dist_method='gaussian')
feats_train = []

# train on SVM
print('SVM Training')
modelSvm = svm.SVC(kernel='precomputed', C=C, probability=True)
modelSvm.fit(miKernelTrain, refs_categorical_train)

miKernelTrain = []

feats_test, refs_test = predict(model, val_loader, train_on_gpu, device=device, channel=1, bag_organization=True)
refs_categorical_test = [np.argmax(iRef[0, :]) for iRef in refs_test]

print('GNN Training')
miKernelTest = buildKernel_threading(feats_test, feats_test, gamma=1, delta=None, delta_method='global',
                                     dist_method='gaussian')
feats_test = []

yPredTest = modelSvm.predict(miKernelTest)
yProbTest = modelSvm.predict_proba(miKernelTest)

# Obtain confusion matrix and plot
plot_confusion_matrix(refs_categorical_test, yPredTest, np.array(classes),
                      dir_out=dir_results + 'cm_miGraph')

