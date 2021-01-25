########################################################
# Imports
########################################################

import pandas as pd
from torch.utils.data import DataLoader
from utils import *
from torch import cuda
from utils import get_metrics, plot_confusion_matrix

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

dir_results = '../data/models_max/'
dir_external_datasets = '../data/External_Databases/'

########################################################
# Hyperparams - testing mode
########################################################

mode = 1  # 0: Teacher, 1: Student
batch_size = 32
input_shape = (3, 224, 224)
aggregation = 'max'  # 'max', 'attention'
classes = ['NC', 'G3', 'G4', 'G5']
dataset = 'SICAP'  # 'SICAP', 'ARVANITI', 'GERTYCH'

########################################################
# LOAD DATASETS AND MODEL
########################################################

# Get dataframe and images directories
if dataset == 'SICAP':
    df = pd.read_excel(dir_external_datasets + 'gt_sicap.xlsx')
    dir_images = dir_external_datasets + '/sicap_normalized_panda/'
elif dataset == 'ARVANITI':
    df = pd.read_excel(dir_external_datasets + 'gt_arvaniti.xlsx')
    dir_images = dir_external_datasets + '/arvaniti_normalized_panda/'
elif dataset == 'GERTYCH':
    df = pd.read_excel(dir_external_datasets + 'gt_gerytch.xlsx')
    dir_images = dir_external_datasets + '/gerytch_normalized_panda/'

# Load model
model, criterion, optimizer = get_model(mode=0, device=device, learning_rate=0.01, multi_gpu=multi_gpu,
                                        aggregation=aggregation)
if mode == 1:
    model.bb.load_state_dict(torch.load(dir_results + 'student_last_model_bb'))
elif mode == 0:
    model.bb.load_state_dict(torch.load(dir_results + 'teacher_last_model_bb'))

model = model.bb
#model = model.module

# Data generator
test_dataset = Dataset(dir_images=dir_images, data_frame=df, batch_size=batch_size, classes=classes,
                       input_shape=input_shape, data_augmentation=False, mode=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

########################################################
# PREDICT
########################################################

preds, refs = predict(model, test_loader, train_on_gpu, device=device, channel=0)


########################################################
# EVALUATE
########################################################

# Obtain confusion matrix and plot
plot_confusion_matrix(np.argmax(refs, axis=1), np.argmax(preds, axis=1), np.array(classes),
                      dir_out=dir_results + 'cm_' + dataset + '_' + str(mode) + '_' + aggregation)

# Obtain metrics
metrics = get_metrics(np.argmax(refs, axis=1), np.argmax(preds, axis=1))
print(metrics)
metrics.to_excel(dir_results + 'metrics_' + dataset + '_' + str(mode) + '_' + aggregation + '.xlsx')




