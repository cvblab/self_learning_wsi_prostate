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

dir_images = '../data/PANDA/parches'
dir_dataframe = '../data/PANDA/partition.xlsx'
dir_results = '../data/models_selfAttention/'

########################################################
# Hyperparams - TEACHER MODEL TRAINING
########################################################

classes = ['NC', 'G3', 'G4', 'G5']
batch_size = 1
input_shape = (3, 224, 224)
n_epochs = 30
#learning_rate = 1*1e-3  # Attention
learning_rate = 1*1e-2   # max, self-attention
mode = 0
aggregation = 'self_attention'  # 'max', 'attention', 'self_attention'

########################################################
# MIL Dataset preparation
########################################################

# Prepare data generators
data_frame = pd.read_excel(dir_dataframe)

train_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'train'],
                        batch_size=batch_size, classes=classes, input_shape=input_shape, data_augmentation=True)
train_loader = DataGeneratorSlideLevelLabels(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'val'],
                      batch_size=batch_size, classes=classes, input_shape=input_shape)
val_loader = DataGeneratorSlideLevelLabels(val_dataset, batch_size=batch_size, shuffle=False)


########################################################
# Teacher model training
########################################################

# Load model
model, criterion, optimizer = get_model(mode, device, learning_rate, multi_gpu=multi_gpu, aggregation=aggregation)
model.bb.load_state_dict(torch.load(dir_results + 'teacher_last_model_bb'))

# Initialize trainer object
trainer = CNNTrainer(n_epochs=n_epochs, criterion=criterion, optimizer=optimizer, train_on_gpu=train_on_gpu,
                     device=device, dir_out=dir_results + 'teacher_', save_best_only=True, multi_gpu=False,
                     lr_exp_decay=True,  learning_rate_esch_half=True, slice=True)

# Train
model, history = trainer.train(model, train_loader, val_loader)

# Save learning curve
history.to_excel(dir_results + 'learning_curve_teacher.xlsx')
learning_curve_plot(history, dir_results, 'lc_teacher')

########################################################
# Dataset pseudo-labeling preparation
########################################################

train_dataset = []
val_dataset = []

train_dataset = Dataset(dir_images=dir_images, data_frame=data_frame[data_frame['Partition'] == 'train'],
                        batch_size=batch_size, classes=classes, input_shape=input_shape, data_augmentation=False,
                        mode=2, images_on_ram=False)
train_loader = DataGeneratorSlideLevelLabels(train_dataset, batch_size=batch_size, shuffle=True)
train_loader.N = 128

# Extract patch-level classifier from Teacher model
model = model.bb

# Predict labels of patches from training set using Teacher model
preds, refs = predict(model, train_loader, train_on_gpu, device=device, channel=0)

# get predicted images names to create pseudo-labelled dataset
used_images = train_loader.used_image_names

# Pseudo-Labeling based on teacher predictions and global labels
labels = []
c = 0
for i in used_images:
    print(c)
    preds_i = preds[c, :]
    refs_i = refs[c, 1:]

    if sum(refs_i) == 0:
        labels.append([i, 1, 0, 0, 0])
    else:
        pred_class = np.argmax(preds_i)
        if pred_class == 1 and refs_i[0] == 1:
                labels.append([i, 0, 1, 0, 0])
        elif pred_class == 2 and refs_i[1] == 1:
                labels.append([i, 0, 0, 1, 0])
        elif pred_class == 3 and refs_i[2] == 1:
                labels.append([i, 0, 0, 0, 1])
    c += 1

pseudo_data_frame = pd.DataFrame(labels,
                                 columns=['image_name', 'NC', 'G3', 'G4', 'G5']
                                 )
pseudo_data_frame.to_csv(dir_results + 'pseudo_labels.csv')

pseudo_data_frame = pd.read_csv(dir_results + 'pseudo_labels.csv')
train_dataset = []

# Class weights calculation
c = np.argmax(np.array(pseudo_data_frame[classes].values.tolist()), axis=1)
class_weights = compute_class_weight('balanced', [0, 1, 2, 3], c)

########################################################
# Student model training
########################################################

# Flip mode to patch-level training
mode = 1
batch_size = 64
n_epochs = 20
learning_rate = 1*1e-3

# Data generator
train_dataset = Dataset(dir_images=dir_images, data_frame=pseudo_data_frame,
                        batch_size=batch_size, classes=classes, input_shape=input_shape, data_augmentation=True,
                        mode=mode)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load model
model, criterion, optimizer = get_model(mode, device, learning_rate, multi_gpu=multi_gpu)

# Initialize trainer object
trainer = CNNTrainer(n_epochs=n_epochs, criterion=criterion, optimizer=optimizer, train_on_gpu=train_on_gpu,
                     device=device, dir_out=dir_results + 'student_', save_best_only=False, multi_gpu=multi_gpu,
                     lr_exp_decay=True,  learning_rate_esch_half=True, class_weights=class_weights, channel=0)

# Train
model, history = trainer.train(model, train_loader)

# Save learning curve
history.to_excel(dir_results + 'learning_curve_student.xlsx')
learning_curve_plot(history, dir_results, 'lc_student')

# Save student model
if multi_gpu is False:
    torch.save(model.bb.state_dict(), dir_results + 'Student')
else:
    torch.save(model.bb.module.state_dict(), dir_results + 'Student')