import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'
phoneme_data1 = 'data\GMM_params_phoneme_01_k_03.npy'
phoneme_data12 = 'data\GMM_params_phoneme_01_k_06.npy'
phoneme_data2 = 'data\GMM_params_phoneme_02_k_03.npy'
phoneme_data21 = 'data\GMM_params_phoneme_02_k_06.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)
phoneme_data1=np.load(phoneme_data1, allow_pickle=True)
phoneme_data1=np.ndarray.tolist(phoneme_data1)
phoneme_data12=np.load(phoneme_data12, allow_pickle=True)
phoneme_data12=np.ndarray.tolist(phoneme_data12)
phoneme_data2=np.load(phoneme_data2, allow_pickle=True)
phoneme_data2=np.ndarray.tolist(phoneme_data2)
phoneme_data21=np.load(phoneme_data21, allow_pickle=True)
phoneme_data21=np.ndarray.tolist(phoneme_data21)
# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for i in range(len(f1)):
    X_full[i,0]=f1[i]
    X_full[i,1]=f2[i]
mean1=phoneme_data1['mu']
size1=phoneme_data1['p']
covariance1=phoneme_data1['s']

mean1=phoneme_data1['mu']
size1=phoneme_data1['p']
covariance1=phoneme_data1['s']

mean12=phoneme_data12['mu']
size12=phoneme_data12['p']
covariance12=phoneme_data12['s']

mean2=phoneme_data2['mu']
size2=phoneme_data2['p']
covariance2=phoneme_data2['s']

mean21=phoneme_data21['mu']
size21=phoneme_data21['p']
covariance21=phoneme_data21['s']
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
id=[]
X_phonemes_1_2 = np.zeros((np.sum((phoneme_id==1) | (phoneme_id==2)), 2))
check=np.where((phoneme_id==1) | (phoneme_id==2))
X_phonemes_1_2[:,:]=X_full[check,:]
print(X_phonemes_1_2.shape)
for i in check:
    id.append(phoneme_id[i])
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
predict1=get_predictions(mean1,covariance1,size1,X_phonemes_1_2)
predict2=get_predictions(mean12,covariance12,size12,X_phonemes_1_2)
predict3=get_predictions(mean2,covariance2,size2,X_phonemes_1_2)
predict4=get_predictions(mean21,covariance21,size21,X_phonemes_1_2)

final1=[]
final2=[]

for i in range(len(predict1)):
    if np.max(predict1[i]>np.max(predict3[i])):
        final1.append(1)
    else:
        final1.append(2)
    if np.max(predict2[i]>np.max(predict4[i])):
        final2.append(1)
    else:
        final2.append(2)

print(final1)
count1=0
count2=0
for i in range(len(predict1)):
    if id[0][i]==final1[i]:
        count1=count1+1
    if id[0][i] == final2[i]:
        count2 = count2 + 1
print(id)
acc1=count1/304*100
acc2=count2/304*100
accuracy=(acc1+acc2)/2
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()