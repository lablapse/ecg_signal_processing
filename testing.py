import gc
import matplotlib.pyplot as plt
import mlcm
import numpy as np
import torch
import tqdm
import utils_lightning
import utils_general
import utils_torch

gc.collect()
torch.cuda.empty_cache()

# model_name = 'ribeiro'
model_name = 'rajpurkar'
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']
name_timestamp = f'saved_model:2023-07-19T15:45:54.712826'

# data = np.load("data.npz")
# X_test = data[f"X_test"]
# y_test = data[f"y_test"]

datasets = utils_torch.creating_datasets(test=True, only_test=True)
# dataloaders = utils_torch.creating_dataloaders(datasets, 107)

dirpath = f'saved_models/{name_timestamp}'
# dirpath = f'saved_models/saved_model:2022-12-23T13:17:10.357274'
filename = f'/{model_name}.ckpt'


model = utils_lightning.LitModel.load_from_checkpoint(dirpath+filename)
model.eval()

# prediction = model(torch.tensor(datasets[0].data).to('cuda'))
# prediction = prediction.to('cpu')
# prediction = prediction.detach().numpy()
# prediction_bin = (prediction > 0.5).astype(np.float32)

# del prediction

# Creating the list that will receive the predictions
prediction_bin = []

# Iterating trought the dataset. I was having GPU memory issues, so I made the code this way.
# The must have a proper way to do this.
for i in tqdm.tqdm(range(datasets[0].labels.shape[0])):
    # Passing data to the loaded model and collecting the result
    prediction = model(torch.tensor(datasets[0].data[i:(i+1)]).cuda())
    
    # Moving the data from cuda to cpu because I was having some issues with numpy
    prediction = prediction.to('cpu')
    
    # Converting to numpy array
    prediction = prediction.detach().numpy()
    
    # Applying a threshold to the data and appending to prediction_bin
    prediction_bin.append((prediction > 0.5).astype(np.float32))
    
    # This two lines below are here to try to prevent some gpu-related memory issues
    gc.collect()
    torch.cuda.empty_cache()

    # deleting prediction. Don't know if it's necessary.
    del prediction
    
# Converting prediction_bin from a list to a numpy array 
prediction_bin = np.array(prediction_bin)
# Unsqueezing dimension [1] so prediction_bin is of shape (datasets[0].labels.shape[0], 5) 
# instead of (datasets[0].labels.shape[0], 1, 5) 
prediction_bin = prediction_bin.squeeze(1)

# Save results
cm, _ = mlcm.cm(datasets[0].labels, prediction_bin, print_note=False)
utils_general.plot_confusion_matrix(cm, model_name, target_names)
print(f"#############################################################################")
mlcm.stats(cm)
utils_general.get_mlcm_report(cm, target_names, model_name)

print()