import gc
import mlcm
import numpy as np
import torch
import tqdm
import utils_lightning
import utils_general
import utils_torch

gc.collect()
torch.cuda.empty_cache()

# Selecting the model that will used
model_name = 'ribeiro'
# model_name = 'rajpurkar'
name_timestamp = f'saved_model:2023-07-27T19:02:00.606094'

# Labels
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# Creating the datasets
# datasets = utils_torch.creating_datasets(test=True, only_test=True)
datasets = utils_torch.creating_datasets(test=True)
# dataloaders = utils_torch.creating_dataloaders(datasets, 107)

dirpath = f'saved_models/{name_timestamp}'
filename = f'/{model_name}.ckpt'


model = utils_lightning.LitModel.load_from_checkpoint(dirpath+filename)
model.eval()
model.freeze()

prediction_bin = utils_torch.computate_predictions(model, datasets[0], 5000)

# Converting prediction_bin from a list to a numpy array 
# prediction_bin = np.array(prediction_bin)
# Unsqueezing dimension [1] so prediction_bin is of shape (datasets[0].labels.shape[0], 5) 
# instead of (datasets[0].labels.shape[0], 1, 5) 
# prediction_bin = prediction_bin.squeeze(1)

# Save results
cm, _ = mlcm.cm(datasets[0].labels, prediction_bin, print_note=False)
utils_general.plot_confusion_matrix(cm, model_name, target_names)
print(f"#############################################################################")
mlcm.stats(cm)
utils_general.get_mlcm_report(cm, target_names, model_name)

print()