import gc
import mlcm
import torch
import utils_lightning
import utils_general
import utils_torch

'''
This script load, predict some results and evaluate the values.
'''

gc.collect()
torch.cuda.empty_cache()

# Selecting the model that will used
model_name = 'ribeiro'
# model_name = 'rajpurkar'
name_timestamp = f'saved_model:2023-07-27T19:02:00.606094'

# Labels
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# Creating the datasets
datasets = utils_torch.creating_datasets(test=True, only_test=True)

# Selecting the path and the saved model
dirpath = f'saved_models/{name_timestamp}'
filename = f'/{model_name}.ckpt'

# Load the saved model
model = utils_lightning.LitModel.load_from_checkpoint(dirpath+filename)
model.eval()
model.freeze()

# Predicting values based on the data
prediction_bin = utils_torch.computate_predictions(model, datasets[0], 5000)

# Evalueting the predictions
cm, _ = mlcm.cm(datasets[0].labels, prediction_bin, print_note=False)
utils_general.plot_confusion_matrix(cm, model_name, target_names)
print(f"#############################################################################")
mlcm.stats(cm)
utils_general.get_mlcm_report(cm, target_names, model_name)