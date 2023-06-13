import utils_general
import utils_lightning
import utils_torch
import pytorch_lightning as pl

datasets = utils_torch.creating_datasets(True)
dataloaders = utils_torch.creating_dataloaders(datasets, 107)

arguments = utils_torch.creating_the_kwargs('ribeiro')
model = utils_lightning.creating_the_model(arguments)

print()