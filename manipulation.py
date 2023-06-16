import utils_general
import utils_lightning
import utils_torch
import pytorch_lightning as pl

datasets = utils_torch.creating_datasets(False)
dataloaders = utils_torch.creating_dataloaders(datasets, 107)

arguments = utils_torch.creating_the_kwargs('ribeiro')
model = utils_lightning.creating_the_model(arguments)

trainer = pl.Trainer(max_epochs=100, accelerator='gpu', fast_dev_run=True)
trainer.fit(model, train_dataloaders=dataloaders[0], val_dataloaders=dataloaders[1])

print()