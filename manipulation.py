from datetime import datetime
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callback
import torch
import utils_lightning
import utils_torch

'''
This script trains one model and saves the weights in 'saved_models'.
'''

# Choosing one of the models to train, Ribeiro or Rajpurkar
model_name = 'ribeiro'
# model_name = 'rajpurkar'

# Defining some variables to saving paths.
timestamp = datetime.now().isoformat()
dirpath = f'saved_models/saved_model:{timestamp}'
filename = f'{model_name}'

# Creating the datasets
batch_size = 64
datasets = utils_torch.creating_datasets(True)
dataloaders = utils_torch.creating_dataloaders(datasets, batch_size)

# Creating the model
arguments = utils_torch.creating_the_kwargs(model_name, torch.optim.Adam, learning_rate=0.001)
model = utils_lightning.creating_the_model(arguments)

# Defining callbacks
# Checkpointing the model
checkpoint_callback = callback.ModelCheckpoint(dirpath=dirpath, filename=filename, 
                                               monitor='train_loss', save_top_k=1, mode='min')
# Receiving information from the model
rich_callback = callback.RichModelSummary(max_depth=3)

# Early stopping callback
early_stopping_callback = callback.EarlyStopping(monitor="val_loss", mode="min", patience=10)

# Accumulating callbacks in a list()
callbacks = [checkpoint_callback, rich_callback, early_stopping_callback]

# Defining the trainer from pytorch lightning
trainer = pl.Trainer(max_epochs=100, accelerator='gpu', callbacks=callbacks, fast_dev_run=False, devices='auto')

# Fitting the model
trainer.fit(model, train_dataloaders=dataloaders[0], val_dataloaders=dataloaders[1])