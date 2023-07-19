from datetime import datetime
import graphviz
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callback
import torchview #https://github.com/mert-kurttutan/torchview
import utils_lightning
import utils_torch

# model_name = 'ribeiro'
model_name = 'rajpurkar'
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# Defining some variables to saving paths
timestamp = datetime.now().isoformat()
dirpath = f'saved_models/saved_model:{timestamp}'
filename = f'{model_name}'

# Creating the datasets
datasets = utils_torch.creating_datasets(True)
dataloaders = utils_torch.creating_dataloaders(datasets, 107)

arguments = utils_torch.creating_the_kwargs(model_name)
model = utils_lightning.creating_the_model(arguments)

# Saving a pdf with the image of the model
# model_graph = torchview.draw_graph(model, input_size=(1, datasets[0].data.shape[1], datasets[0].data.shape[2]),
                                #    graph_name=f'{model_name}', hide_module_functions=False, depth=100)
# model_graph.visual_graph
# model_graph.visual_graph.save()
# dot = graphviz.Source.from_file(f'{model_name}.gv')
# dot.render()
# del dot

# Defining callbacks
# Checkpointing the model
checkpoint_callback = callback.ModelCheckpoint(dirpath=dirpath, filename=filename, 
                                               monitor='train_loss', save_top_k=1, mode='min')
# Receiving information from the model
rich_callback = callback.RichModelSummary(max_depth=3)

# Early stopping callback
early_stopping_callback = callback.EarlyStopping(monitor="val_loss", mode="min", patience=10)

# # Creating the CSV logger
# logger = callback.CSVLogger("pasta_logger", name=f'losses')

# Accumulating callbacks in a list()
callbacks = [checkpoint_callback, rich_callback, early_stopping_callback]

# Defining the trainer from pytorch lightning
trainer = pl.Trainer(max_epochs=100, accelerator='gpu', callbacks=callbacks, fast_dev_run=False, devices='auto')

# Fitting the model
trainer.fit(model, train_dataloaders=dataloaders[0], val_dataloaders=dataloaders[1])