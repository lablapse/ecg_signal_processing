import pytorch_lightning as pl
import torch

'''
This script uses a library called 'pytorch lightning' -> https://lightning.ai/docs/pytorch/stable/ 
This library is used because Pytorch doesn't have some functions that Tensorflow/keras have. 
'''

# Creating the Lightning class to have access to some convenient features from pytorch lightning
class LitModel(pl.LightningModule):
    
    '''
    inputs in __init__():
        model: the model that will be used;
        optimizer: torch.optim. that will be used;
        **kwargs: key-arguments that will be used to properly create the passed model;
    '''
    
    def __init__(self, model, optimizer, learning_rate, optim_kwargs, **kwargs):
        
        super(LitModel, self).__init__()
        
        # Saving hiperparameters
        self.save_hyperparameters()
        
        self.model = model(**kwargs)
        
        # Selecting the loss function and other hiperparameters
        self.metric = torch.nn.BCELoss(reduction='mean')
        self.optimizer_function = optimizer
        self.optim_kwargs = optim_kwargs
        self.learning_rate = learning_rate

    def forward(self, input):
        return self.model.forward(input)

    def configure_optimizers(self):
        self.optimizer = self.optimizer_function(self.parameters(), lr=self.learning_rate, **self.optim_kwargs)
        self.schedular = {'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True,
            eps=0
        ), 'monitor':'val_loss'}
        return {'optimizer': self.optimizer, 'lr_scheduler': self.schedular}

    # Creating the training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.metric(y_hat, y)
        self.log("train_loss", loss)
        return loss

    # Creating the validation step
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.metric(y_hat, y)
        self.log("val_loss", val_loss)

    # Creating the test step
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        test_loss = self.metric(y_hat, y)
        self.log("test_loss", test_loss)

    # Creating the predictiong step
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        pred = self(x)
        return pred

def creating_the_model(model_kwargs):
    
    '''
    This function receives its inputs from the returned values of 
    "utils_torch.py -> 'creating_the_kwargs'" function
    and returns the selected model. It also selects the values for 
    the selected optimizer to match the default ones used in keras.
    '''
    
    if torch.optim.Adam == model_kwargs[2]:
        optim_kwargs = {"eps":1e-7}
        
    if torch.optim.RMSprop == model_kwargs[2]:
        optim_kwargs = {"alpha":0.9, "eps":1e-7}
        
    if torch.optim.SGD == model_kwargs[2]:
        optim_kwargs = {"momentum":0} # Not really necessary to put this one here like this, I am just maintaining a pattern
        
    # the arguments are -> model, optimizer, learning_rate, optim_kwargs, **kwargs. 
    model = LitModel(model_kwargs[0], model_kwargs[2], model_kwargs[3], optim_kwargs, **model_kwargs[1])
    return model