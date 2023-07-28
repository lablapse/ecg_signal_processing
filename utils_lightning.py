import pytorch_lightning as pl
import torch

# Creating the Lightning class to have access to some convenient features
class LitModel(pl.LightningModule):
    
    '''
    inputs in __init__():
        model: the model that will be used;
        optimizer: torch.optim. that will be used;
        **kwargs: key-arguments that will be used to properly create the passed model;
    '''
    
    # Passing values to the __init()__ function
    def __init__(self, model, optimizer, learning_rate, **kwargs):
        
        # Calling the pl.LightningModule 'constructor'
        super(LitModel, self).__init__()
        
        # Saving hiperparameters
        self.save_hyperparameters()
        
        # Internalizing inputed the model
        self.model = model(**kwargs)
        
        # Selecting the loss function
        self.metric = torch.nn.BCELoss()
        
        self.optimizer_function = optimizer
        self.learning_rate = learning_rate

    # Creating the 'forward()' function
    def forward(self, input):
        # this calls the 'forward()' function of the selected model
        return self.model.forward(input)

    # This selects the optimizer and the learning rate of the selected model
    def configure_optimizers(self):
        self.optimizer = self.optimizer_function(self.parameters(), lr=self.learning_rate)
        self.schedular = {'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
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
    and returns the selected model.
    '''
    
    model = LitModel(model_kwargs[0], model_kwargs[2], model_kwargs[3], **model_kwargs[1])
    return model