import numpy as np # some fundamental operations
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import utils_general

# Creating the kernel initializer for nn.Conv1d and nn.Linear 
def _weights_init(m):
    '''
        This function will be called using self.apply(_weights_init) in some class later.
        Don't worry about 'm'.
    '''
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)

# Creating the residual block to the Rajpurkar architecture
class residual_blocks_rajpurkar_torch(nn.Module):
    
    '''
    This class creates the residual block for the Rajpurkar architecture
    
    inputs in __init__:

    '''
    # Passing values to the object
    def __init__(self, i=0, stride=1, in_channels=64, out_channels=64, rate_drop=0.5):
        
        # Calling the nn.Module 'constructor' 
        super(residual_blocks_rajpurkar_torch, self).__init__()
    
        # Internalizing the input values
        self.i = i
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rate_drop = rate_drop
        
        # Creating the layer
        # Some default 'torch.nn' values were modified to match the default ones presented in 'keras' 
        self.layer = nn.Sequential(nn.BatchNorm1d(num_features=self.in_channels, eps=0.001, momentum=0.99),
                                   nn.ReLU(),
                                   nn.Dropout(p=self.rate_drop), # VER SE OS DROPOUTS ESTAO IGUAIS CONCEITUALMENTE -> torch: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html keras -> https://keras.io/api/layers/regularization_layers/dropout/
                                   nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=16,
                                             stride=1, padding='same'),
                                   nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                   nn.ReLU(),
                                   nn.Dropout(p=self.rate_drop), # VER SE OS DROPOUTS ESTAO IGUAIS CONCEITUALMENTE -> torch: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html keras -> https://keras.io/api/layers/regularization_layers/dropout/
                                   nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=16,
                                             stride=stride, padding=7)
        )
        
        # Creating the short connection
        if self.i == 3 or self.i == 7 or self.i == 11:
            self.skip = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=16,
                                                stride=1, padding='same'),
                                                nn.MaxPool1d(kernel_size=1, stride=self.stride)
            )
        
        else:
            self.skip = nn.MaxPool1d(kernel_size=1, stride=self.stride)

    # The operation function: the one that will calculate what is needed
    def forward(self, input):
        out = self.layer(input)
        short = self.skip(input)
        if out.shape[2] != short.shape[2]:
            out = nn.functional.pad(out, (0,1))
        out = out + short
        return out
    
class skip_connection_torch(nn.Module):
    
    '''
    inputs in __init__():

    '''
    
    # Passing values to the object
    def __init__(self, in_channels=128, out_channels=128, downsample=1):
        
        # Calling the nn.Module 'constructor' 
        super(skip_connection_torch, self).__init__()
    
        # Internalizing the input values
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
    
        # Creating the short connection
        self.skip = nn.Sequential(nn.MaxPool1d(kernel_size=self.downsample, stride=self.downsample),
                                  nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                            kernel_size=1, stride=1, padding='same')
        )
    
    # The operation function: the one that will calculate what is needed
    def forward(self, input):
        out = self.skip(input)
        return out
    
    

class residual_blocks_ribeiro_torch(nn.Module):
    
    '''

    '''
    
    # Passing values to the object
    def __init__(self, skip_connection, in_channels=128, out_channels=128, rate_drop=0, downsample=1):
        
        # Calling the nn.Module 'constructor' 
        super(residual_blocks_ribeiro_torch, self).__init__()
    
        # Internalizing the input values
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rate_drop = rate_drop
        self.downsample = downsample
        self.skip_connection = skip_connection

        # This object will be summed with the skip_connection object at the 'forward()' method        
        self.layer_sum = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                                 kernel_size=16, stride=1, padding='same'),
                                       nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.rate_drop),
                                       nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, 
                                                 kernel_size=16, stride=self.downsample, padding='same')            
        )
        
        self.layer_middle = self.skip_connection(self.in_channels, self.out_channels, self.downsample)
        
        self.layer_alone = nn.Sequential(nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                         nn.ReLU(),
                                         nn.Dropout(p=self.rate_drop)
        )
        
    def forward(self, input):
        layer, skip = input
        skip = self.layer_middle(skip)
        layer = self.layer_sum(layer)
        layer = layer + skip
        skip = layer
        layer = self.layer_alone(layer)
        return layer, skip
        
class rajpurkar_torch(nn.Module):
    
    '''
    This class implements the Rajpurkar model like-wise was made by Sarah
    '''
    
    # Passing values to the object
    def __init__(self, residual_blocks_rajpurkar_torch, rate_drop, in_channels):
        
        # Calling the nn.Module 'constructor' 
        super(rajpurkar_torch, self).__init__()
        
        #Internalizing the values
        self.rate_drop = rate_drop
        self.in_channels = in_channels
        self.residual_blocks_rajpurkar_torch = residual_blocks_rajpurkar_torch
        
        
        # Creating a 'dict' with values that will be used multiple times in nn.Conv1d() function
        self.conv_config = dict(in_channels=64, 
                                out_channels=64, 
                                kernel_size=16
        )
        
        
        # Creating the first block
        self.layer_to_be_passed = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                                          out_channels=64,
                                                          kernel_size=16, 
                                                          stride=1, 
                                                          padding='same'),
                                                nn.BatchNorm1d(num_features=64, 
                                                               eps=0.001, momentum=0.99),
                                                nn.ReLU()
        )
        
        # Short  connection
        self.skip_alone = nn.MaxPool1d(kernel_size=1, stride=2)
        
        # Creating the second block
        self.layer_to_be_summed = nn.Sequential(nn.Conv1d(stride=1, padding='same', **self.conv_config),
                                                nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.99),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.rate_drop),
                                                nn.Conv1d(stride=2, **self.conv_config, padding=7)            
        )
        
        # Creating the channels matrix that will be used to create the connection layers
        self.num_channels = [64, 64, 64, 64,
                             128, 128, 128, 128,
                             192, 192, 192, 192,
                             256, 256, 256, 256
        ]
        
        # Creating the list that will append the middle layers
        middle_layers_list = nn.ModuleList()
        
        # Appending the middle layers to the middle_layer_list
        for i in range(15):
            middle_layers_list.append(self.residual_blocks_rajpurkar_torch(i=i, stride=(i%2)+1, 
                                                                           in_channels=self.num_channels[i], 
                                                                           out_channels=self.num_channels[i+1], 
                                                                           rate_drop=self.rate_drop))
        # Creating the 'nn.Sequential()' using the 'middle_layer_list'
        self.middle_layers = nn.Sequential(*middle_layers_list)
        
        # End layer
        self.layer_end = nn.Sequential(nn.BatchNorm1d(num_features=self.num_channels[-1], eps=0.001, momentum=0.99),
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       nn.Linear(in_features=1024, out_features=32),
                                       nn.Linear(in_features=32, out_features=5),
                                       nn.Sigmoid()
        )
    
        # This applies to this module and all children of it. This line below initializes the 
        # 'nn.Conv1d()', 'nn.Linear()' weights.
        self.apply(_weights_init)
    
    # Calculating "Rajpurkar's" model
    def forward(self, input):
        layer = self.layer_to_be_passed(input)
        skip = self.skip_alone(layer)
        layer = self.layer_to_be_summed(layer)
        out = layer + skip
        out = self.middle_layers(out)
        out = self.layer_end(out)
        return out
    
class ribeiro_torch(nn.Module):
    
    '''
    This class implements the Ribeiro model like-wise was made by Sarah
    '''
    
    # Passing values to the object
    def __init__(self, residual_blocks_ribeiro_torch, skip_connection_torch, rate_drop, in_channels, downsample):
        
        # Calling the nn.Module 'constructor' 
        super(ribeiro_torch, self).__init__()
        
        #Internalizing the values
        self.rate_drop = rate_drop
        self.in_channels = in_channels
        self.downsample = downsample
        self.residual_blocks_ribeiro_torch = residual_blocks_ribeiro_torch
        self.skip_connection_torch = skip_connection_torch
    
        # Input block
        self.layer_initial = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=64, 
                                                     kernel_size=16, stride=1, padding='same'),
                                           nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.99),
                                           nn.ReLU()
        )
    
        # The channels dimensions
        self.num_channels = np.array([64, 128, 192, 256, 320])
        
        # Crating the list that will receive the middle blocks
        middle_layers_list = nn.ModuleList()
        
        # Appending to 'middle_layers_list'
        for i in range(4):
            middle_layers_list.append(self.residual_blocks_ribeiro_torch(self.skip_connection_torch, self.num_channels[i], 
                                                                         self.num_channels[i+1], self.rate_drop, self.downsample))
        
        # Creating the middle layers
        self.layers_middle = nn.Sequential(*middle_layers_list)

        # Output block
        self.layer_end = nn.Sequential(nn.Flatten(),
                                       nn.Linear(320000, 5),
                                       nn.Sigmoid())
            
        # This applies to this module and all children of it. This line below initializes the 
        # 'nn.Conv1d()', 'nn.Linear()' weights.
        self.apply(_weights_init)

    # Calculating "Ribeiro's" model
    def forward(self, input):
        input = self.layer_initial(input)
        input = self.layers_middle((input, input))
        out = self.layer_end(input[0])
        return out
    
# Creating a CustomDataset class to be used
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        # Converting to float because of BCELoss from Pytorch
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx, :, :]
        current_label = self.labels[idx, :]
        return current_sample, current_label
    
# Function that will load the data and return a custom dataloader
def creating_datasets(test=False, only_test=False):
    
    '''
    
    '''
    
    if only_test:
        test = True
    
    # Loading the data
    info = utils_general.load_data(test)
    
    # List that will receive the datasets
    datasets = list()
    
    if only_test:
        datasets.append(CustomDataset(info[4], info[5]))
        return datasets
    
    # Appending the train dataset
    datasets.append(CustomDataset(info[0], info[1]))
    
    # Appending the validation dataset
    datasets.append(CustomDataset(info[2], info[3]))
    
    if test:
        # Appending the test dataset
        datasets.append(CustomDataset(info[4], info[5]))
        
    return datasets
    
# Creating the Dataloaders using torch
def creating_dataloaders(datasets, batch_size):
    
    '''
    '''
    
    if len(datasets) == 1:
        raise ValueError('Datasets must contain at least training and validation to create the DataLoaders.')
    
    dataloaders = list()
    
    # Creating the train DataLoader
    dataloaders.append(DataLoader(datasets[0], batch_size=batch_size))
    
    # Creating the validation DataLoader
    dataloaders.append(DataLoader(datasets[1], batch_size=batch_size))
    
    if len(datasets) == 3:
        # Creating the test DataLoader
        dataloaders.append(DataLoader(datasets[1], batch_size=batch_size))
        
    return dataloaders
        

def creating_the_kwargs(model_name):
    
    '''
    This fuction receives a string with the desidred model name and returns
    the torch chass model with a dictionary of whats will later be **kwargs
    to other function or class.
    '''

    # Selecting the model that will be called
    if model_name == 'rajpurkar':
        # Selecting "Rajpurkar's" model
        model = rajpurkar_torch
        arguments = dict(residual_blocks_rajpurkar_torch=residual_blocks_rajpurkar_torch,
                         rate_drop=0.5,
                         in_channels=12
        )
        
    elif model_name == 'ribeiro':    
        # Selecting "Ribeiro's" model
        model = ribeiro_torch
        arguments = dict(residual_blocks_ribeiro_torch=residual_blocks_ribeiro_torch,
                         skip_connection_torch=skip_connection_torch,
                         rate_drop=0.5,
                         in_channels=12,
                         downsample=1
        )
        
    else:
        # Raising an error if an invalid string was passed
        raise NameError(" Wrong Name. Allowed names are 'rajpurkar' and 'ribeiro'. ") 
    
    return model, arguments