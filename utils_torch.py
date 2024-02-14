import gc
import numpy as np # some fundamental operations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import utils_general

''' 
    This script compiles a lot of functions used in the main script - grid_search_torch.py -.
    The name - utils_torch.py -, is granted because it does have all the Pytorch code
    used in the formulation of the model. This script is seized from the - utils_general.py -
    script for debugging reasons.
'''

# Creating the kernel initializer for nn.Conv1d and nn.Linear 
def _weights_init(m):
    '''
        This function will be called using self.apply(_weights_init) in some class later.
        Don't worry about 'm'.
        This function sets the weights for Linear and Convolutional layers.
    '''
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight) # 'He normal' initialization

# Creating the residual block to the Rajpurkar architecture
class residual_blocks_rajpurkar_torch(nn.Module):
    
    '''
    This class creates the residual block for the Rajpurkar architecture.
    
    inputs in __init__:
        All the inputs are passed by other functions/classes
    
        i: int;
        stride: int;
        in_channels: int;
        out_channels: int;
        rate_drop: float;
    '''

    def __init__(self, i=0, stride=1, in_channels=64, out_channels=64, rate_drop=0.5):
        
        super(residual_blocks_rajpurkar_torch, self).__init__()
    
        self.i = i
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rate_drop = rate_drop
        
        # Creating the layer
        # Some default 'torch.nn' values were modified to match the default ones presented in 'keras' 
        self.layer = nn.Sequential(nn.BatchNorm1d(num_features=self.in_channels, eps=0.001, momentum=0.99),
                                   nn.ReLU(),
                                   nn.Dropout(p=self.rate_drop), 
                                   nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=16,
                                             stride=1, padding='same'),
                                   nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                   nn.ReLU(),
                                   nn.Dropout(p=self.rate_drop) 
        )
        
        self.conv_after_layer = nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=16,
                                             stride=stride)
        
        # Creating the short connection
        if self.i == 3 or self.i == 7 or self.i == 11:
            self.skip = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=16,
                                                stride=1, padding='same'),
                                                nn.MaxPool1d(kernel_size=1, stride=self.stride)
            )
        
        else:
            self.skip = nn.MaxPool1d(kernel_size=1, stride=self.stride)

    def forward(self, input):
        out = self.layer(input)
        out = F.pad(out, (8, 7))
        out = self.conv_after_layer(out)
        short = self.skip(input)
        out = out + short
        return out
    
class skip_connection_torch(nn.Module):
    
    '''
    inputs in __init__():
        All the inputs are passed by other functions/classes.
        
        in_channels: int;
        out_channels: int;
        downsample: int;
    '''
    
    def __init__(self, in_channels=128, out_channels=128, downsample=1):
        
        super(skip_connection_torch, self).__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
    
        # Creating the short connection
        self.skip = nn.Sequential(nn.MaxPool1d(kernel_size=self.downsample, stride=self.downsample, ceil_mode=True),
                                  nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                            kernel_size=1, stride=1, padding='same')
        )
    
    def forward(self, input):
        out = self.skip(input)
        return out

class residual_blocks_ribeiro_torch(nn.Module):
    
    '''
    inputs in __init__():
        All the inputs are passed by other functions/classes.

        skip_connection: 
        in_channels: int;
        out_channels: int;
        rate_drop: float;
        downsample: int;
    '''
    
    def __init__(self, skip_connection, in_channels=128, out_channels=128, rate_drop=0, downsample=1):
        
        super(residual_blocks_ribeiro_torch, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rate_drop = rate_drop
        self.downsample = downsample
        self.skip_connection = skip_connection

        # This object will be summed with the skip_connection object in the 'forward()' method
        self.layer_sum = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                                 kernel_size=16, stride=1, padding='same'),
                                       nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                       nn.ReLU(),
                                       nn.Dropout(p=self.rate_drop)     
        )
        
        self.conv_after_layer_sum = nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, 
                                                 kernel_size=16, stride=self.downsample)     
        
        self.layer_middle = self.skip_connection(self.in_channels, self.out_channels, self.downsample)
        
        self.layer_alone = nn.Sequential(nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                         nn.ReLU(),
                                         nn.Dropout(p=self.rate_drop)
        )
        
    def forward(self, input):
        layer, skip = input
        skip = self.layer_middle(skip)
        layer = self.layer_sum(layer)
        layer = F.pad(layer, (8, 7))
        layer = self.conv_after_layer_sum(layer)
        layer = layer + skip
        skip = layer
        layer = self.layer_alone(layer)
        return layer, skip
        
class rajpurkar_torch(nn.Module):
    
    '''
    This class implements the Rajpurkar model like-wise was made by Sarah
    inputs in __init__():
        residual_blocks_rajpurkar_torch:
        rate_drop: float;
        in_channels: int; 
    '''
    
    def __init__(self, residual_blocks_rajpurkar_torch, rate_drop, in_channels):
        
        super(rajpurkar_torch, self).__init__()
        
        self.rate_drop = rate_drop
        self.in_channels = in_channels
        self.residual_blocks_rajpurkar_torch = residual_blocks_rajpurkar_torch
        
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
                                                nn.Dropout(p=self.rate_drop)            
        )
        
        self.conv_after_layer_to_be_summed = nn.Conv1d(stride=2, **self.conv_config)
        
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
    
    def forward(self, input):
        layer = self.layer_to_be_passed(input)
        skip = self.skip_alone(layer)
        layer = self.layer_to_be_summed(layer)
        layer = F.pad(layer, (8, 7))
        layer = self.conv_after_layer_to_be_summed(layer)
        out = layer + skip
        out = self.middle_layers(out)
        out = self.layer_end(out)
        return out
    
class ribeiro_torch(nn.Module):
    
    '''
    This class implements the Ribeiro model like-wise was made by Sarah
    inputs in __init__():
        residual_blocks_ribeiro_torch:
        skip_connection_torch:
        rate_drop: float;
        in_channels: int;
        downsample: int;
    '''
    
    def __init__(self, residual_blocks_ribeiro_torch, skip_connection_torch, rate_drop, in_channels, downsample):
        
        super(ribeiro_torch, self).__init__()
        
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
    
        # The channels list
        self.num_channels = np.array([64, 128, 192, 256, 320])
        
        # Crating the list that will receive the middle blocks
        middle_layers_list = nn.ModuleList()
        
        for i in range(4):
            middle_layers_list.append(self.residual_blocks_ribeiro_torch(self.skip_connection_torch, self.num_channels[i], 
                                                                         self.num_channels[i+1], self.rate_drop, self.downsample))
        
        # Creating the middle layers
        self.layers_middle = nn.Sequential(*middle_layers_list)

        # Output block
        self.layer_end = nn.Sequential(nn.Flatten(),
                                       nn.Linear(1280, 5),
                                       nn.Sigmoid()
                                       )
            
        # This applies to this module and all children of it. This line below initializes the 
        # 'nn.Conv1d()', 'nn.Linear()' weights.
        self.apply(_weights_init)

    def forward(self, input):
        input = self.layer_initial(input)
        input = self.layers_middle((input, input))
        out = self.layer_end(input[0])
        return out
    
# Creating a CustomDataset class to be used.
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx, :, :]
        current_label = self.labels[idx, :]
        return current_sample, current_label
    
# Function that will load the data and return a custom dataset
def creating_datasets(test=False, only_test=False):
    
    '''
    This function calls the function load_data from utils_general.py and creates
    custom datasets.
    
    inputs:
        test: bool -> True if the test dataset is desired, False if it's not;
        only_test: bool -> True if just the test dataset is desired;
 
    return:
        returns a list of the train, val and/or test of the custom datasets;

    If 'only_test' is True, the returned list have length of 2, otherwise, 
    the length is 6.
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
    This function creates DataLoaders to be used with Pytorch.
    
    inputs:
        datasets: preferable the list returned from 'creating_datasets'. A list of CustomDataset.
        batch_size: int -> the batch size used;
        
    return:
        dataloaders: a list of the created DataLoaders.
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
        dataloaders.append(DataLoader(datasets[2], batch_size=batch_size))
        
    return dataloaders
        

def creating_the_kwargs(model_name, optimizer, learning_rate):
    
    '''
    This fuction receives a string with the desidred model name, a optimizer funtion from 'torch.optim', 
    a desired learning rate and returns the torch class model with a dictionary of what will 
    later be **kwargs to other function or class, and the same optimizer function and learning rate passed
    as a argument to this function. It's made this way because it's easier to debug and 
    create the models.
    '''


    if model_name == 'rajpurkar':
        model = rajpurkar_torch
        arguments = dict(residual_blocks_rajpurkar_torch=residual_blocks_rajpurkar_torch,
                         rate_drop=0.2,
                         in_channels=12
        )
        
    elif model_name == 'ribeiro':    
        model = ribeiro_torch
        arguments = dict(residual_blocks_ribeiro_torch=residual_blocks_ribeiro_torch,
                         skip_connection_torch=skip_connection_torch,
                         rate_drop=0.2,
                         in_channels=12,
                         downsample=4
        )
        
    else:
        raise NameError(" Wrong Name. Allowed names are 'rajpurkar' and 'ribeiro'. ") 
    
    optim = optimizer
    
    return model, arguments, optim, learning_rate

# A function to help with memory issues.
class ClearCache:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()
        
def computate_predictions(model, dataset, size_test):

    '''
    This function computes the predictions of the model.

    inputs:
        model: the trained Pytorch model;
        dataset: the dataset used for the predictions;
        size_test: it's something like a 'batch size'. It will divide the 'dataset'
        into smaller chuncks to help with memory problems;
        
    return:
        prediction_bin: the binary predictions. It have a shape of (quantity, predictions).
                        In the 'predictions' dimension, 0 is False and 1 is True, assigning the label.                        

    '''

    # Creating the list that will receive the predictions
    prediction_bin = np.empty(dataset.labels.shape)

    if dataset.labels.shape[0] <= size_test:
        size_test = dataset.labels.shape[0]

    # Iterating trought the dataset. I was having GPU memory issues, so I made the code this way.
    # There must have a proper way to do this.
    for i in tqdm.tqdm(range(dataset.labels.shape[0]//size_test)):
        with ClearCache():
            first = (i*size_test)
            second = ((i+1)*size_test)
            
            passing = torch.tensor(dataset.data[first:second]).cuda()
            prediction = model(passing)
            del passing
                                            
            prediction = prediction.to('cpu')
            
            prediction = prediction.detach().numpy()
            
            # Applying a threshold to the data and appending to prediction_bin
            prediction_bin[first:second] = (prediction > 0.5).astype(np.float32)

            del prediction
        
    if (dataset.labels.shape[0]//size_test)*size_test != dataset.labels.shape[0]:
        first = ((dataset.labels.shape[0]//size_test))*size_test
        with ClearCache():
            passing = torch.tensor(dataset.data[first:-1]).cuda()
            prediction = model(passing)
            del passing
            
            # Moving the data from cuda to cpu
            prediction = prediction.to('cpu')

            prediction = prediction.detach().numpy()
            
            # Applying a threshold to the data and appending to prediction_bin
            prediction_bin[first:-1] = (prediction > 0.5).astype(np.float32)

            del prediction
        
    return prediction_bin

''' The functions below are failed attempts to write code with another possibilities. They were not 
destroyed because they can be rewrite in future time.'''

# # This function receives any torch model and searches for a desired type
# def searching_layer_torch(torch_model, types, desired_type, list_with_found_layers):
    
#     '''
#     The objective of this function is to search for specific layers in a torch model.
#     Doing this, it's possible, for exemple, to see a copy of the weights of all Conv1d layers
#     in a torch model. Probably there is a smarter/builted-in way to do this, but, at 
#     this moment, I could not find it.  
    
#     This function returns a list with the found layers.
#     '''
    
#     try:
#         for layer in torch_model:
#             if type(layer) not in types:
#                 searching_layer_torch(layer, types, desired_type, list_with_found_layers)
#             if type(layer) == desired_type:
#                 list_with_found_layers.append(layer)
#     except TypeError:
#         if type(torch_model) not in types:
#             searching_layer_torch(torch_model.children(), types, desired_type, list_with_found_layers)
            
#     return list_with_found_layers

# def easy_calling_searching_layer_torch(torch_model):
#     types = [nn.BatchNorm1d, nn.ReLU, nn.Sigmoid, nn.Dropout1d, nn.MaxPool1d]
#     desired_type = nn.Conv1d
    
#     list_with_found_layers = searching_layer_torch(torch_model, types, desired_type, [])
#     return list_with_found_layers