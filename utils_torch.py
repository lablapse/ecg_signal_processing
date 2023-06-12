import numpy as np # some fundamental operations
import torch.nn as nn
from torch.utils.data import Dataset

# Creating the kernel initializer for nn.Conv1d and nn.Linear 
def _weights_init(m):
    '''
        This function will be called using self.apply(_weights_init) in some class later.
        Don't worry about 'm'.
    '''
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Sigmoid):
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
        self.layer = nn.Sequential(nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                   nn.ReLU(),
                                   nn.Dropout(p=self.rate_drop), # VER SE OS DROPOUTS ESTAO IGUAIS CONCEITUALMENTE -> torch: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html keras -> https://keras.io/api/layers/regularization_layers/dropout/
                                   nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=16,
                                             stride=1, padding='same'),
                                   nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                   nn.ReLU(),
                                   nn.Dropout(p=self.rate_drop), # VER SE OS DROPOUTS ESTAO IGUAIS CONCEITUALMENTE -> torch: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html keras -> https://keras.io/api/layers/regularization_layers/dropout/
                                   nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=16,
                                             stride=stride, padding='same')
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
        self.skip = nn.Sequential(nn.MaxPool1d(kernel_size=self.downsample, stride=self.downsample, padding='same'),
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
        
        self.layer_alone = nn.Sequential(nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                         nn.ReLU(),
                                         nn.Dropout(p=self.rate_drop)
        )
        
    def forward(self, input):
        layer, skip = input
        skip = self.skip_connection(self.in_channels, self.out_channels, self.downsample)(skip)
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
    def __init__(self, rate_drop, in_channels):
        
        #Internalizing the values
        self.rate_drop = rate_drop
        self.in_channels = in_channels
        
        # Calling the nn.Module 'constructor' 
        super(rajpurkar_torch, self).__init__()
        
        # Creating a 'dict' with values that will be used multiple times in nn.Conv1d() function
        self.conv_config = dict(in_channels=self.out_channels, 
                                out_channels=self.out_channels, 
                                kernel_size=16, 
                                padding='same'
        )
        
        
        # Creating the first block
        self.layer_to_be_passed = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                                          out_channels=self.out_channels,
                                                          kernel_size=16, 
                                                          stride=1, 
                                                          padding='same'),
                                                nn.BatchNorm1d(num_features=self.out_channels, 
                                                               eps=0.001, momentum=0.99),
                                                nn.ReLU()
        )
        
        # Short  connection
        self.skip_alone = nn.MaxPool1d(kernel_size=1, stride=2)
        
        # Creating the second block
        self.layer_to_be_summed = nn.Sequential(nn.Conv1d(stride=1, **self.conv_config),
                                                nn.BatchNorm1d(num_features=self.out_channels, eps=0.001, momentum=0.99),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.rate_drop),
                                                nn.Conv1d(stride=2, **self.conv_config)            
        )
        
        # Creating the channels matrix that will be used to create the connection layers
        self.num_channels = [64, 64, 64,
                             128, 128, 128, 128,
                             192, 192, 192, 192,
                             256, 256, 256, 256
        ]
        
        # Creating the list that will append the middle layers
        middle_layers_list = list()
        
        # Appending the middle layers to the middle_layer_list
        for i in range(15):
            middle_layers_list.append(residual_blocks_rajpurkar_torch(i=i, stride=(i%2)+1, 
                                                                      in_channels=self.num_channels[i], 
                                                                      out_channels=self.num_channels[i+1], 
                                                                      rate_drop=self.rate_drop))
        # Creating the 'nn.Sequential()' using the 'middle_layer_list'
        middle_layers_list = nn.ModuleList(middle_layers_list)
        self.middle_layers = nn.Sequential(middle_layers_list)
        
        # End layer
        self.layer_end = nn.Sequential(nn.BatchNorm1d(num_features=self.num_channels[-1], eps=0.001, momentum=0.99),
                                       nn.ReLU(),
                                       nn.Linear(in_features=1000, out_features=32),
                                       nn.Sigmoid(32, 5)
        )
    
        # This applies to this module and all children of it. This line below initializes the 
        # 'nn.Conv1d()', 'nn.Linear()' and 'nn.Sigmoid()' weights.
        self.apply(_weights_init)
    
    # Calculating "Rajpurkar's" model
    def forward(self, input):
        layer = self.layer_to_be_passed(input)
        skip = self.skip_alone(layer)
        layer = self.layer_to_be_summed(layer)
        out = layer + skip
        out = self.middle_layers(out)
        self.layer_end(out)
        return out
    
class ribeiro_torch(nn.Module):
    
    '''
    This class implements the Ribeiro model like-wise was made by Sarah
    '''
    
    # Passing values to the object
    def __init__(self, rate_drop, in_channels, downsample):
        
        # Calling the nn.Module 'constructor' 
        super(ribeiro_torch, self).__init__()
        
        #Internalizing the values
        self.rate_drop = rate_drop
        self.in_channels = in_channels
        self.downsample = downsample
    
        # Input block
        self.layer_initial = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=64, 
                                                     kernel_size=16, stride=1, padding='same'),
                                           nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.99),
                                           nn.ReLU()
        )
    
        # The channels dimensions
        self.num_channels = np.array([64, 128, 192, 256, 320])
        
        # Crating the list that will receive the middle blocks
        middle_layers_list = list()
        
        # Appending to 'middle_layers_list'
        for i in range(4):
            middle_layers_list.append(residual_blocks_ribeiro_torch(skip_connection_torch, self.num_channels[i], 
                                                                    self.num_channels[i+1], self.rate_drop, self.downsample))
        
        # Creating the middle layers
        self.layers_middle = nn.Sequential(middle_layers_list)

        # Output block
        self.layer_end = nn.Sigmoid(1000, 5)
            
        # This applies to this module and all children of it. This line below initializes the 
        # 'nn.Conv1d()', 'nn.Linear()' and 'nn.Sigmoid()' weights.
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
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx, :, :]
        current_label = self.labels[idx, :]
        return current_sample, current_label