import numpy as np
import os
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class pytorch_tests(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.BatchNorm1d(64, eps=0.001, momentum=0.99),
            nn.Conv1d(64, 4, 1),
            nn.Flatten(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        out = self.stack(x)
        return out
    
def train_loop(dataloader, model, loss_fn, optimizer, path):
    model.train()
    for (X, Y) in dataloader:
        # Compute prediction and loss
        Y = Y.float()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)

        # saving gradients before and after the optimization process
        weights_before = [] 
        for param in model.parameters():
            weights_before.append(param)
        
        dict_weights_before = dict(
            batchnorm0 = weights_before[0].clone().detach().numpy(),
            batchnorm1 = weights_before[1].clone().detach().numpy(),
            conv_weights = weights_before[2].clone().detach().numpy(),
            conv_bias = weights_before[3].clone().detach().numpy(),
            linear_weights = weights_before[4].clone().detach().numpy(),
            linear_bias = weights_before[5].clone().detach().numpy(),
            forward = pred.clone().detach().numpy(),
            loss = loss.clone().detach().numpy()
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # saving gradients after and after the optimization process
        weights_after = []
        gradients_after_backward_call = []
        for param in model.parameters():
            weights_after.append(param)
            gradients_after_backward_call.append(param.grad)
        
        dict_weights_after = dict(batchnorm0 = weights_after[0].clone().detach().numpy(),
                                   batchnorm1 = weights_after[1].clone().detach().numpy(),
                                   conv_weights = weights_after[2].clone().detach().numpy(),
                                   conv_bias = weights_after[3].clone().detach().numpy(),
                                   linear_weights = weights_after[4].clone().detach().numpy(),
                                   linear_bias = weights_after[5].clone().detach().numpy()
        )

        # Here, the forward and loss are not calculated, but, are passed to the dictionary
        # to maintain a pattern.
        dict_gradients_after_backward_call = dict(batchnorm0 = gradients_after_backward_call[0].clone().detach().numpy(),
                                   batchnorm1 = gradients_after_backward_call[1].clone().detach().numpy(),
                                   conv_weights = gradients_after_backward_call[2].clone().detach().numpy(),
                                   conv_bias = gradients_after_backward_call[3].clone().detach().numpy(),
                                   linear_weights = gradients_after_backward_call[4].clone().detach().numpy(),
                                   linear_bias = gradients_after_backward_call[5].clone().detach().numpy(),
                                   forward = 0,
                                   loss = 0
        )

        new_pred = model(X)
        new_loss = loss_fn(new_pred, Y)

        dict_weights_after['forward'] = new_pred.clone().detach().numpy()
        dict_weights_after['loss'] = new_loss.clone().detach().numpy()

        # managing the saving of the gradients
        path_before = pathlib.Path(f'{path}/weights_before.npz')
        path_after = pathlib.Path(f'{path}/weights_after.npz')
        path_gradients = pathlib.Path(f'{path}/gradients.npz')
        path_before.parent.mkdir(parents=True, exist_ok=True)

        np.savez(path_before, **dict_weights_before)
        np.savez(path_after, **dict_weights_after)
        np.savez(path_gradients, **dict_gradients_after_backward_call)


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

def putting_alltogether(path='torch_gradients'):
    # loading the numpy arrays that will be used as X and Y
    data = np.load('data.npz')
    datasets = CustomDataset(data['X'], data['Y'])
    dataloader = DataLoader(datasets, batch_size=data['X'].shape[0])
    
    # defining some of the structure of the model
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    model = pytorch_tests()

    # loading the weights for convolution and linear, and applying them to the model
    weights = np.load('weights.npz')
    for layer in model.named_children():
        layer[1][1].weight = nn.Parameter(data=torch.tensor(weights['conv_weights'], requires_grad=True))
        layer[1][1].bias = nn.Parameter(data=torch.tensor(weights['conv_bias'], requires_grad=True))
        layer[1][3].weight = nn.Parameter(data=torch.tensor(weights['linear_weights'], requires_grad=True))
        layer[1][3].bias = nn.Parameter(data=torch.tensor(weights['linear_bias'], requires_grad=True))

    # finish the model configuration
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # calling the 'train_loop' function
    train_loop(dataloader, model, loss_fn, optimizer, path)

putting_alltogether()