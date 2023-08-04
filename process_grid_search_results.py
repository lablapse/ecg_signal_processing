import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
This script opens a .csv file and plot the f1-scores by learning rate, model, optimizer and batch size.
'''

# Load the csv file
df = pd.read_csv('maisTestes.csv')

# Drop columns that are not needed
columns_to_drop = ['index', 'train_loss', 'train_accuracy', 'train_precision_macro_avg',\
                    'train_recall_macro_avg', 'train_f1_score_macro_avg','val_loss',\
                    'val_accuracy','val_precision_macro_avg', 'val_recall_macro_avg']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

def plot_f1_score_by_batch_size(df, model, learning_rates):
    
    '''
        This function plots the f1-score from the grid_search.py file for each trained model.
        
    inputs:
        df: pandas.core.frame.DataFrame from the cleaned .csv file;
        model: str;
        learning_rate: list of floats;
    '''
    
    # Set up subplots
    fig, axes = plt.subplots(nrows=len(learning_rates), ncols=1, figsize=(9, 4.5), sharey=True)  
    fig.subplots_adjust(wspace=.3, hspace=.3, top=0.9, bottom=0.1, left=0.1, right=0.9)
    grayscale_colors = ['0.2', '0.5', '0.8']
    
    # Creating rows of plots based on 'learning_rate' values
    for idx, lr in enumerate(learning_rates): 
        try:
            ax = axes[idx]
        except:
            ax = axes

        # Extract the data
        columns_of_interest = ['optimizer', 'batch_size', 'val_f1_score_macro_avg']
        df2 = df.query(f'model_name == "{model}" and learning_rate == {lr}')[columns_of_interest]

        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
        # data from https://allisonhorst.github.io/palmerpenguins/
        # Selecting the information based on 'batch_size' value
        group_by = np.unique(df2['batch_size'])
        data = {}   
        for opt in np.unique(df2['optimizer']):
            data[opt] = df2[df2['optimizer'] == opt]['val_f1_score_macro_avg'].values
        
        x = np.arange(len(group_by))  # the label locations
        width = 0.3  # the width of the bars

        # Creating bars based on the 'optimizer' values for every value in 'batch_size'
        for i, (attribute, value) in enumerate(data.items()):
            offset = width * i
            rects = ax.bar(x + offset, value, width, label=attribute, color=grayscale_colors[i])
            ax.bar_label(rects, padding=.05, fmt='%.2f')

        # Setting the bar plots with the value information of the respective 'batch_size'
        ax.set_xticks(x + width, group_by)
        ax.set_xticklabels(group_by)
        ax.set_ylim(0, 1)
        
        # Add text with the learning rate at the upper left corner of each subplot
        ax.text(0.005, 1, f'$\mu$={lr}', ha='left', va='top', transform=ax.transAxes)       

    # Add shared axis labels
    fig.text(0.5, 0.02, 'Batch size', ha='center', va='center')
    fig.text(0.04, 0.5, 'F1-score (validation)', ha='center', va='center', rotation='vertical')
    fig.text(0.03, .98, model.upper(), ha='left', va='top')
    
    # Add the legend below the subplots
    handles, _ = ax.get_legend_handles_labels()
    
    # Creating the legend about the 'optimizer' values at the upper center of the figure
    fig.legend(handles, ['Adam', 'RMSProp', 'SGD'], loc='upper center',\
               ncol=len(data), bbox_to_anchor=(0.5, 1.0), edgecolor='none', facecolor='none')
    
    # Saving the Figure
    fig.savefig(f'1 - 256_0.01_all {model}.png')
    
    # Showing the created figure
    plt.show()

# Extract unique values for plotting
learning_rates = np.unique(df['learning_rate'])
models = np.unique(df['model_name'])

for model in models:
    plot_f1_score_by_batch_size(df, model, learning_rates)