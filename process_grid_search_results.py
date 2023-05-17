import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the csv file
df = pd.read_csv('grid_search_results.csv')
# Drop columns that are not needed
columns_to_drop = ['index', 'train_loss', 'train_accuracy', 'train_precision_macro_avg',\
                    'train_recall_macro_avg', 'train_f1_score_macro_avg','val_loss',\
                    'val_accuracy','val_precision_macro_avg', 'val_recall_macro_avg']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Exhibit the first 10 rows
df.head(10)

def plot_f1_score_by_batch_size(df, model, learning_rates):
    # Set up subplots
    fig, axes = plt.subplots(nrows=len(learning_rates), ncols=1, figsize=(9, 4.5), sharey=True)  
    fig.subplots_adjust(wspace=.3, hspace=.3, top=0.9, bottom=0.1, left=0.1, right=0.9)
    grayscale_colors = ['0.2', '0.5', '0.8']
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]

        # Extract the data
        columns_of_interest = ['optimizer', 'batch_size', 'val_f1_score_macro_avg']
        df2 = df.query(f'model_name == "{model}" and learning_rate == {lr}')[columns_of_interest]

        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
        # data from https://allisonhorst.github.io/palmerpenguins/
        group_by = np.unique(df2['batch_size'])
        data = {}   
        for opt in np.unique(df2['optimizer']):
            data[opt] = df2[df2['optimizer'] == opt]['val_f1_score_macro_avg'].values
        
        x = np.arange(len(group_by))  # the label locations
        width = 0.3  # the width of the bars

        for i, (attribute, value) in enumerate(data.items()):
            offset = width * i
            rects = ax.bar(x + offset, value, width, label=attribute, color=grayscale_colors[i])
            ax.bar_label(rects, padding=.05, fmt='%.2f')

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
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['Adam', 'RMSProp', 'SGD'], loc='upper center',\
               ncol=len(data), bbox_to_anchor=(0.5, 1.0), edgecolor='none', facecolor='none')
    plt.show()

# Extract unique values for plotting
learning_rates = np.unique(df['learning_rate'])
models = np.unique(df['model_name'])
for model in models:
    plot_f1_score_by_batch_size(df, model, learning_rates)