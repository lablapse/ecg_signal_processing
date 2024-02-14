# Python packages
import matplotlib.pyplot as plt # for plotting
import numpy as np # some fundamental operations
import pathlib # for the paths 
import plot_utils as putils # importing custom code
import pandas as pd # for .csv manipulation
import seaborn as sns # used in some plotting

''' 
    This script compiles a lot of functions used in the main script - grid_search_torch.py -.
    The name - utils_general.py -, is granted because it does not have any Pytorch
    code in it.
'''

# Plot results
def plot_results(history, name, metric, plot_path='plots'):

    '''
    inputs:
        history; name: str;
        metric: str;
        plot_path:str;
        
    return:
        This function returns nothing
    '''

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(history.epoch, history.history[metric], '-o')
    ax.plot(history.epoch, history.history[f'val_{metric}'], '-*')

    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric}'.capitalize())
    ax.legend(['Training set', 'Validation set'])

    # Save figure in multiple formats
    filename = plot_path / f'[{name}][{metric}]'
    fig.savefig(f'{filename}.png', format='png', dpi=600)
    fig.savefig(f'{filename}.pdf', format='pdf')

    return


# This function plots the normalized confusion matrix from mlcm
def plot_confusion_matrix(cm, model_name, target_names, plot_path='results'):

    '''
    inputs:
        cm: np.ndarray; 
        model_name: str; 
        target_names: list;
        plot_path: str;
        
    return:
        This function returns nothing
    '''

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path) / model_name
    plot_path.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    target_names = np.array([*target_names, 'NoC'])

    # Calculating the normalization of the confusion matrix
    divide = cm.sum(axis=1, dtype='int64')
    divide[divide == 0] = 1
    cm_norm = 100 * cm / divide[:, None]

    # Plot the confusion matrix
    fig = plot_cm(cm_norm, target_names)
    name = f"{model_name.split('-')[0]}-cm"
    tight_kws = {'rect' : (0, 0, 1.1, 1)}
    putils.save_fig(fig, name, path=plot_path, figsize='square',
                    tight_scale='both', usetex=False, tight_kws=tight_kws)

    return


def plot_cm(confusion_matrix, class_names, fontsize=10, cmap='Blues'):

    '''
    inputs:
        confusion_matrix: np.ndarray; 
        class_names: list;
        fontsize:int; 
        cmap: str;
        
    return:
        fig: Figure;
    '''

    # Plot the confusion matrix
    fig, ax = plt.subplots()

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    sns.heatmap(df_cm, annot=True, square=True, fmt='.1f', cbar=False, annot_kws={"size": fontsize},
        cmap=cmap, xticklabels=class_names, yticklabels=class_names, ax=ax)
    for t in ax.texts:
        t.set_text(t.get_text() + '%')

    xticks = ax.get_xticklabels()
    xticks[-1].set_text('NPL')
    ax.set_xticklabels(xticks)

    yticks = ax.get_yticklabels()
    yticks[-1].set_text('NTL')
    ax.set_yticklabels(yticks)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.set_xlabel('Rótulo predito')
    ax.set_ylabel('Rótulo verdadeiro')
    fig.tight_layout()

    return fig


def get_mlcm_metrics(conf_mat):
    
    '''
    
    input:
        conf_mat: np_ndarray -> the 'conf_mat' returned in the 'cm' function from the 'mlcm' paper;
        
    return:
        d: dict;
    '''
    
    num_classes = conf_mat.shape[1]
    tp = np.zeros(num_classes, dtype=np.int64)  
    tn = np.zeros(num_classes, dtype=np.int64)  
    fp = np.zeros(num_classes, dtype=np.int64)  
    fn = np.zeros(num_classes, dtype=np.int64)  

    precision = np.zeros(num_classes, dtype=float)  
    recall = np.zeros(num_classes, dtype=float)  
    f1_score = np.zeros(num_classes, dtype=float)  

    # Calculating TP, TN, FP, FN from MLCM
    for k in range(num_classes): 
        tp[k] = conf_mat[k][k]
        for i in range(num_classes):
            if i != k:
                tn[k] += conf_mat[i][i]
                fp[k] += conf_mat[i][k]
                fn[k] += conf_mat[k][i]

    # Calculating precision, recall, and F1-score for each of classes
    epsilon = 1e-7 # A small value to prevent division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * tp / (2 * tp + fn + fp + 2 * epsilon)

    divide = conf_mat.sum(axis=1, dtype='int64') # sum of each row of MLCM

    if divide[-1] != 0: # some instances have not been assigned with any label 
        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

    else:
        precision = precision[:-1]
        recall = recall[:-1]
        f1_score = f1_score[:-1]
        divide = divide[:-1]
        num_classes -= 1

        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

    # construct a dict to store values
    d = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision,\
        'recall': recall, 'f1_score': f1_score, 'divide': divide,\
        'micro_precision': micro_precision, 'macro_precision': macro_precision,\
        'weighted_precision': weighted_precision, 'micro_recall': micro_recall,\
        'macro_recall': macro_recall, 'weighted_recall': weighted_recall,\
        'micro_f1': micro_f1, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
    
    return d

def load_data(test=False):
    """
    Load data from the 'data.npz' file.

    inputs:
        test: boolean;

    Returns:
        X_train, y_train, X_val, y_val: list of train and validation data if test=False;
        X_train, y_train, X_val, y_val, X_test, y_test: list of train, validation and test data if test=True;
        
        The values in the returned list are numpy.ndarray type
    """
    info = []
    with np.load('data.npz') as data:
        info.append(data['X_train'])
        info.append(data['y_train'])
        info.append(data['X_val'])
        info.append(data['y_val'])
        if test:
            info.append(data['X_test'])
            info.append(data['y_test'])
            
    return info

def get_mlcm_report(conf_mat, target_names, model_name):
    '''
    This function is a modified version of the 'stats' function presented in the mlcm paper.
    
    About mlcm:
    Please read the following paper for more information:\
    M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix, 
    IEEE Access, Feb. 2022, DOI: 10.1109/ACCESS.2022.3151048

    
    inputs:
        conf_mat: numpy.ndarray -> the 'conf_mat' returned from the 'cm' function of the 'mlcm' paper;
        target_names: list;
        model_name: str;
    '''
    
    num_classes = conf_mat.shape[1]
    tp = np.zeros(num_classes, dtype=np.int64)  
    tn = np.zeros(num_classes, dtype=np.int64)  
    fp = np.zeros(num_classes, dtype=np.int64)  
    fn = np.zeros(num_classes, dtype=np.int64)  

    precision = np.zeros(num_classes, dtype=float)  
    recall = np.zeros(num_classes, dtype=float)  
    f1_score = np.zeros(num_classes, dtype=float)  

    # Calculating TP, TN, FP, FN from MLCM
    for k in range(num_classes): 
        tp[k] = conf_mat[k][k]
        for i in range(num_classes):
            if i != k:
                tn[k] += conf_mat[i][i]
                fp[k] += conf_mat[i][k]
                fn[k] += conf_mat[k][i]

    # Calculating precision, recall, and F1-score for each of classes
    epsilon = 1e-7 # A small value to prevent division by zero
    precision = tp/(tp+fp+epsilon)
    recall = tp/(tp+fn+epsilon)
    f1_score = 2*tp/(2*tp+fn+fp+epsilon)

    divide = conf_mat.sum(axis=1, dtype='int64') # sum of each row of MLCM

    d = {}

    if divide[-1] != 0: # some instances have not been assigned with any label 
        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

        total_weight = divide.sum()
        
        for k in range(num_classes-1):
            d[f'{target_names[k]}'] = {'precision':precision[k], 'recall':recall[k], \
                                     'f1_score':f1_score[k], 'weight':divide[k]}
        k = num_classes-1
        d['NTL'] = {'precision':precision[k], 'recall':recall[k], \
                    'f1_score':f1_score[k], 'weight':divide[k]}

        d['micro avg'] = {'precision':micro_precision, 'recall':micro_recall, \
                          'f1_score':micro_f1, 'weight':total_weight}
        
        d['macro avg'] = {'precision':macro_precision, 'recall':macro_recall, \
                          'f1_score':macro_f1, 'weight':total_weight}
        
        d['weighted avg'] = {'precision':weighted_precision, 'recall':weighted_recall, \
                          'f1_score':weighted_f1, 'weight':total_weight}
    else:
        precision = precision[:-1]
        recall = recall[:-1]
        f1_score = f1_score[:-1]
        divide = divide[:-1]
        num_classes -= 1

        micro_precision = tp.sum()/(tp.sum()+fp.sum())
        macro_precision = precision.sum()/num_classes
        weighted_precision = (precision*divide).sum()/divide.sum()

        micro_recall = tp.sum()/(tp.sum()+fn.sum())
        macro_recall = recall.sum()/num_classes
        weighted_recall = (recall*divide).sum()/divide.sum()

        micro_f1 = (2*tp.sum())/(2*tp.sum()+fn.sum()+fp.sum())
        macro_f1 = f1_score.sum()/num_classes
        weighted_f1 = (f1_score*divide).sum()/divide.sum()

        total_weight = divide.sum()
        
        for k in range(num_classes):
            d[f'{target_names[k]}'] = {'precision':precision[k], 'recall':recall[k], \
                                     'f1_score':f1_score[k], 'weight':divide[k]}
            
        # print(sp,' NoC',sp,'There is not any data with no true-label assigned!')
        d[f'NoC'] = {'precision':'no data', 'recall':'no data', \
                     'f1_score':'no data', 'weight':'no data'}

        d['micro avg'] = {'precision':micro_precision, 'recall':micro_recall, \
                          'f1_score':micro_f1, 'weight':total_weight}        
        
        d['macro avg'] = {'precision':macro_precision, 'recall':macro_recall, \
                          'f1_score':macro_f1, 'weight':total_weight}    
            
        d['weighted avg'] = {'precision':weighted_precision, 'recall':weighted_recall, \
                          'f1_score':weighted_f1, 'weight':total_weight}    
        
    # Path
    csv_report = f'results/{model_name}/report.csv'
    csv_path_auc = f'results/{model_name}/roc_auc.csv'
    
    # Convert strings to Path type
    csv_report = pathlib.Path(csv_report)
    csv_path_auc = pathlib.Path(csv_path_auc)

    # Make sure the files are saved in a folder that exists
    csv_report.parent.mkdir(parents=True, exist_ok=True)
    csv_path_auc.parent.mkdir(parents=True, exist_ok=True)
        
    pd.DataFrame.from_dict(d, orient='index').to_csv(csv_report)

    return

''' 
The function 'gv_and_pdf_model' gives a pdf of the structure of the model. I had some problems
with this function, that's the reason of it existing as a comment. But it can be used 
to generate an image of a model.
'''

# def gv_and_pdf_model(model, model_name, shape1, shape2):
    
#     '''
#     This function generates a .gv and a PDF file with an image of the structure of the passes model
    
#     inputs: model -> 
#     model_name: str;
#     shape1: int;
#     shape2: int;
#     '''
    
#     # Saving a .gv of the model
#     model_graph = torchview.draw_graph(model, input_size=(1, shape1, shape2),
#                                     graph_name=f'{model_name}', hide_module_functions=False, depth=100)
#     # I need to check if this line of code below is really necessary
#     model_graph.visual_graph
#     model_graph.visual_graph.save()
#     # Saving a PDF of the model
#     dot = graphviz.Source.from_file(f'{model_name}.gv')
#     dot.render()
#     return