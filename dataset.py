import ast # used to transform strings into objects from ast.literal_eval
import numpy as np # used for basic manipulations
import pandas as pd # used to manipulate the .csv files
import pathlib # used to create the path information
import tqdm # used to measure the progress of an iteration
import wfdb # used to extract the ECG signal

'''

    This script collects the information in the PTB-XL database and separate in 'train', 'validation' and 'test' information.
    Train dataset - folds 1 to 8;
    Validation dataset - fold 9;
    Test dataset - fold 10;
    
    The PTB-XL was manipulated to rule out diagnoses with less than 50% certainty.
    Signals with empty labels are discarted.
    
'''

def load_data(df, data_folder, sampling_rate):
    
    '''
    inputs:
        df: pandas.core.frame.DataFrame;
        data_folder: pathlib.PosixPath; 
        sampling_rate: int -> 100 or 500 Hz;
        
    return:
        data: numpy.ndarray;
    '''
    
    # Better to use pathlib instead of os and strings
    data_folder = pathlib.Path(data_folder)

    # Get information about one example file. It collects for 100Hz sampling.
    if sampling_rate == 100:
        _, info = wfdb.rdsamp(str(data_folder / df.filename_lr[0]))
        selected = df.filename_lr

    # Get information about one example file. It collects for 500Hz sampling.
    elif sampling_rate == 500:
        _, info = wfdb.rdsamp(str(data_folder / df.filename_hr[0]))
        selected = df.filename_hr
        
    # Raising an error for invalid sampling_rate value 
    else:
        raise ValueError(" Wrong value to samplig_rate. The accepted ones are 100 and 500 Hz. ")

    # Initialize dataset
    num_examples = df.shape[0]
    num_samples = info['sig_len']
    num_channels = info['n_sig']
    data = np.empty([num_examples, num_samples, num_channels])

    # Fill numpy array
    for i, filename in enumerate(tqdm.tqdm(selected)):
        x, _ = wfdb.rdsamp(str(data_folder / filename))
        data[i, ] = x

    return data


def simple_diagnostic(scp_codes):
    
    '''
    This function ignores the diagnoses with less than 50% centainty
    
    inputs:
        scp_codes: dict;
    
    return:
        vec: numpy.ndarray;
    
    '''
    
    vec = np.zeros(len(super_classes), dtype='int')
    for key, item in scp_codes.items():
        if key in meta_scp.index:
            diag_class = subdiag_dict[key]
            if item >= 50:
                vec[super_classes.index(diag_class)] = 1

    # No diagnostic class present.
    if vec.sum() == 0:
        return '???'

    return vec

# Defining path
path = '/datasets/ptbxl' # path to the PTB-XL database
path = pathlib.Path(path) # creating the pathlib.PosixPathvariable to be passed to the load_data function

# selecting the sampling rate from the PTB-XL database. The other option would be 500 Hz.
sampling_rate = 100 

# Load and convert annotation data
metadata = pd.read_csv(path / 'ptbxl_database.csv') 
metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
meta_scp = pd.read_csv(path / 'scp_statements.csv', index_col=0)
meta_scp = meta_scp[meta_scp.diagnostic == 1]

# Defining the superclasses
super_classes = ['NORM','STTC','CD','MI','HYP']

# Defining a type dict variable with the subclasses and superclasses values
subdiag_dict = dict(meta_scp.diagnostic_class) # Key = subclasses, item = superclasses

# Simplify diagnostic
metadata['diagnostic_superclass'] = metadata.scp_codes.apply(simple_diagnostic)
metadata = metadata.drop(np.where(metadata.diagnostic_superclass == '???')[0])

# Load labels
Y = metadata.diagnostic_superclass.values
Y = np.array(Y.tolist())    

# Load raw ECG data
X = load_data(metadata, path, sampling_rate)

# The dataset has 10 possible "validation folds",
# Folds 1-8 for training, fold 9 for validation and fold 10 for test
test_fold = 10
val_fold = 9


''' Creating the datasets '''
# Test (fold 10)
X_test = X[metadata.strat_fold.values == test_fold]
y_test = Y[metadata.strat_fold.values == test_fold]

# Validation (fold 9)
X_val = X[metadata.strat_fold.values == val_fold]
y_val = Y[metadata.strat_fold.values == val_fold]

# Train (folds 1-8)
X_train = X[metadata.strat_fold.values < val_fold]
y_train = Y[metadata.strat_fold.values < val_fold]


# Save metadata
metadata.to_csv('metadata.csv', index=False)
meta_scp.to_csv('metadata_scp.csv', index=False)

# Transposing dimensions to work correctly with Pytorch
X_train = np.transpose(X_train, axes=(0, 2, 1))
X_test = np.transpose(X_test, axes=(0, 2, 1))
X_val = np.transpose(X_val, axes=(0, 2, 1))

# Prepare data for saving
data = dict(
    X_train=X_train.astype('float32'),
    y_train=y_train.astype('int8'),
    X_val=X_val.astype('float32'),
    y_val=y_val.astype('int8'),
    X_test=X_test.astype('float32'),
    y_test=y_test.astype('int8')
)

# Save data as compressed numpy binaries
np.savez('data.npz', **data)
