"""Save npz files to pickle and create train/valid/test sets"""
import numpy as np 
import gzip as gz 
from tqdm import tqdm
import torch as th 
import pickle
import hydra
from omegaconf import DictConfig



# The four channels in x are 'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'
# You can take more if you modify the preparation script and rerun it. 
# To get a list all the files:
import os 
import glob



@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def pipeline(config: DictConfig) : 

    # Define the directory containing the data
    datad = config["local_data_npz"]
    fnames = glob.glob(os.path.join(datad, "*npz.gz"))

    # Define the features list and their corresponding indices
    features_list = config["dgi"]["training_params"]["features_list"]
    features_index_dict = {'EEG_Fpz-CZ': 0, 'EEG_PZ-Oz': 1, 'EOG_horizontal': 2, 'EMB_submental': 3}
    features_index = [features_index_dict[feature] for feature in features_list]

    devpart = 10
    xtrain, xvalid = None, None
    ytrain, yvalid = None, None

    for fn in tqdm(fnames):
        fp = gz.open(fn, 'rb')
        data = np.load(fp, allow_pickle=False)  # for now, don't care about headers
        x = data['x'][:, :, features_index]  # Take specific features based on indices
        y = data['y']  # Take the labels
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        devlim = x.shape[0] // devpart

        if xtrain is None:
            xtrain = np.zeros((1, x.shape[1], x.shape[2]))
            xvalid = np.zeros((1, x.shape[1], x.shape[2]))
            ytrain, yvalid = np.zeros(1), np.zeros(1)

        xvalid = np.concatenate((xvalid, x[idx[:devlim]]), axis=0)
        yvalid = np.concatenate((yvalid, y[idx[:devlim]]), axis=0)
        xtrain = np.concatenate((xtrain, x[idx[devlim:]]), axis=0)
        ytrain = np.concatenate((ytrain, y[idx[devlim:]]), axis=0)
        del x, y

    # Clean the first dummy example
    xtrain, xvalid = xtrain[1:], xvalid[1:]
    ytrain, yvalid = ytrain[1:], yvalid[1:]

    # Convert to Torch tensors (assuming you are using PyTorch)
    xtrain, xvalid = th.FloatTensor(xtrain), th.FloatTensor(xvalid)
    ytrain, yvalid = th.IntTensor(ytrain), th.IntTensor(yvalid)

    # Define the filename for the pickle file
    filename = 'cassette'+'_'.join(features_list) + '.pck'
    outf= os.path.join(config["save_dir"], filename)
    fp = open(outf,"wb")
    pickle.dump((xtrain , xvalid , ytrain , yvalid), fp)

    
if __name__ == "__main__":
    pipeline()  # pylint: disable=E1120