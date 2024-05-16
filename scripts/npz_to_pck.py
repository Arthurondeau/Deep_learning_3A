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
    features_list = config["training"]["training_params"]["features_list"]
    features_index_dict = {'EEG_Fpz-CZ': 0, 'EEG_PZ-Oz': 1, 'EOG_horizontal': 2, 'EMB_submental': 3}
    features_index = [features_index_dict[feature] for feature in features_list]

    devpart = 10
    xtrain, xvalid = None, None
    ytrain, yvalid = None, None
    subject_dict = {key: value for key, value in zip(range(40, 49), range(10))}

    for fn in tqdm(fnames):
        file_name = os.path.splitext(os.path.basename(fn))[0]
        subject_label = subject_dict[int(file_name[2:4])]
        fp = gz.open(fn, 'rb')
        data = np.load(fp, allow_pickle=False)  # for now, don't care about headers
        x = data['x'][:, :, features_index]  # Take specific features based on indices
        subject_labels = np.zeros(len(data['y']))
        subject_labels[:] = subject_label
        y = np.array([data['y'],subject_labels])  # Take the labels
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        devlim = x.shape[0] // devpart
        if xtrain is None:
            xtrain = np.zeros((1, x.shape[1], x.shape[2]))
            xvalid = np.zeros((1, x.shape[1], x.shape[2]))
            ytrain, yvalid = np.zeros((2,1)), np.zeros((2,1))
        xvalid = np.concatenate((xvalid, x[idx[:devlim]]), axis=0)
        yvalid = np.concatenate((yvalid, y[:,idx[:devlim]]), axis=1)
        xtrain = np.concatenate((xtrain, x[idx[devlim:]]), axis=0)
        ytrain = np.concatenate((ytrain, y[:,idx[devlim:]]), axis=1)
        del x, y

    # Clean the first dummy example
    xtrain, xvalid = xtrain[1:], xvalid[1:]
    ytrain, yvalid = ytrain[:,1:], yvalid[:,1:]
    #Transpose the label tensor
    ytrain, yvalid = np.transpose(ytrain), np.transpose(yvalid)
    # Convert to Torch tensors (assuming you are using PyTorch)
    xtrain, xvalid = th.FloatTensor(xtrain), th.FloatTensor(xvalid)
    ytrain, yvalid = th.IntTensor(ytrain), th.IntTensor(yvalid) 
    print('ytrain',ytrain)
    # Reshape xtrain and xvalid tensors (Nb Samples,Input dim,channels) -> (Nb Samples,channels, Input dim)
    xtrain, xvalid = th.reshape(xtrain,(xtrain.size()[0],xtrain.size()[2],xtrain.size()[1])), th.reshape(xvalid,(xvalid.size()[0],xvalid.size()[2],xvalid.size()[1]))

    # Define the filename for the pickle file
    filename = 'cassette_with_patient_label_'+'_'.join(features_list) + '.pck'
    outf= os.path.join(config["save_dir"], filename)
    fp = open(outf,"wb")
    pickle.dump((xtrain , xvalid , ytrain , yvalid), fp)

    
if __name__ == "__main__":
    pipeline()  # pylint: disable=E1120