# standard python packages
import os, sys
from os.path import join, dirname
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import h5py
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append(join(dirname(__file__), "../"))
from CNNpipeline.utils import arrayDatasetWithSubID
from CNNpipeline.models import *

sys.path.append(join(dirname(__file__), "../../nitorch/"))
from nitorch.transforms import  *

# trained models should be provided as a tuple with (dir, modelfile, results csv file)
TRAINED_MODELS = [
    # ('../results/viztest/', 'run0_*.h5', 'run0.csv'),
    ('CNNpipeline/results/IMAGEN*fullbrain-fu3-*/20220624-*', 
     'run0_*-final.h5', 'run.csv'),
    ('CNNpipeline/results/IMAGEN*fullbrain-fu3-*/20220624-*',
     'run3_*-final.h5', 'run.csv'),
    ('CNNpipeline/results/IMAGEN*fullbrain-fu3-*/20220624-*',
     'run4_*-final.h5', 'run.csv'),
    ('CNNpipeline/results/IMAGEN*fullbrain-fu3-*/20220624-*',
     'run7_*-final.h5', 'run.csv')# TODO: automatically determine
]

DATASET = "/ritter/share/data/IMAGEN/h5files/fullbrain-fu3-z2-bingel3u6-n696.h5"
DF_DATA = "/ritter/share/data/IMAGEN/posthoc/all_IMAGEN_Binge_FU3_SS_ver02.csv"
DF_DATA_INDEX = 'ID'
OUT_FILE = "/ritter/roshan/workspace/playground/build/data/IMAGEN.csv"
GPU=6


def main():
    
    df_out = pd.read_csv(DF_DATA).set_index(DF_DATA_INDEX)
    meta_data_dict = {}
    
    # 1) load the data X and i 
    with h5py.File(DATASET, 'r') as data:
        N = None
        X, idx = data['X'][:N], data['i'][:N]
        print(f"Successfully loaded data from {DATASET}")
            
        # iterate over all trained models
        for path, modelfile, resultsfile in TRAINED_MODELS:
            
            # 2) Load results file
            resultsfile = glob(join(path, resultsfile))
            assert len(resultsfile)==1, f"check path provided to the results csv file.\
Found {resultsfile} in the provided path."
            results = pd.read_csv(resultsfile[0])
            # extract the row related to this run as a pd series
            run_id = modelfile.split("_")[0]
            print(f"Starting extraction for run_id = '{run_id}'")
            results =results.loc[results['run_id']==run_id].squeeze(axis=0)
            
            # reload the label         
            y = data[results.out][:N]
            if N is None: N=len(y)
            
            # 3) generate the pytorch Dataset
            dataset = arrayDatasetWithSubID(X, y, idx, transform=transforms.Compose(
                                               [IntensityRescale(), ToTensor()]))
            
            batch_size = 20 if len(y)>20 else len(y)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # 4) Load model weights
            trained_model = glob(join(path, modelfile))[0]
            print(f"Extracting 3D features from '{trained_model}' \n\
    trained with n= {results.n_samples} samples and label= {results.out}..")
            # load trained model weights and print the output of loading
            # model = eval(results['model'].values[0])(**eval(results['model_params'].values[0])) ## TODO
            model = FCN_3D_hooked(**eval(results.model_params))
            print(f"Loading trained weights.. \n ", model.load_state_dict(torch.load(trained_model)))
            # enable storing results
            model._reset_hook()
            model._hook_viz_layer()


            # 5) Run inference on loaded data
            # init vars
            all_preds = np.empty(0)
            all_pred_probas = np.empty((0,2))
            all_labels = np.empty(0)
            all_subidxs = np.empty(0)

            model = model.cuda(GPU)
            model.eval()
            
            with torch.no_grad():
                for sample in tqdm(data_loader):
                    img, label, sub_id = sample["image"], sample["label"], sample["i"]

                    img = img.to(GPU)
                    output = model.forward(img)

                    # the class with the highest logit is the predicted class
                    pred = torch.argmax(output, dim=-1).cpu().numpy().reshape(-1)
                    # the predicted probability per class        
                    pred_proba = F.softmax(output, dim=-1).cpu().numpy().reshape(-1, 2)
                    # store everything for calculating accuracy 
                    all_preds = np.append(all_preds, pred, axis=0)
                    all_pred_probas = np.append(all_pred_probas, pred_proba, axis=0)
                    all_labels = np.append(all_labels, label.numpy().reshape(-1), axis=0)
                    all_subidxs = np.append(all_subidxs, sub_id.numpy().reshape(-1), axis=0)

            assert (len(all_preds)==len(all_labels)) and (len(all_preds)==len(all_subidxs))
            print(f"Finished inference on n={len(all_subidxs)} data..")
            
            # 5) covert the generated results to df with DF_DATA_INDEX as index
            method_name = f"{results.model}-{results.out}-{results.run_id}".replace('_','') # TODO change to results['model_name'] # 
            print("Unique label used for this method =", method_name)
            # create a table with the features as X Y Z
            df_features = pd.DataFrame(data=model.features_3D_outputs, 
                 columns=[f'Rep_{method_name}_X',f'Rep_{method_name}_Y',f'Rep_{method_name}_Z'], # TODO change DR to something more useful
                 index=all_subidxs.astype(int))
            df_features.index.name = DF_DATA_INDEX

            # add predicted probabities per class
            for i in range(all_pred_probas.shape[-1]):
                df_features[f"Pred_{method_name}_class{i}"] = all_pred_probas[:,i]

            # add TP TN FP FN related info
            # label_map = {i:val for i,val in enumerate(df_out[lbl.title()].unique())} ## TODO map values to Names
            df_preds = pd.DataFrame(data = np.array([all_labels, all_preds]).T, 
                            columns=['true', 'pred'], dtype='int', index=all_subidxs.astype(int))
            df_preds = df_preds.apply(lambda row: f"True-{row['true']} Pred-{row['pred']}", axis=1)
            df_preds.index.name = DF_DATA_INDEX
            df_out[f"Pred_{method_name}"] = df_preds
            
            # add info on which subjects were used as validation IDs
            val_ids = eval(results.val_ids)
            df_features[f'Pred_{method_name}_valdata'] = False
            val_ids = [val_id for val_id in val_ids if val_id in df_features.index]
            df_features.loc[val_ids, f'Pred_{method_name}_valdata']=True
            
            # TODO: also add the 'hold_ids' # must provide the holdout h5 file
            # try:
            #     test_ids = eval(results.hold_ids)
            #     df_features[f'Pred_{method_name}_testdata'] = False
            #     test_ids = [test_id for test_id in test_ids if test_id in df_features.index]
            #     df_features.loc[test_ids, f'Pred_{method_name}_testdata']=True
            # except AttributeError as e:        
            #     print("[WARN] no holdout IDs found in results csv. skipping with below exception..")
            #     print(e)
            
            # 6) add any meta data
            # add prediction accuracy as a meta data
            meta_data_dict.update({
                f"{method_name}_accuracy(balanced)": balanced_accuracy_score(all_labels, all_preds)})
            # add layer used as a meta data
            meta_data_dict.update({
                f"{method_name}_y(label)": results.out})
            # add model args as a meta data
            meta_data_dict.update({
                f"{method_name}_augs": list(eval(results.model_params).items())})
            
            # 7) append results to tool's csv file and save it          
            df_out = pd.concat([df_out, df_features], axis=1, join="inner") # only keep subjects that were evaluated here
            
            print(f"Finished with {modelfile} !!")
            del model, data_loader, dataset
            
            
    # 8) add any additional meta data information collected
    if meta_data_dict:
        for k,v in meta_data_dict.items():
            if not isinstance(v, (np.ndarray, list, tuple)): v = [v]
            df_out[f'Meta_{k}'] =  v + (len(df_out)-len(v))*[np.nan]

    print(f"Saving at {OUT_FILE}..")

    df_out.index.name = "subjectID" # for the tool compatibility
    df_out.to_csv(f"{OUT_FILE}")
        
        
if __name__ == "__main__": main()