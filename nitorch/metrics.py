import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score, roc_curve, auc, log_loss
import numpy as np
from scipy.special import softmax, expit
from sklearn.dummy import DummyClassifier

def specificity(y_true, y_pred):
    """Gets the specificity of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The specificity.

    """
    return recall_score(y_true, y_pred, pos_label=0)


def sensitivity(y_true, y_pred):
    """Gets the sensitivity of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The sensitivity.

    """
    return recall_score(y_true, y_pred, pos_label=1)


def auc_score(y_true, y_pred):
    """Gets the auc score of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The auc score.

    """

    y_true, y_pred = prepare_values(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def explained_deviance(y_true, y_pred_logits=None, y_pred_probas=None, 
                       returnloglikes=False):
    """Computes explained_deviance score to be comparable to explained_variance"""
    assert y_pred_logits is not None or y_pred_probas is not None, "Either the predicted probabilities \
(y_pred_probas) or the predicted logit values (y_pred_logits) should be provided. But neither of the two were provided."
    
    #y_pred_logits is a tensor a we need to change it to a np array as y_true
    y_pred_logits = torch.cat(y_pred_logits, dim=0).cpu().detach().numpy()
    
    if y_pred_logits is not None and y_pred_probas is None:
        # check if binary or multiclass classification
        if y_pred_logits.ndim == 1: 
            y_pred_probas = expit(y_pred_logits)
        elif y_pred_logits.ndim == 2: 
            y_pred_probas = softmax(y_pred_logits)
        else: # invalid
            raise ValueError(f"logits passed seem to have incorrect shape of {y_pred_logits.shape}")
            
    if y_pred_probas.ndim == 1: y_pred_probas = np.stack([1-y_pred_probas, y_pred_probas], axis=-1)
    
    # compute a null model's predicted probability
    X_dummy = np.zeros(len(y_true))
    y_null_probas = DummyClassifier(strategy='prior').fit(X_dummy,y_true).predict_proba(X_dummy)
    #strategy : {"most_frequent", "prior", "stratified", "uniform",  "constant"}
    # suggestion from https://stackoverflow.com/a/53215317
    llf = -log_loss(y_true, y_pred_probas, normalize=False)
    llnull = -log_loss(y_true, y_null_probas, normalize=False)
    ### McFadden’s pseudo-R-squared: 1 - (llf / llnull)
    explained_deviance = 1 - (llf / llnull)
    ## Cox & Snell’s pseudo-R-squared: 1 - exp((llnull - llf)*(2/nobs))
    # explained_deviance = 1 - np.exp((llnull - llf) * (2 / len(y_pred_probas))) ## TODO, not implemented
    if returnloglikes:
        return explained_deviance, {'loglike_model':llf, 'loglike_null':llnull}

    else:
        return explained_deviance
    
def mz_rsquare(y_pred_logits):
    y_pred_logits = torch.cat(y_pred_logits, dim=0).cpu().detach().numpy()
    return np.var(y_pred_logits) / (np.var(y_pred_logits) + (np.power(np.pi, 2.0) / 3.0))

    
    """
def explained_deviance(y_true, y_pred_probas, normalize = False):
    Computes explained_deviance score to be comparable to explained_variance
    print(y_true,len(y_true), y_pred_probas,  len(y_pred_probas)) 
    #y_true = torch.tensor(np.asarray(y_true)).cpu()
    
    b = torch.cat(y_pred_probas, dim=0)
    y_pred_probas = b.cpu().detach().numpy()
    
    dummy = DummyClassifier(strategy="uniform")
    x = np.zeros(len(y_true))
    dummy.fit(x,y_true)
    
    deviance_model = 2*log_loss(y_true, y_pred_probas, normalize = True)
    deviance_null = 2*log_loss(y_true, dummy.predict_proba(x), normalize = True)
    explained_deviance = 1 - (deviance_model / deviance_null)
    print(deviance_model, deviance_null, explained_deviance)
    return explained_deviance

def llnull(self):
    return self.family.loglike(self._endog, self.null,
                                   var_weights=self._var_weights,
                                   freq_weights=self._freq_weights,
                                   scale=self.scale)
"""