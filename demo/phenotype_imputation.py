#-----------------------------------------------------------------------------
# Impute the phenotype (co-training)
# 
# Reference: Menno Witteveen
#            Damian Roqueiro
#-----------------------------------------------------------------------------

# External modules 
import config.sys_config as sysconfig

# Class imports
from classes.dataset import Dataset
from classes.config_state import ConfigState

# System libraries
import logging
import numpy as np
import sklearn as skl
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn import metrics #roc_curve, auc
from sklearn import cross_validation
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing

# Pipeline auxiliary functions
from generic_functions import find_vec_entries_that_contain
from generic_functions import harden_labels

import IPython as ip

def phenotype_imputation(data, config):
    ''' 
    Function to impute the labels on II based on the classifier learned on I.
    
    Parameters 
    ---------- 
    data : an object of class Dataset that contains: genotypes, covariates, 
        labels and information about random folds 

    config : an object of class ConfigState. It contains the user-entered 
        parameters in a YAML format.
        See the config_file parameter in the main script for more details.
    '''
    # Parameters for this task
    num_folds = data.num_folds  
    task_name    = "phenotype_imputation"
    n_estimators = config.get_entry(task_name, "n_estimators")
    romans_trn   = config.get_entry(task_name, "romans_used_for_learning")
    romans_tst   = config.get_entry(task_name, "romans_used_for_imputing")
    
    # Iterate through the folds: 
    i = 0
    size_of_two = find_vec_entries_that_contain(data.folds[:,0], romans_tst).shape[0]
    soft_labels = np.zeros((size_of_two, num_folds))
    X_scaled = preprocessing.scale(data.clin_covariate.transpose()).transpose()
    fpr = dict()
    tpr = dict()
    thres = dict()
    roc_auc = np.zeros(num_folds)
    for fold in data.folds.transpose():      
        logging.info("Fold=%d" % (i + 1))
        sel_trn = find_vec_entries_that_contain(fold,[romans_trn])
        sel_tst = find_vec_entries_that_contain(fold,[romans_tst])

        model = BaggingClassifier(base_estimator=linear_model.LogisticRegression(),
                    n_estimators=n_estimators, max_samples=0.632, 
# for small set I   n_estimators=n_estimators, max_samples=0.8, 
                    max_features=5, 
                    bootstrap=True, bootstrap_features=True, oob_score=False, 
# for small set I   bootstrap=False, bootstrap_features=True, oob_score=False, 
                    n_jobs=1, random_state=None, verbose=0)
            
        model.fit(X_scaled[:,sel_trn].transpose(), data.labels[:,sel_trn].transpose())

        soft_labels[:,i] = model.predict_proba(X_scaled[:,sel_tst].transpose())[:,1]
        fpr[i], tpr[i], thres[i] = metrics.roc_curve(data.labels[0,sel_tst], soft_labels[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        i+=1

    # Save the output of this task
    config.save_variable(task_name, "%f", soft_labels=soft_labels, roc_auc=roc_auc)
