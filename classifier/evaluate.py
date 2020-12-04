import sys, os, csv, math, re, collections, random
import gc, glob

from scipy import interp
from scipy.stats import sem  # compute standard error
import numpy as np
import scipy as sp

from pandas import DataFrame, Series
import pandas as pd 

# plotting 
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

from scipy.stats import sem  # compute standard error

try:
    import cPickle as pickle
except:
    import pickle

import statistics

# feature selection 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE

# classifiers 
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyClassifier

# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.utils.extmath import density
from sklearn import metrics

# CV 
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing   # standardizing feature vectors

# Unsupervised
from sklearn.decomposition import RandomizedPCA

# probability calibration
from sklearn.calibration import CalibratedClassifierCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from batchpheno.utils import div
from batchpheno import icd9utils, sampling, qrymed2, utils, dfUtils
from config import seq_maker_config, sys_config

import math
from sklearn.metrics import roc_curve, auc, roc_auc_score # performance metrics 

def bootstrap_auc_score(y_true, y_pred, n_folds=None, **kargs): 
    """

    Note
    ----
    1. if n_folds = 5, then in each iteration, we only need 
       n_bootstraps/n_folds examples assuming that the size 
       of desired sample is fixed.  
    """
    # from sklearn.metrics import roc_curve, auc, roc_auc_score

    n_total = kargs.get('n_total', 1200)
    assert len(y_true) == len(y_pred), "y_true:\n%s\ny_pred:\n%s\n" % (y_true, y_pred)
    
    # some parameters
    n_bootstraps_min = 5
    n_labels = 2  # positive and negative label

    if not n_folds: n_folds = ROC.n_folds
    if n_folds < 1: n_folds = 1
    if n_folds > n_total: 
        print('warning> n_folds is large: %d > intended n_sample: %d (in LOO mode?)' % (n_folds, n_total))
    n_bootstraps = int(math.ceil(n_total/(n_folds+0.0))) # [1]
    if n_bootstraps < n_bootstraps_min: n_bootstraps = n_bootstraps_min

    rng_seed = 53  # control reproducibility
    bootstrapped_scores = []

    average = kargs.get('average', 'macro')
    n_unique = len(np.unique(y_true))
    # print('info> n_bootstraps: %d, n_unique_label: %d' % (n_bootstraps, n_unique))
    assert n_unique >= n_labels, "number of unique labels may be too small (incomplete data?): %d" % n_unique

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))  # lowest, highest, n_samples
        n_unique = len(np.unique(y_true[indices]))
        if n_unique < 2: # binary classification
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            # print('warning> at iter=%d number of unique labels is only %d > indices(n=%d):\n%s\n' % \
            #     (i, n_unique, len(indices), indices[:20]))
            continue
    
        # print('info> caller: %s, indices: %s' % (kargs.get('message_'), indices) )
        score = roc_auc_score(y_true[indices], y_pred[indices], average=average)
        bootstrapped_scores.append(score)
        # if i % 100 == 0: print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # [test]
    n_samples = len(bootstrapped_scores)
    if n_samples < n_bootstraps_min: 
        lc = collections.Counter(y_true)
        for k, v in lc.items(): 
            print('bootstrap-test> label: %s => size: %d' % (k, v))
        raise ValueError, "Number of bootstrap sample is too small: %d" % n_samples

    return bootstrapped_scores


def selectCV(y, n_folds=5, force_loo=False, p=1, shuffle=False):
    from sklearn.cross_validation import StratifiedKFold, LeaveOneOut  # need cv to take on an iterator of indices

    lc =  collections.Counter(y) 
    n_least = min(lc.values())
    r = n_least/(n_folds+0.0)

    small_sample = True if r < 1 else False

    if not (small_sample or force_loo): 
        cv = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    else: 
        print('selectFeatureCV> LOO mode | sample size: %d wrt n_folds: %d, ratio: %f' % (len(y), n_folds, r))
        if p == 1: 
            cv = LeaveOneOut(n=len(y)) 
        else: 
            print('info> Applying leave-%d-out CV ...' % p)
            if p >= n_folds: print('selectCV> Warning: p: %d > n_folds: %d, why LeavePOut but not StratifiedKFold?' % (p, n_folds))
            nrow = len(y)
            n_folds = int(math.ceil(nrow/p)) 
            cv = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
            # cv = LeavePOut(n=X.shape[0], p=p)
    return cv

def drawROC(y_true, y_pred, msg='', **kargs):
    from sklearn.cross_validation import StratifiedKFold

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # clear previous plots
    plt.clf()
    plot_msg = msg  # text that serves as part of the plot description

    # output 
    ret = {}

    y_pred_prob = y_pred  # synonymous because y_pred has to be probablistic 

    # # [test]
    # if kargs.get('test_', False): 
    #     msg_ =  'drawROC-test> input dim > y_true (n=%d, t=%s) > example:\n%s\n' % (len(y_true), type(y_true), y_true[:10])
    #     msg_ += '                        > y_pred (n=%d, t=%s) > example:\n%s\n' % (len(y_pred), type(y_pred), y_pred[:10])   
    #     print(msg_); msg_ = ''

    n_folds = kargs.get('n_folds', 10)
    force_loo = kargs.get('force_loo', False)
    shuffle = False 

    if force_loo: assert n_folds == 1
    n_bootstraps_max = kargs.get('n_bootstraps_max', 1000)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    auc_scores = []
    summary_msg = kargs.get('summary_msg', '') # e.g. specifies the caller/evaluator

    n_samples = len(y_true)
    assert n_samples == len(y_pred)
    print('info> size(y_true): %d' % len(y_true))
    
    if n_folds == 1: 
        y_test = y_true
        print('info> input y_pred:\n%s\n------\n' % y_pred)
        print('info> using leave-one-out CV')

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        
        # bootstrap estimate
        roc_auc = auc(fpr, tpr)
        auc_scores = bootstrap_auc_score(y_test, y_pred, n_folds=1, message_=kargs.get('message_', 'n/a'))

        print('info> number of bootstrapped estimated auc scores: %d ~ %d' % (len(auc_scores), n_bootstraps_max))
    
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)
    else: 
        X_placeholder = np.zeros(n_samples)
        cvfs = StratifiedKFold(y_true, n_folds=n_folds, shuffle=shuffle) # selectCV(y_true, n_folds=n_folds, force_loo=force_loo)
        print('verify> final n_folds: %d' % len(cvfs)) 

        for i, (_, test) in enumerate(cvfs):
            
            # print('test> test indices:\n%s\n' % test[:100])
            y_true_i, y_pred_i = y_true[np.array(test)], y_pred[test]   
            
            fpr, tpr, thresholds = roc_curve(y_true_i, y_pred_i)
            roc_auc = auc(fpr, tpr)

            # bootstrap estimate
            sx = bootstrap_auc_score(y_true_i, y_pred_i, n_folds=(n_folds-1))
            # print('   + n_bootstraps_%d => %d' % (i, len(sx)))
            auc_scores.extend(sx)
            
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            
            if n_folds <= 5: 
                plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            else: 
                #print('verify> adding fold #%d' % (i+1))
                if i % 2 == 1: # only show even folds 
                    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % ( (i+1), roc_auc))

    n_bootstraps = len(auc_scores)
    if n_bootstraps > n_bootstraps_max: 
        auc_scores = random.sample(auc_scores, n_bootstraps_max)
    print('drawROC> (msg_id=%s), number of bootstrapped estimated auc scores: (%d >= %d) ~ %s' % \
        (msg, n_bootstraps, len(auc_scores), n_bootstraps_max))
            
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    if n_folds > 1: 
        plt.plot(mean_fpr, mean_tpr, 'k--',
                  label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if plot_msg: 
        plt.title('%s' % plot_msg)
    else: 
        plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # confidence interval
    cdict  = sampling.ci4(auc_scores, low=0.05, high=0.95) # mean=mean_auc
    ci_auc = (cdict['ci_low'], cdict['ci_high'])

    msg_ = '\nSummary> Given prediction results (type: %s)\n' % msg
    msg_ += "          + mean AUC: %f ~? bootstrap estimate: %f > CI: %s\n" % (mean_auc, cdict['mean'], str(ci_auc))
    msg_ += "          + sample AUCs:\n%s\n" % np.array(sampling.sorted_interval_sampling(auc_scores, 30)) # np.array has better format
    msg_ += "            ... report footnote: %s\n" % summary_msg
    div(message=msg_, symbol='*')

    # output: mean_auc, mean_auc_bootstrap, high_auc, low_auc
    ret['mean_auc'] = mean_auc
    ret['mean_auc_bootstrap'] = cdict['mean']
    ret['high_auc'] = cdict['ci_high']
    ret['low_auc'] = cdict['ci_low']

    save_fig = kargs.get('save_fig', True)
    if save_fig: 
        ext = 'pdf' # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
        fpath = kargs.get('opath', kargs.get('ofile', None))
        if fpath is None: 
            basedir = 'data-learner' # /phi/proj/poc7002/bulk_training/data-learner
            ext_plot = 'eps'  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
            output_dir = os.path.join(basedir, 'model_combined')
            fpath = os.path.join(output_dir, 'roc.%s' % ext)

        print('drawROC> saving performance (msg:%s) to %s\n' % (msg, fpath))

        # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
        plt.savefig(fpath, bbox_inches='tight')  
    else: 
        plt.show()  # matplotlib.use('Agg'): AGG backend is for writing to file, not for rendering in a window
    plt.close()

    return ret