# encoding: utf-8
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
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.dummy import DummyClassifier

# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.utils.extmath import density
from sklearn import metrics

# CV 
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

# from matplotlib import pyplot as plt

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
import seqparams
# from seqparams import TSet

import math
from sklearn.metrics import roc_curve, auc, roc_auc_score # performance metrics 

class TSet(seqparams.TSet):
    index_field = 'index'
    date_field = 'date'
    target_field = 'target'  # usually surrogate labels
    annotated_field = 'annotated'
    content_field = 'content'  # representative sequence elements 
    label_field = 'mlabels'  # multiple label repr of the underlying sequence (e.g. most frequent codes as constituent labels)

    meta_fields = [target_field, content_field, label_field, index_field, ]
    # index field: indexed into the original document set 

class Domain: 
    n_bootstraps_max = 1050
    # identifier = sys_config.read('identifier')

class Params(seqparams.ClassifierParams): 

    weight_sample = True

    @staticmethod
    def getClassifier(**kargs):
        """

        (*) LogisticRegression
            + http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
              C: Inverse of regularization strength; must be a positive float. Like in support vector machines, 
                 smaller values specify stronger regularization.
        (*) SVM 
            C: A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly 
            by giving the model freedom to select more samples as support vectors.
            gamma: the gamma parameter defines how far the influence of a single training example reaches, 
                   with low values meaning 'far' and high values meaning 'close'.

        Memo
        ----
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)

        """
        name = kargs.pop('name', 'logistic')
        penalty = kargs.pop('penalty', None)
        reg = kargs.get('reg', Params.c_scope)  # regularization strength default: Params.c_scope = np.logspace(-3, 3, 7)

        # global params 
        if not kargs.has_key('class_weight'): kargs['class_weight'] = 'balanced'

        # if c_scope is None: c_scope = Params.c_scope
        if name.startswith('logis') or isinstance(name, LogisticRegression): 
            if penalty is None: penalty = ['l1', 'l2']
            if isinstance(penalty, str): 
                assert penalty in ('l1', 'l2', )  # elasticnet? 
                penalty = [penalty]
            if reg is None: reg = Params.c_scope
            penalties = kargs.get('penalties', penalty)
            params = [{'C': reg, 'penalty': penalties}] # np.logspace(-1, 2, 4)
            
            tol = kargs.pop('tol', 0.01)
            cw = kargs.get('class_weight', 'balanced')
            print('info> class weight: %s' % cw)
            penalty_ = penalty[0]

            if penalty_ != 'l1': 
                if not kargs.has_key('solver'): 
                    kargs['solver'] = 'sag'
            
            kargs['penalty'] = penalty_
            # else: 
            #     kargs.pop('penalty', None) # only for setting params

            clf = LogisticRegression(tol=tol, **kargs) # set other params later
            if kargs.get('verbose', False): 
                print('getClassifier> %s' % clf)

        elif name in ('svm', 'svc', ) or isinstance(name, SVC):  # test for all kernels
            # if not kargs.has_key('probability'): kargs['probability']=True
            clf = SVC(probability=True, class_weight='balanced')
            
            # only use this option to choose kernel, don't care about an optimal regularization strength
            if reg is None: reg = np.logspace(-4, 4, 6) # Params.c_scope  # np.logspace(-3, 3, 7)

            # configure paramemter grid over which optimal hyperparameters are selected
            gammax = kargs.get('gammax', np.logspace(-9, 5, 15))  # gamma: inverse of bandwidth
            degreex = kargs.get('degreex', range(2, 10))
            params = [{'kernel': ['rbf', ], 'gamma': gammax, 'C': reg},
                      {'kernel': ['linear', ], 'C': reg}, 
                      {'kernel': ['poly'], 'degree': degreex, 'C': reg}, 
                    ]
        elif name in ('linear_svm', ):
            # if not kargs.has_key('probability'): kargs['probability']=True
            clf = SVC(kernel='linear', probability=True, class_weight='balanced')
            if reg is None: reg = Params.c_scope
            params = [{'kernel': ['linear'], 'C': reg}, ]
        elif name in ('rbf_svm', ): # squared expoential kernel 
            # if not kargs.has_key('probability'): kargs['probability']=True

            # gamma: 'auto' <- 1/n_features
            gammax = kargs.get('gammax', np.logspace(-9, 5, 15))  # np.logspace(-9, 3, 13)
            clf = SVC(kernel='rbf', gamma='auto', probability=True, class_weight='balanced')
            if reg is None: reg = Params.c_scope
            # gamma: 'auto' <- 1/n_features

            # for model selection
            params = [{'kernel': ['rbf', ], 'gamma': gammax, 'C': reg},]
        elif name in ('poly_svm', ):
            # if not kargs.has_key('probability'): kargs['probability']=True
            clf = SVC(kernel='poly', degree=3, probability=True, class_weight='balanced')
            if reg is None: reg = Params.c_scope
            degreex = kargs.get('degreex', ange(2, 10))
            params = [{'kernel': ['poly', ], 'degree': degreex, 'C': reg}, ]

        elif name.startswith('lasso') or isinstance(name, LassoLogistic): 
            clf = LassoLogistic(alpha=1.0, **kargs)
            if reg is None: reg = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            params = [{'alpha': reg, }]

        elif name.startswith('near') or isinstance(name, NearestCentroid):
            pass 

        return (clf, params)

### end class Params

def select_kernel_svm(ts, **kargs):  # used for SVM and other classifiers with extra hyperparams other than regularization
    """

    Memo
    ----
    1. At level 0.5 (eature set of 2225-D by combining those from all models), 
       the best parameters are {'kernel': 'rbf', 'C': 100.0, 'gamma': 10.0} with a score of 0.69

    2. C: A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly 
            by giving the model freedom to select more samples as support vectors.

    3. model evaluation 
       see http://scikit-learn.org/stable/modules/model_evaluation.html
    """
    def profile(): 
        print('config> classifier: %s' % str(clf_.get_params()))
        print('config> CV:  %s' % str(clf.get_params()))

        return

    # from sklearn.cross_validation import StratifiedShuffleSplit

    level = 0
    n_folds_ms = 5 
    clf_type = kargs.get('clf_type', kargs.get('name', 'linear_svm'))
    metric = 'roc_auc' 

    clf_ = None
    if clf_type == 'all':  # ... select from a mixture of pre-specified kernels, see Params.getClassifier ... expensive!
        clf_, params = Params.getClassifier(name='svm')
        # result: {'kernel': 'rbf', 'C': 100.0, 'gamma': 10.0}

    else: # ... use specific SVM kernel (e.g. linear_svm, poly_svm, rbf_svm, all)
        # clf, params = Params.getClassifier(name=clf_type, reg=np.logspace(-4, 4, 9)) # default c_scope: np.logspace(-3, 3, 7)

        params = tuned_parameters = [{'kernel': ['linear'], 'C': np.logspace(-4, 4, 9)}]
        clf_ = SVC(C=1, kernel='linear', probability=True) # probability=True, class_weight='balanced'
        # {'kernel': 'linear', 'C': 0.0001} with a score of 0.39 

    div(message='hyperparams> model params (svm type: %s) > clf: %s, params: %s' % (clf_type, clf_, params), symbol='#')

    X, y = transform(ts, standardize_='minmax')  # defined in evaluate
    n_features = X.shape[1]-1
    print('proposal> feature set has %d-D' % n_features)
    
    # cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=10)
    clf = GridSearchCV(clf_, param_grid=params, cv=n_folds_ms, scoring=metric)
    clf.fit(X, y)

    profile() 

    print("info> The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
    best_params, best_score = clf.best_params_, clf.best_score_

    div(message="Detailed grid scores", symbol='#')
    print('\n')
    means = clf.cv_results_['mean_test_score']  # this attribute doesn't come with clf_ (without the CV wrapper)
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('\n')

    # best_params: e.g. {'kernel': 'rbf', 'C': 100.0, 'gamma': 10.0}  
    div(message='select_kernel_svm > best params (score: %f):\n%s\n' % \
        (best_score, best_params), symbol='#')  

    clf_.set_params(**best_params)

    return (clf_, best_params) # [todo] best_params is redundant!

def select_features(X, y, **kargs):  # LassoCV by default
    def encode_labels(y): 
        # check if y has already been binarized? 
        # is_binarized = hasattr(y[0], '__iter__') and set(class_labels).issubset([0, 1])
        class_labels = np.unique(y)
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        lb.fit(class_labels)
        label_pairs = zip(lb.transform(class_labels), class_labels)
        print('label_encoder> labels vs numeric labels ...')
        for l, cl in label_pairs: 
            print('  + %s ~> %s' % (l, cl))
        lookup = dict([(tuple(l), cl) for l, cl in label_pairs])
        return (lb, lookup)
    def encode_labels2(y):
        class_labels = sorted(np.unique(y))
        
        lookup = {}
        for i, label in enumerate(class_labels):
            lookup[label] = i 

        yp = []
        for ye in y: 
            yp.append(lookup[ye])
        return (np.array(yp), lookup)  # lookup: label -> number

    def dummy_feature_set(prefix='f'): 
        ncol = X.shape[1]
        return ['%s_%s' % (prefix, i) for i in range(ncol)]
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s (sampled)-> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def subsample2(ts, n=None, sort_index=True, random_state=53):
        if n is not None: # 0 or None => noop
            ts = cutils.sampleDataframe(ts, col=TSet.target_field, n_per_class=n, random_state=random_state)
            n_classes_prime = check_tset(ts)
            print('  + after subsampling, size(ts)=%d, n_classes=%d (same?)' % (ts.shape[0], n_classes_prime))
        else: 
            # noop 
            pass 
        return ts

    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV, RandomizedLogisticRegression
    # from sklearn.ensemble import RandomForestClassifier
    from sklearn import preprocessing
    import classifier.utils as cutils

    max_iter = kargs.get('max_iter', 1500)  # only used for LassoCV
    threshold = kargs.get('threshold', None)

    n_iter = kargs.get('n_iter', 1)  # number of iterations
    n_features = kargs.get('n_features', None)
    if n_features is not None and n_iter == 1: 
        n_iter = 10  # default number of iterations when n_features is given

    fset = kargs.get('feature_set', dummy_feature_set())
    y, lookup = encode_labels2(y)

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = kargs.get('estimator', None)
    if clf is None: 
        clf = RandomizedLogisticRegression() # options: LassoCV(max_iter=max_iter) # 1000 by default 
    else: 
        pass
    print('select_features> n_features: %d (total: %d), n_iter: %d | n_classes: %d, estimator:%s' % \
        (n_features, len(fset), n_iter, len(lookup), clf))

    fcount = collections.Counter()
    fset_active = fset
    indices = range(X.shape[1])
    if n_iter == 1: # [todo] no control over the number of features obtained
    
        # Set a minimum threshold of 1e-5 via l1
        sfm = SelectFromModel(clf, threshold=threshold)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]
        print('info> number of features (via l1): %d' % n_features)

        # [test]
        Xp = X[:,sfm.get_support()]
        print('info> Xp dim: %s ~? %d' % (str(Xp.shape), n_features))

        # feature selected 
        fset_active = fset[sfm.get_support()]  # feature is of string type
        print('info> n_fset_active: %d =?= %d > example:\n%s\n' % \
           (len(fset_active), Xp.shape[1], random.sample(fset_active, min(n_features, 10))) )
        
        indices = sfm.get_support(indices=True)
    else: 
        n_classes = len(lookup)
        maxNSamplePerClass = kargs.get('n_per_class', max(X.shape[0]/n_classes/3, 10))

        if kargs.get('test_importance', False):
            Xs, ys = subsample(X, y, n=maxNSamplePerClass, sort_index=False, random_state=53) 
            clf.fit(Xs, ys)
            try: 
                print('    + feature importance: %s' % np.mean(clf.feature_importances_))
            except: 
                print('    + clf: %s does not have feature_importances_' % clf.__class__.__name__)

        n = 0
        while n <= n_iter: 
            sfm = SelectFromModel(clf, threshold=threshold)

            # set maxNPerClass to a moderate size to reduce overhead
            Xs, ys = subsample(X, y, n=maxNSamplePerClass, sort_index=False, random_state=53)
            print('... iteration #%d | subsampling? n=%s' % (n+1, maxNSamplePerClass))

            sfm.fit(Xs, ys)

            n_selected = sfm.transform(Xs).shape[1]
            print('info> number of features selected: %d' % n_selected)  

            fset_active = fset[sfm.get_support()]  # feature is of string type
            fset_indices = sfm.get_support(indices=True)

            # fcount.update(fset_active)
            fcount.update(fset_indices)  # keep track of feature indices
            n +=1  
        
        fset_active_index_cnt = fcount.most_common(n_features)  # each iteration looks at a different data subset
        # print('select_features> select %d features out of %d' % (n_features, len(fcount)))

        indices = [f[0] for f in fset_active_index_cnt]
        fset_active = fset[indices]

        # fset_active_cnt = fcount.most_common(n_features)
        # fset_active = [f[0] for f in fset_active_cnt] 
    return (fset_active, indices) 

def select_features_lasso(ts, **kargs):
    from sklearn.feature_selection import SelectFromModel
    # from sklearn.linear_model import LassoCV
    import collections

    outputdir = basedir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir

    # [params] cohort and table
    # cohort_name = 'PTSD'
    # ctrl_cohort_name = '%s-Negative' % cohort_name
    # table_name = 'measurement'

    meta_fields = kargs.get('meta_fields', TSet.meta_fields)
    max_iter = kargs.get('max_iter', 3000)

    n_iter = kargs.get('n_iter', 1)  # number of iterations
    n_features = kargs.get('n_features', None)
    if n_features is not None and n_iter == 1: 
        n_iter = 10  

    ts = ts.reindex(np.random.permutation(ts.index))
    fset = get_feature_set(ts, meta_fields=meta_fields)

    print('io> Given tset of dim: %s, n_fset: %d' % (str(ts.shape), len(fset))) # [log] tset of dim: (12444, 2207)
    X, y = transform(ts, standardize_='minmax')

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV(max_iter=max_iter) # 1000 by default
    fcount = collections.Counter()
    fset_active = fset
    indices = range(X.shape[1])
    if n_iter == 1: # no control over the number of features obtained
    
        # Set a minimum threshold of 1e-5 via l1
        sfm = SelectFromModel(clf, threshold=None)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]
        print('info> number of features (via l1): %d' % n_features)

        # [test]
        Xp = X[:,sfm.get_support()]
        print('info> Xp dim: %s ~? %d' % (str(Xp.shape), n_features))

        # feature selected 
        fset_active = fset[sfm.get_support()]  # feature is of string type
        print('info> n_fset_active: %d =?= %d > example:\n%s\n' % \
           (len(fset_active), Xp.shape[1], random.sample(fset_active, min(n_features, 10))) )
        
        indices = sfm.get_support(indices=True)
    else: 
        
        n = n_iter
        while n: 
            sfm = SelectFromModel(clf, threshold=None)
            sfm.fit(X, y)
            n_features = sfm.transform(X).shape[1]
            print('info> number of features (via l1): %d' % n_features)

            fset_active = fset[sfm.get_support()]  # feature is of string type
            fset_indices = sfm.get_support(indices=True)

            # fcount.update(fset_active)
            fcount.update(fset_indices)  # keep track of feature indices
            n -= 1 
        
        fset_active_index_cnt = fcount.most_common(n_features)  # positions
        indices = [f[0] for f in fset_active_index_cnt]
        fset_active = fset[indices]

        # fset_active_cnt = fcount.most_common(n_features)
        # fset_active = [f[0] for f in fset_active_cnt] 
    return (fset_active, indices)

def select_model(ts, **kargs): 
    def profile(): 
        print('config> classifier: %s' % str(clf_.get_params()))
        print('config> CV:  %s' % str(clf.get_params()))

        return

    # from sklearn.cross_validation import StratifiedShuffleSplit

    level = 0  # not normally used
    n_folds_ms = 5 
    clf_type = kargs.get('clf_type', kargs.get('name', 'logistic'))
    metric = 'roc_auc' 

    clf_ = params = None
    if clf_type.startswith('log'):
        clf_, params = Params.getClassifier(name='logistic', penalty='l2')
    else: 
        raise NotImplementedError 

    div(message='params> model params (svm type: %s) > clf: %s, params: %s' % (clf_type, clf_, params), symbol='#')

    X, y = transform(ts, standardize_='minmax')  # defined in evaluate
    print('proposal> Feature set dimension: %s' % str(X.shape))
    
    # cv = StratifiedShuffleSplit(y, n_iter=5, ltest_size=0.2, random_state=10)
    clf = GridSearchCV(clf_, param_grid=params, cv=n_folds_ms, scoring=metric)
    clf.fit(X, y)

    profile() 

    print("info> The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
    best_params, best_score = clf.best_params_, clf.best_score_

    div(message="Detailed grid scores", symbol='#')
    print('\n')
    means = clf.cv_results_['mean_test_score']  # this attribute doesn't come with clf_ (without the CV wrapper)
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('\n')

    # best_params: e.g. {'kernel': 'rbf', 'C': 100.0, 'gamma': 10.0}  
    div(message='select_model> best params (score: %f):\n%s\n' % \
        (best_score, best_params), symbol='#')  

    clf_.set_params(**best_params)

    return (clf_, best_params) # [todo] best_params is redundant!

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

    rng_seed = 10  # control reproducibility
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


def specificity_loss_func(ground_truth, predictions, verbose=False):
    if verbose: 
        print 'info> predictions:\n%s' % predictions
        print 'info> truth:      \n%s\n' % ground_truth
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    for l,m in enumerate(ground_truth):        
        if m==predictions[l] and m==1:
            tp+=1
        if m==predictions[l] and m==0:
            tn+=1
        if m!=predictions[l] and m==1:
            fn+=1
        if m!=predictions[l] and m==0:
            fp+=1
    n = tn + fp 
    if n == 0: 
        print('specificity> n=(tn+fp)=0')
        return 0.0
    r = tn/n
    print('specificity> tn/(tn+fp): %d/%d = %f' % (tn, n, r))
    return r

def sensitivity_loss_func(ground_truth, predictions, verbose=False):  # recall
    if verbose: 
        print 'info> predictions:\n%s' % predictions
        print 'info> truth:      \n%s\n' % ground_truth
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    for l,m in enumerate(ground_truth):        
        if m==predictions[l] and m==1:
            tp+=1
        if m==predictions[l] and m==0:
            tn+=1
        if m!=predictions[l] and m==1:
            fn+=1
        if m!=predictions[l] and m==0:
            fp+=1
    
    p = tp + fn
    if p == 0: 
        print('sensitivity> n=(tp+fn)=0')
        return 0.0 
    r = tp/p
    print('sensitivity> tp/(tp+fn): %d/%d = %f' % (tp, p, r))
    return tp/p

def standardize(X, y=None, method='minmax'): 
    """

    Note
    ----
    1. transformer can be used to transform new data
    2. use sklearn.decomposition.PCA or sklearn.decomposition.RandomizedPCA with whiten=True 
       to further remove the linear correlation across features.

    # normalization 
    Normalization is the process of scaling individual samples to have unit norm. 
    This process can be useful if you plan to use a quadratic form such as the dot-product 
    or any other kernel to quantify the similarity of any pair of samples.

    # Gaussian variable 

    # MinMax scaling

    """
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.preprocessing import Normalizer
   
    transformer = None
    if method.startswith('stand'):
        # div(message='Standardizing X via centering and scaling ...')
        # X = preprocessing.scale(X)   # assuming approx. gaussian (mean: 0, std: 1)
        transformer = scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif method.startswith('norm'):   # so that the length of feature vector is 1
        # div(message='Standardizing X via normalizing ...')
        transformer = normlizer = Normalizer()
        X = normalizer.fit_transform(X)
    elif method.startswith('m'):  # minmax
        # div(message="Standardizing X via min-max scaler ...")
        transformer = minmax = MinMaxScaler()  # default range (0,1)
        X = minmax.fit_transform(X)
    # [todo] dimensionwise standardization

    return (X, y, transformer)  # [1]


def selectCV(y, n_folds=5, force_loo=False, p=1, shuffle=False):
    # from sklearn.cross_validation import StratifiedKFold, LeaveOneOut  # need cv to take on an iterator of indices

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

# too specific
def profile(ts): 
    def survey(p=2):
        """
        p: p-norm

        """

        # this includes pseduo annotations
        stats = {}

        stats['dimension'] = stats['dim'] = ts.shape
        stats['nrow'] = ts.shape[0]
        stats['ncol'] = ts.shape[1]

        # cannot use n_pos_anno since we need to modify it; in Python 3, can do nonlocal
        stats['n_type1'] = ts.loc[ts[f_target]==0].shape[0]
        stats['n_type2'] = ts.loc[ts[f_target]==1].shape[0]
        stats['n_type3'] = ts.loc[ts[f_target]==2].shape[0]

        nonzeros = []
        norms = []
        for i, row in ts.iterrows(): 
            nonzeros.append(np.count_nonzero(row))
            norms.append(np.linalg.norm(row, p))
        assert max(nonzeros) <= ts.shape[1]

        stats['nonzeros'] = nonzeros

        stats['norms'] = norms
        # [log] all seems to be normalized to 1

        return stats

    def summary(): 
        dim_cutoff = 50

        div(message='Summary of Model-Combined Training Set', symbol='#')
        print('1. Dimension: %s' % str(stats['dimension']))
        print('2. Sample Size: ')
        print('   + size of type-1:      %d' % stats['n_type1'])
        print('   + size of type-2:      %d' % stats['n_type2'])
        print('   + size of gestational: %d' % stats['n_type3'])

        print('3. Data distribution: ')
        print('   + example nonzero counts (up to %d-th dim):\n%s\n' % (dim_cutoff, str(stats['nonzeros'][:dim_cutoff])))
        print('   + example norms:\n%s\n' % str(stats['norms'][:dim_cutoff])) 

        return
    
    f_target = TSet.target_field
    labels = list(set(ts[f_target].values))

    tp_identifier = 'condition_drug'
    ts_stem = 'tset'
    ts_ext = 'csv' 

    # t_format = "%Y-%m-%d" 

    # [params] tset 
    ts_sep = ','

    # annotated sample 
    f_index = 'index'
    f_target = TSet.target_field
    f_annotated = TSet.annotated_field
    f_content = TSet.content_field
    f_label = TSet.label_field

    # assert not (tsr.empty and tsv.empty)
    stats = survey(p=2)

    summary()

    return stats

def get_feature_set(ts, meta_fields=None): 
    if meta_fields is None: meta_fields = TSet.meta_fields

    return ts.columns.drop([f for f in meta_fields if f in ts.columns])

def transform(ts, f_target=None, **kargs): 
    def drop_meta(): 
        return ts.columns.drop([f for f in meta_fields if f in ts.columns])

    # [params]
    # f_meta = ['target', 'content', 'mlabels', ]  # content: representative codes, mlabels: multilabels 
    
    if f_target is None: f_target = TSet.target_field # predicting annotated field
    meta_fields = kargs.get('meta_fields', TSet.meta_fields)  # meta fields should include target field

    fX, fy = drop_meta(), f_target  # get features associated with the given label; group only used at level0
    # print('transform> n_total: %d <- n_features: %d + n_meta: %d' % (ts.shape[1], len(fX), len(Meta.f_meta_l0)))
     
    # dim0 = ts.shape
    # ts = ts.dropna(subset=fX, how='all')
    X = ts.as_matrix(columns=fX)
    
    try: 
        y = ts[fy].values
    except: 
        print('Warning: No target field in the dataframe.')
        y = None
        
    method_ = kargs.get('std_method', 'minmax')
    if isinstance(method_, bool): 
        if method_ is False: 
            method_ = None 
        else: 
            method_ = 'minmax'

    if method_ is not None: 
        X, y, transformer = standardize(X, y=y, method=method_)

    # print('test> X:\n%s' % X[-1])
    # print('test> y:\n%s' % y[-1])
    return (X, y)

# evaluator for a binary classifier
def evaluateTrainTestSplit(X_train, X_test, y_train, y_test, **kargs): 
    """

    Input
    -----
    (X_train, X_test, y_train, y_test)

    Params
    ----- 
    * file naming 
        identifier
    * output 
        outputdir

    * classifier/estimator 
        estimator
        sample_weight


    """
    def to_labels(y_prob, threshold=0.5): # convert y_prob to predicted class labels
        y_labels = []
        for i, e in enumerate(y_prob): 
            if e >= threshold: 
                y_labels.append(1)
            else: 
                y_labels.append(0)
        return np.array(y_labels)
    def separate_polarity(tslocal): 
        idx_positive, idx_negative = set(), set()
        tsp = tslocal.loc[tslocal[TSet.target_field]==1]
        idx_positive.update(tsp.index)
        tsn = tslocal.loc[tslocal[TSet.target_field]==0]
        idx_negative.update(tsn.index)
        return (idx_positive, idx_negative)
    
    from tset import TSet  # training data abstraction 
    random.seed(10)
    force_loo = False
    n_folds_ms = 5  # k-fold CV for model selection
    n_folds = 10 # k-fold CV for training
    is_cv_est = False  # model selection  (is the estimator/classifier wrapped in a CV class such as GridSearchCV?)
    use_nested_cv = kargs.get('use_nested_cv', True) # force model seletion within CV; this depends on params, which will subsume this value    

    # file identifier 
    identifier = kargs.get('identifier', 'default')
    target_field = TSet.target_field   # 'target'

    # classifier 
    classifier = kargs.get('estimator', None)
    if classifier is None: 
        print('warning> classifier not specified ... use logistic regression by default ...')
        classifier = LogisticRegression(penalty='l2', class_weight='balanced', solver='sag') # sag works for large dataset, default: liblinea
    sample_weight = kargs.get('sample_weight', None)

    # OUTPUT file naming 
    ext_plot = 'tif'  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    fp_roc = os.path.join(output_dir, 'roc-P%s.%s' % (identifier, ext_plot))


    return

def classification_report(y_true, y_pred, **kargs): 
    def to_labels(y_prob, threshold=0.5): # convert y_prob to predicted class labels
        y_labels = []
        for i, e in enumerate(y_prob): 
            if e >= threshold: 
                y_labels.append(1)
            else: 
                y_labels.append(0)
        return np.array(y_labels)

    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc
    
    # classification report, confusion matrix 
    yt, yp = y_true, y_pred
    print('eval> making classification report ...')
    p_threshold = kargs.get('threshold', 0.5) # probability threshold

    # [output]
    output_dir = kargs.get('outputdir', os.getcwd())
    identifier = kargs.get('identifier', 'default')
    fp_cls_report = os.path.join(output_dir, 'classification_report-%s.dat' % identifier)
    fp_confusion_matrix = os.path.join(output_dir, 'confusion_matrix-%s.dat' % identifier)
        
    try: 
        output = metrics.classification_report(yt, yp) # note: yp must be label prediction
    except Exception, e: 
        print('warning> input y_predx are probabilities? need label prediction here): %s (err: %s)' % (yp[:5], e))
        try: 
            yp = to_labels(yp, threshold=p_threshold)
            output = metrics.classification_report(yt, yp) # note: yp must be label prediction
        except Exception, e: 
            raise ValueError, e 

    with open(fp_cls_report, "w") as text_file:
        print('info> level=%s > writing classification report (meta=%s) to %s' % (level, meta, fp_cls_report))
        text_file.write(str(output))
        text_file.write('\n')

    output = metrics.confusion_matrix(yt, yp)
    with open(fp_confusion_matrix, "w") as text_file:
        print('info> level=%s > writing classification report (meta=%s) to %s' % (level, meta, fp_confusion_matrix))
        text_file.write(str(output))
        text_file.write('\n')

    # compute sensitivity and specificity
    div(message='classification_report(ID: %s) Sensitivity and Specificity' % identifier)
    score_sen = sensitivity_loss_func(yt, yp)
    score_spe = specificity_loss_func(yt, yp)
    print('metric> sensitivity: %f, specificity: %f\n' % (score_sen, score_spe))

    return (score_sen, score_spe)

# evaluator for a binary classifier
def evaluate(ts, classifier, params=None, ts_test=None, **kargs): 
    """
    Params
    ------


    """
    def to_labels(y_prob, threshold=0.5): # convert y_prob to predicted class labels
        y_labels = []
        for i, e in enumerate(y_prob): 
            if e >= threshold: 
                y_labels.append(1)
            else: 
                y_labels.append(0)
        return np.array(y_labels)

    def separate_polarity(tslocal): 
        idx_positive, idx_negative = set(), set()
        tsp = tslocal.loc[tslocal[TSet.target_field]==1]
        idx_positive.update(tsp.index)
        tsn = tslocal.loc[tslocal[TSet.target_field]==0]
        idx_negative.update(tsn.index)

        return (idx_positive, idx_negative)

    def pred_breakdown(ytx, ypx, ts_ref=None, policy='default'):
        n = len(ytx)
        assert n == len(ypx)
        n_correct = sum(1 for i, yt in enumerate(ytx) if ypx[i] == yt)
        ratio = n_correct/(n+0.0)
        fraction = '%d/%d' % (n_correct, n)
        return (fraction, ratio)
              
    random.seed(10)

    # [params] d2v identifiers
    seq_ptype = seqparams.normalize_ptype(**kargs) # regular, random, diag (only), med (only), etc.
    d2v_method = kargs.get('d2v_type', kargs.get('d2v_method', 'd2v_default') ) 

    # [params]
    n_cv_cycles = kargs.get('n_cv_cycles', 10) 
    is_multirun = False
    force_loo = False
    n_folds_ms = 5  # k-fold CV for model selection
    n_folds = 10 # k-fold CV for training
    metric = kargs.get('metric', 'roc_auc')
    is_cv_est = False  # model selection 
    use_nested_cv = kargs.get('use_nested_cv', True) # force model seletion within CV; this depends on params, which will subsume this value

    # experimental settings
    setting = kargs.get('setting_', 'single model')
    meta = kargs.get('meta_', '')  # extra file delineator e.g. classifier type
    level = 0 # not normally in use

    # plot
    footnote = 'sequence pattern type: %s, d2v method: %s' % (seq_ptype, d2v_method)
    note = kargs.get('note', '')
    if not note: 
        note = '%s_%s' % (seq_ptype, d2v_method)  # [I/O] Output file identifier
     
    # experimental file identifier
    identifier = '%s_%s' % (seq_ptype, d2v_method)

    # [params]
    # f_index = 'index'
    f_target = 'target'

    # output 
    ret = {}

    # classifier 
    if classifier is None: 
        print('warning> classifier not specified ... use logistic regression by default ...')
        classifier = LogisticRegression(penalty='l2', class_weight='balanced', solver='sag') # sag works for large dataset, default: liblinear
        params = None 
    sample_weight = kargs.get('sample_weight', None)

    p_threshold = 0.5 

    # CV 
    shuffle = False

    # [params] output
    basedir = output_dir = os.path.join(os.getcwd(), 'plot')  # or sys_config.read('DataExpRoot')
    ext_plot = 'tif'  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    fp_cls_report = os.path.join(output_dir, 
        'classification_report-P%s.dat' % identifier if not meta else 'classification_report-%s-P%s.dat' % (meta, identifier))
    fp_confusion_matrix = os.path.join(output_dir, 
        'confusion_matrix-P%s.dat' % identifier if not meta else 'confusion_matrix-%s-P%s.dat' % (meta, identifier))
    fp_roc = os.path.join(output_dir, 
        'roc-P%s.%s' % (identifier, ext_plot) if not meta else 'roc-%s-P%s.%s' % (meta, identifier, ext_plot))

    # for the purpose of only evaluting true annotations, need to use these to readjust train and test splits
    idx_regular = set() 
    idx_virtual = set()

    # for computing sensitivity, etc.
    idx_positive = set()  
    idx_negative = set()

    # charaterize the training data 
    tsr = yr = None
    
    stats = profile(ts)
    n_type1 = n_neg = stats['n_type1']
    n_type2 = n_pos = stats['n_type2']
    n_type3 = n_ges = stats['n_type3'] # binary classifier doesn't consider this
    n_regular = n_pos + n_neg
    n_virtual = 0 

    if n_ges > 0: 
        print('evaluate> binary classifier should not have the 3rd class but got %d instances' % n_type3)

    if kargs.get('shuffle_', True): 
        ts = ts.reindex(np.random.permutation(ts.index))  # shuffle data before each CV-cycle begins
        
    nrow0 = ts.shape[0]
    tsr = ts.loc[ts[TSet.target_field].isin([0, 1, ])]; 
    yr = tsr[TSet.target_field].values  # predict targets
    nrow = tsr.shape[0]
    print('verify> nrow0: %d =?= nrow for label 0&1 only: %d' % (nrow0, nrow))

    if params is not None: 
        if isinstance(params, list) or isinstance(params, tuple): 
            use_nested_cv = True  # embed model selection within CV
        elif isinstance(params, dict): 
            use_nested_cv = False  # model selection has been done prior to this call
        else: 
            raise ValueError, "Invalid params:\n%s\n" % str(params)

    # [note] 4 scenarios for the input classifier
    #   a. it is already an GridSearchCV object [todo] any CV-wrapped object 
    #   b. it is a pure classifier with params specifiying the search space
    #   c. it is a pure classifier that was already being optimized i.e. Params is just a dict (but NOT a list of dicts)
    #   d. if params is None and classifier is not already CV-wrapped => use default use_nested_cv
    print('verify> use nested CV? %s | n_folds_model: %d, n_folds_training: %d, CV cycles: %d' % \
        (use_nested_cv, n_folds_ms, n_folds, n_cv_cycles))

    # model selection 
    estimator = classifier
    if not isinstance(classifier, GridSearchCV):    
        # model selection in this case will be performed within each CV fold, expensive! 
        # 1. optimial hyperparams vary within each fold (due to slightly different training data)
        # 2. within each fold (of training data), need a separate set of data to evaluate the hyperparam configuration => nested CV
        if use_nested_cv: 
            estimator = GridSearchCV(classifier, params, cv=n_folds_ms, scoring=metric)  # use k-fold to estimate model params (e.g. C)
            is_cv_est = True
        else: 
            # model selection is assumed to have been done prior to this call.
            print("info> using plain CV, assuming that the input classifier's parameters were optimized > params:\n%s\n" % classifier.get_params())
    else: # already wrapped 
        print("info> Input classifier is a GridSearchCV object => using nested CV!")

    # desired label depends on the training and evaluation: build model
    # tsp = ts
    # X, y = transform(ts)  # this ts has a mixture of both real and virtual annotated samples
    # idx_positive, idx_negative = separate_polarity(ts)

    # global values wrt overall CV-cycles
    y_truex, y_predx, y_prob_predx = [], [], []  # flat overall
    yt_cycles, yl_cycles, yp_cycles = [], [], [] # flat within 1 cycle        

    if ts_test is not None: 
        raise NotImplementedError 
    else: 
        # n_folds_eff = n_folds_ms if use_nested_cv else n_folds
        cvfs = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=shuffle) # selectCV(ts[Meta.f_annotated], n_folds=n_folds, force_loo=force_loo)
        
        header = ['index', 'y_true', 'y_prob', 'y_pred']

        for i in range(n_cv_cycles): 
            print('progress> CV cycle #%d' % (i+1))
            adict, subdict = {}, {}

            # overwrite global values
            tsp = ts.reindex(np.random.permutation(ts.index))  # shuffle data before each CV-cycle begins
            X, y = transform(tsp) 
            
            indices = []
            yt_cycle, yl_cycle, yp_cycle = [], [], []  # y_true, y_pred, y_prob ~ true labels, label predictions, prob predictions 

            for j, (train, test) in enumerate(cvfs.split(X, y)):  # train: row indices
    
                X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]  # training on new indices

                # if do_pcalib: estimator = CalibratedClassifierCV(estimator, cv=n_folds_calib, method='isotonic')

                # prediction
                if sample_weight is None: 
                    estimator.fit(X_train, y_train)
                else: 
                    estimator.fit(X_train, y_train, sample_weight=sample_weight[train])

                if use_nested_cv: 
                    print('progress> (CV cycle #%d, %d-th fold) > best params: %s | best_score: %f' % \
                        (i, j, estimator.best_params_, estimator.best_score_))

                # need indices to be in order 
                ts_cv_test = tsp.loc[test]
                y_truex.extend(y_test)  # global across all CV cycles
                yt_cycle.extend(y_test) # local to each CV cycle; eventuall go to yt_cycles

                # probability prediciton
                probas_ = estimator.predict_proba(X_test)
                y_prob_pred = probas_[:, 1] # p(y=1|x)
                y_prob_predx.extend(y_prob_pred) # flat wrt both CV folds and cycles
                yp_cycle.extend(y_prob_pred)  # yp resets for each cycle, n cycles => n lists
        
                # label prediction
                y_pred = estimator.predict(X_test)
                y_predx.extend(y_pred)
                yl_cycle.extend(y_pred)   

            ### end CV loop
            yt_cycles.append(yt_cycle) # true labels 
            yl_cycles.append(yl_cycle) # label predictions
            yp_cycles.append(yp_cycle) # prob predictions

        ### end all cycles

    n_cv_examples = n_cv_cycles * n_regular
    print('eval> number of instances across all cycles: %d' % n_cv_examples)
    # assert len(yt_fold) == n_samples, "number of total iterations in repeated CV: %d but got %d" % (len(yt_fold), n_samples)
    # assert len(yt_cycles) == n_cv_cycles
    # assert len(yt_cycles[0]) == n_regular, "Got %d examples per cycle but expected %d" % (len(yt_cycles[0]), n_regular)

    # classification report, confusion matrix 
    for i, (yt, yp) in enumerate([(y_truex, y_predx), ]): # don't use y_prob_pred
        
        print('eval> making classification report #%d (level: %s,  setting: %s) ...' % (i, level, setting))
        try: 
            output = metrics.classification_report(yt, yp) # note: yp must be label prediction
        except Exception, e: 
            print('warning> input y_predx are probabilities? need label prediction here): %s (err: %s)' % (yp[:5], e))
            try: 
                yp = to_labels(yp, threshold=p_threshold)
                output = metrics.classification_report(yt, yp) # note: yp must be label prediction
            except Exception, e: 
                print('warning> level=%s, inconsistent y_true and y_pred (type:%s)? > +y_true:\n%s\n+y_pred:\n%s\n' % (level, tmap[i], yt, yp))
                raise ValueError, e 

        with open(fp_cls_report, "w") as text_file:
            print('info> level=%s > writing classification report (meta=%s) to %s' % (level, meta, fp_cls_report))
            text_file.write(str(output))
            text_file.write('\n')

        output = metrics.confusion_matrix(yt, yp)
        with open(fp_confusion_matrix, "w") as text_file:
            print('info> level=%s > writing classification report (meta=%s) to %s' % (level, meta, fp_confusion_matrix))
            text_file.write(str(output))
            text_file.write('\n')

    # compute sensitivity and specificity
    div(message='result> (ID: %s) Sensitivity and Specificity in Repeated CV for %d Cycles' % (identifier, n_cv_cycles))
    score_sen = sensitivity_loss_func(y_truex, y_predx)
    score_spe = specificity_loss_func(y_truex, y_predx)
    print('metric> sensitivity: %f, specificity: %f' % (score_sen, score_spe))
    print('\n')

    # ROC 
    div(message='ROC', symbol='#')
    print('info> size of y_truex: %d' % len(y_truex))

    scores, mean_scores, highs, lows = [], [], [], []
    for i in range(n_cv_cycles):
        save_fig = True if i % 3 == 0 else False 
        fp_roc = os.path.join(output_dir, 
            'roc-%s-P%s.%s' % (i, seq_ptype, ext_plot) if not meta else 'roc_%s-%s-P%s.%s' % (meta, i, identifier, ext_plot)) # overwrite global 
        ret = drawROC(yt_cycles[i], yp_cycles[i], identifier=note, opath=fp_roc, summary_msg=footnote, n_folds=n_folds, save_fig=save_fig)

        scores.append(ret['mean_auc_bootstrap']) # or ret['mean_auc'] 
        mean_scores.append(ret['mean_auc'])
        highs.append(ret['high_auc'])
        lows.append(ret['low_auc'])

    print('\nresult> mean auc: %f =?= %f (low %f ~ high %f) averaged over %d rounds\n' % \
        (np.mean(scores), np.mean(mean_scores), np.mean(lows), np.mean(highs), len(scores)))
    print('  + scores: %s' % scores[:10])
    print('  + lows:   %s' % lows[:10])
    print('  + highs:  %s' % highs[:10])

    print('\nglobal> feed in all y_true and y_pred from across all CV-cycles and then subdivide them into k folds ...')
    fp_roc = os.path.join(output_dir, 
         'global_roc-%s-P%s.%s' % (i, identifier, ext_plot) if not meta else 'global_roc_%s-C%s-P%s.%s' % \
         (meta, n_cv_cycles, identifier, ext_plot)) # overwrite global
    ret = drawROC(y_truex, y_predx, identifier=note, opath=fp_roc, summary_msg=footnote, n_folds=n_folds, save_fig=True)

    return

def drawROC(y_true, y_pred, **kargs):
    """

    Note
    -----
    y_true and y_pred can be produced by applying a classifier to multiple train-test splits (say n times). 
    then 'n_fold' parameter should be set to that number
    """

    from sklearn.cross_validation import StratifiedKFold

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # clear previous plots
    plt.clf()

    level = 0 # not normall in use 
    msg = kargs.get('identifier', kargs.get('message', ''))  # at least sequence pattern type (seq_ptype) should be in it
    plot_msg = msg  # text that serves as part of the plot description
    assert len(msg) > 0, "empty msg (consider at least sequence pattern type)"

    # output 
    ret = {}

    y_pred_prob = y_pred  # synonymous because y_pred has to be probablistic 

    # # [test]
    # if kargs.get('test_', False): 
    #     msg_ =  'drawROC-test> input dim > y_true (n=%d, t=%s) > example:\n%s\n' % (len(y_true), type(y_true), y_true[:10])
    #     msg_ += '                        > y_pred (n=%d, t=%s) > example:\n%s\n' % (len(y_pred), type(y_pred), y_pred[:10])   
    #     print(msg_); msg_ = ''

    # seq_ptype = kargs.get('seq_ptype')
    n_folds = kargs.get('n_folds', 10)
    force_loo = kargs.get('force_loo', False)
    shuffle = False 

    if force_loo: assert n_folds == 1
    n_bootstraps_max = kargs.get('n_bootstraps_max', Domain.n_bootstraps_max)

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
    print('drawROC> Identifier: %s > number of bootstrapped estimated auc scores: (%d >= %d) ~ %s' % \
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
        plt.title('ROC (%s)' % plot_msg)
    else: 
        plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # confidence interval
    cdict  = sampling.ci4(auc_scores, low=0.05, high=0.95) # mean=mean_auc
    ci_auc = (cdict['ci_low'], cdict['ci_high'])

    msg_ = '\nResult> Given prediction results at (%s)\n' % msg
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
        ext = 'tif' # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
        fpath = kargs.get('opath', kargs.get('ofile', None))
        if fpath is None: 
            basedir = output_dir = sys_config.read('DataExpRoot') # /phi/proj/poc7002/bulk_training/data-learner
            ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
            # output_dir = os.path.join(basedir, 'model_combined')
            fpath = os.path.join(output_dir, 'roc-P%s.%s' % (ext_plot, msg))

        print('output> saving performance (%s) to %s\n' % (msg, fpath))

        # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
        plt.savefig(fpath, bbox_inches='tight')  
    else: 
        plt.show()  # matplotlib.use('Agg'): AGG backend is for writing to file, not for rendering in a window
    plt.close()

    return ret

def saveFig(plt, fpath, ext='tif'):
    """
    fpath: 
       name of output file
       full path to the output file

    Memo
    ----
    1. supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff 

    """
    import os
    outputdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 

    # [todo] abstraction
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]  

    # supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not outputdir: outputdir = os.getcwd() # sys_config.read('DataExpRoot') # ./bulk_training/data-learner)
    assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir
    ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not fname: fname = 'roc-test.%s' % ext_plot
    fbase, fext = os.path.splitext(fname)
    assert fext[1:] in supported_formats, "Unsupported graphic format: %s" % fname

    fpath = os.path.join(outputdir, fname)
    print('output> saving performance to:\n%s\n' % fpath)
    
    # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight')   
    return

def binary_classify(ts, **kargs):  
    """

    Memo
    ----
    1. for bulk_train6 and prior versions, see learn.py
    """ 
    def get_fpath(code, model): 
        # e.g. cerner_blood_038.8_tset_bt_190.csv
        fname = '%s_%s_%s_%s_%s.%s' % (labname, model, code, ts_stem, bt_identifier, ts_ext)
        return os.path.join(tset_dir, fname)

    def get_fpath_mm(code): # get filename: model marginalized
        # e.g. cerner_038.8_tset_bt_190.csv
        fname = '%s_%s_%s_%s.%s' % (labname, code, ts_stem, bt_identifier, ts_ext)
        return os.path.join(combined_tset_dir, fname)

    def make_instance(idx): 
        for id_ in idx: 
            for l in labels: 
                yield (id_, l)

    import vector
    # [params]
    seq_ptype = seqparams.normalize_ptype(kargs.get('seq_ptype', 'regular'))
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)  

    bt_identifier = 'condition_drug'
    ts_stem = 'tset'
    ts_ext = 'csv' 
    verbose = kargs.get('verbose', False)
    f_dtype = {TSet.target_field: int, }
    ts_sep = ','

    # [params] files and dirs
    basedir = output_dir = os.path.join(os.getcwd(), 'test')  # or sys_config.read('DataExpRoot')
    tset_dir = os.path.join(os.getcwd(), 'data')
    assert os.path.exists(tset_dir)

    # [params] classifiers 
    clf_type = 'rbf_svm'  # logistic, rbf_svm/svm_rbf, poly_svm, linear_svm

    # load data 
    if ts is None or ts.empty: 
        raise ValueError

    # [test]
    div('Stats> Data profile (sequence ptype: %s, d2v_method: %s) ...' % (seq_ptype, d2v_method))
    # n_last = 10
    # fset_lastn = tsc.columns[-n_last:]
    # print('info> the last %d features:\n%s\n' % (n_last, fset_lastn))
    stats = profile(ts)

    clf = params = None
    if clf_type.startswith('sv') or clf_type.find('svm') >= 0: 
        # for SVM, using nested CV may be too expensive ... do model selection separately though this is not the best practice
        div(message='classify> SVM > Computing optimal kernel parameters', symbol='#')
        
        # 1. use model selection to determine kernel and its parameters
        clf, params = select_kernel_svm(ts, **kargs) # {'kernel': 'rbf', 'C': 10.0, 'gamma': 1000.0}
        
        # 2. hardcode 
        # clf = SVC(C=1, probability=True)

        # 2a. linear
        # params = params_linear = {'kernel': 'linear', 'C': 0.0001}
        # clf.set_params(**params)

        # 2b. rbf 
        # params = params_rbf = {'kernel': 'rbf', 'C': 100.0, 'gamma': 10.0}
        # clf.set_params(**params)

        # best_kernel = params['kernel']

    elif clf_type.startswith('log'): 
        # [note] sag is good for large data; liblinear for small data
        #        newton-cg, sag and lbfgs handle multinomial loss

        # 1. perform model selection separately to speed up
        clf, params = select_model(ts, clf_type='logistic')  
        # result: best params (score: 0.387592): {'penalty': 'l2', 'C': 0.001}

        # 1a. hardcode result
        # clf = LogisticRegression(C=1)
        # params = {'penalty': 'l1', 'C': 1000}  # using 'l1' similar to SVM takes longer to fit, why?
        # clf.set_params(**params)

        # 2. perform nested CV
        # clf, params = Params.getClassifier(name='logistic', penalty='l2')  # [{'C': reg, 'penalty': penalties}]
    else: 
        raise NotImplementedError

    div(message='Classifier parameters (prior to evaluate()): %s' % clf.get_params(), symbol='#')

    # evaluate model
    div(message='Evaluating the model', symbol='*')
    kargs['meta_'] = clf_type
    evaluate(ts, classifier=clf, params=params, ts_test=None, **kargs)
    
    return

