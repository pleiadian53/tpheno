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
plt.style.use('ggplot')  # 'seaborn'

from scipy.stats import sem  # compute standard error

try:
    import cPickle as pickle
except:
    import pickle

import statistics
import scipy  # sampling 

# feature selection 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE

### classifiers 
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
    
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.dummy import DummyClassifier

# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics

### CV 
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

# for randomize CV
from time import time
# from scipy.stats import randint as sp_randint


#########################################################################################################
#
# Module for turnning model parameters (hyperparameters of classifiers). 
# 
# Also see predecessor module: seqmaker.evaluate
#
#
#


NUM_TRIALS = 30

# SVM
gammax = np.logspace(-5, 4, 10) # gamma: inverse of bandwidth
degreex = range(2, 10)
regC = np.logspace(-3, 3, 7) 
isProba = True  # need probabilities when using ROC for performance evaluation
tolerance = 1e-4

{'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}

# Basic SVM options
# method: selectEstimator(name) -> (estimator, param_grid)
SVM_Models = {'rbf_svm': [SVC(kernel='rbf', probability=isProba, class_weight='balanced'), {"C": regC, "gamma": gammax}], 
             'rbf_svm_auto': [SVC(kernel='rbf', gamma='auto', probability=isProba, class_weight='balanced'), {"C": regC, }], 
             'linear_svm': [SVC(kernel='linear', probability=isProba, class_weight='balanced'), {"C": regC, }], 
             'poly_svm': [SVC(kernel='poly', probability=isProba, class_weight='balanced'), {"C": regC, "degree": degreex}], 
          }
Logistic_Models = {'l2_logistic': [LogisticRegression(tol=tolerance, class_weight='balanced', solver='saga', penalty='l2'), 
                                       {'C': regC, }], 
                   'l1_logistic': [LogisticRegression(tol=tolerance, class_weight='balanced', solver='saga', penalty='l1'), 
                                       {'C': regC, }],
}

RandomForest_Models = {'random_forest': [RandomForestClassifier(n_estimators=20),
                                           {"max_depth": [3, None],
                                            "max_features": [1, 3, 10],
                                            "min_samples_split": [2, 3, 10],
                                            "min_samples_leaf": [1, 3, 10],
                                            "bootstrap": [True, False],
                                            "criterion": ["gini", "entropy"]}], 
                                            }

# method: selectEstimatorRandomCV(name) -> (estimator, param_dist)   where param_dist: parameter distribution (or any function with rvs method)
# RandomForest_ModelsRandomCV = {'random_forest': [RandomForestClassifier(n_estimators=20), 
#                                            {"max_depth": [3, None],
#                                                          "max_features": sp_randint(1, 11),
#                                                          "min_samples_split": sp_randint(2, 11),
#                                                          "min_samples_leaf": sp_randint(1, 11),
#                                                          "bootstrap": [True, False],
#                                                          "criterion": ["gini", "entropy"]}], 
# }

cycledColors = ['aqua', 'turquoise', 'cornflowerblue', 'teal', 'green', 'darkorange']

def runNestedCV(X, y, estimator=None, param_grid=None, nfold_inner=5, nfold_outer=5, n_trials=30, **kargs): 
    """
    Run nested CV on training data (X, y) 
    PS: Very time-consuming. 
    
    If we are not trying evaluate the model at the same time, then there's no need. 

    Params
    ------
      estimator
        classifier, param_grid = SVMModel['rbf_svm']
      param_grid 

      <optional>
      estimator_name 

    """
    # from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    if estimator is None or param_grid is None: 
        clf_name = kargs.get('estimator_name', 'linear_svm')
        print('runNestedCV> Use %s by default ...' % clf_name)
        estimator, param_grid = selectEstimator(clf_name)

    if n_trials is None: n_trials = NUM_TRIALS
    
    # GridSearch CV metric 
    scoring = kargs.get('scoring', 'roc_auc')

    # Arrays to store scores
    non_nested_scores = np.zeros(n_trials)
    nested_scores = np.zeros(n_trials)
    best_params = np.zeros(n_trials)

    # Loop for each trial
    for i in range(n_trials):
 
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
        inner_cv = KFold(n_splits=nfold_inner, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=nfold_outer, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_cv, scoring=scoring)
        clf.fit(X, y)
        # best_params[i], non_nested_scores[i] = clf.best_params_, clf.best_score_
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    score_difference = non_nested_scores - nested_scores
    print("Average difference of {0:6f} with std. dev. of {1:6f}."
          .format(score_difference.mean(), score_difference.std()))

    return (non_nested_scores, nested_scores, score_difference)

def selectModelRandomCV(X, y, estimator=None, param_dist=None, n_iter=20, **kargs): 
    # specify parameters and distributions to sample from
    # run randomized search
    n_iter_search = n_iter
    estimator_cv = RandomizedSearchCV(estimator, param_distributions=param_dist,
                                        n_iter=n_iter_search)
    return estimator_cv

def selectParams(X, y, classifier=None, param_grid=None, n_folds=5, verbose=True):
    """
    Similar to selectModel but returns best parameters (instead of the 
    tuned classifier)

    Memo
    ----
    1. To view the best params (e.g. RBF SVM)
        print('Best C:',clf.best_estimator_.C) 
        print('Best Kernel:',clf.best_estimator_.kernel)
        print('Best Gamma:',clf.best_estimator_.gamma)

    """
    if classifier is None: # then default RBF SVM
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        classifier = svm.SVC(kernel='rbf')
    else: 
        # if param_grid is None: # then nothing left to do 
        #    print('selectParams> Warning: No param_grid given => Assume default parameters ...')
        #    return classifier.get_params()
        assert param_grid is not None, "No param_grid from which to search best parameters."

    # performance metric
    best_params = {}
    if verbose: 
        scores = ['precision', 'recall', 'roc_auc', ]
        tuned_parameters = param_grid

        # Split the dataset in two equal parts (so that the performance evaluation is done on a separate set of data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(classifier, tuned_parameters, cv=n_folds,
                       scoring=score)  # '%s_macro' % score
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            best_params = clf.best_params_
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
    else: 
        grid_search = GridSearchCV(classifier, param_grid, cv=nfolds)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
    assert len(best_params) > 0
    return best_params

def selectModel(X, y, estimator=None, param_grid=None, n_folds=5, **kargs): 
    """
    Select the best hyperparameter setting for the input estimator (usu a classifier such as SVC) using 
    grid search strategy. 

    Related
    -------
    selectModelRandomCV()

    """
    def profile(): 
        print('selectModel> classifier parameters after CV: %s' % str(estimator_cv.get_params()))
        print("   + The best parameters are %s with a score of %0.2f\n" % (estimator_cv.best_params_, estimator_cv.best_score_))
        print("   + The best estimator:\n%s\n" % estimator_cv.best_estimator_)

        div(message="Detailed grid scores", symbol='#')
        print('\n')
        means = estimator_cv.cv_results_['mean_test_score']  # this attribute doesn't come with clf_ (without the CV wrapper)
        stds = estimator_cv.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, estimator_cv.cv_results_['params']):
            print("    + %0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print('\n')
        return
    def get_clf_name(estimator): 
        name = None
        try: 
            name = estimator.__name__
        except: 
            print('info> infer classifier name from class name ...')
            # name = str(estimator).split('(')[0]
            name = estimator.__class__.__name__
        return name
    def isMulticlass(y): 
        if len(np.unique(y)) > 2: 
            return True
        return False
    def isProbClassifier(estimator): 
        # assert hasattr(estimator, '__call__'), "Not a valid function: %s" % estimator
        # estimator itself is not a function
        return hasattr(estimator, 'predict_proba')
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
    from batchpheno.utils import div
    # from sklearn.cross_validation import StratifiedShuffleSplit

    n_folds_ms = n_folds
    clf_type = kargs.get('clf_type', kargs.get('name', 'logistic'))
    estimator_name = kargs.get('estimator_name', 'linear_svm')
    if estimator is None or param_grid is None: 
        print('selectModel> Use %s by default ...' % estimator_name)
        estimator, param_grid = selectEstimator(estimator_name)
    else: 
        estimator_name = get_clf_name(estimator)  # classifier class may not have a name
        assert param_grid is not None, "No parameter setting provided!"
    
    # roc_auc does not support multiclass (use 'f1_micro', 'f1_macro', 'neg_log_loss')
    default_metric_multiclass = 'neg_log_loss' if isProbClassifier(estimator) else 'f1_micro'
    metric = kargs.get('scoring', 'roc_auc' if not isMulticlass(y) else default_metric_multiclass) # evaluation metric in grid search 
    div(message='selectModel: classifier: %s, params grid: %s' % (estimator_name, param_grid), symbol='%')

    # X, y = transform(ts, standardize_='minmax')  # use TSet.toXY(ts)
    
    # cv = StratifiedShuffleSplit(y, n_iter=5, ltest_size=0.2, random_state=10)

    ### parameter search strategy 
    # 0. algorithm-specific search (e.g. LassoCV)
    # a. randomized search 

    # b. grid search
    print('  + Grid search (metric=%s) | X (dim=%s, n_classes=%d)' % \
        (metric, str(X.shape), len(np.unique(y))))
    estimator_cv = GridSearchCV(estimator, param_grid=param_grid, cv=n_folds_ms, scoring=metric)
    estimator_cv.fit(X, y)
    best_params, best_score = estimator_cv.best_params_, estimator_cv.best_score_
    profile()

    # is this needed?
    # estimator.set_params(**best_params[0][0])  # do not set estimator_cv! 

    return estimator_cv.best_estimator_ # estimator_cv.get_params() to get its tuned parameters

def selectEstimatorByName(name, **kargs):
    return selectEstimator(name, **kargs) 
def selectEstimator(name, is_multiclass=False): 
    """
    Select estimator (and parameter grid) by its name. 

    Memo
    ----
    In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, 
    and uses the cross- entropy loss if the ‘multi_class’ option is set to ‘multinomial’. 
    (Currently the 'multinomial' option is supported only by the ‘lbfgs’, ‘sag’ and ‘newton-cg’ solvers.)

    """
    from sklearn.multiclass import OneVsRestClassifier
    # if is_multiclass: return selectMulticlassEstimator(name)
    name = name.lower()
    estimator = param_grid = None
    if name.find('svm') >= 0: 
        if SVM_Models.has_key(name): 
            estimator, param_grid = SVM_Models[name]
        else: 
            print('selectEstimator> Warning: Use %s' % ', '.join([k for k in SVM_Models.keys()]))
            name = 'linear_svm'; print('  + use %s by default' % name)
            estimator, param_grid = SVM_Models[name]

        if is_multiclass: 
            estimator = OneVsRestClassifier(estimator)
    elif name.find('log') >= 0:  # or logistic
        if Logistic_Models.has_key(name): 
            estimator, param_grid = Logistic_Models[name]
        else: 
            print('selectEstimator> Warning: Use %s' % ', '.join([k for k in Logistic_Models.keys()]))
            name = 'l2_logistic'
            print('  + use %s by default' % name)
            estimator, param_grid = Logistic_Models[name]

        if is_multiclass: 
            # Currently the 'multinomial' option is supported only by the ‘lbfgs’, ‘sag’ and ‘newton-cg’ solvers.
            # solver <- 'saga' allows {'multinomial', 'l1'}
            estimator.set_params(multi_class='multinomial', solver='saga')  # inplace, 'ovr': one-vs-all
            # tune optimization algorithm? ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
            
    else: 
        raise NontImplementedError, "Unsupported estimator: %s" % name

    return (estimator, param_grid)

def selectOptimalEstimatorByName(name, X, y, **kargs): 
    return selectOptimalEstimator(name, X, y, **kargs)
def selectOptimalEstimator(name, X, y, **kargs):  # [params] is_multiclass
    """

    Params
    ------

    **kargs
      scoring: 'roc_auc' (binary classification only)
               'f1_micro'
               'f1_macro'
               'neg_log_loss' (probablistic classifier)

    """
    from sklearn.model_selection import train_test_split

    N = X.shape[0]
    maxNSample = kargs.get('max_n_samples', 5000)
    Xp, yp = X, y
    if N > maxNSample: 
        Xp, X_test, yp, y_test = train_test_split(X, y, train_size=maxNSample, random_state=53)

    estimator, param_grid = selectEstimator(name, is_multiclass=kargs.get('is_multiclass', False)) # first select classifier by name 
    
    # model selection using data (X, y)
    print('selectOptimalEstimator> X (dim=%s), classifier profile:\n%s\n' % (str(Xp.shape), estimator))
    return selectModel(Xp, yp, estimator=estimator, param_grid=param_grid, **kargs) # [params] n_folds/5, scoring/'roc_auc' (if binary)

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
    if not fname: fname = 'generic-test.%s' % ext_plot
    fbase, fext = os.path.splitext(fname)
    assert fext[1:] in supported_formats, "Unsupported graphic format: %s" % fname

    fpath = os.path.join(outputdir, fname)
    print('output> saving performance to:\n%s\n' % fpath)
    
    # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight')   
    return

def plotComparison(non_nested_scores, nested_scores, score_difference, fpath=None): 
    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
              x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
               ["Non-Nested CV - Nested CV Score"],
                bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    # plt.show()
    if fpath is None: fpath = 'roc-iris.tif'
    saveFig(plt, fpath=fpath)

    return 

def runNestedCVROC(X, y, estimator=None, param_grid=None, nfold_inner=5, nfold_outer=5, n_trials=1, **kargs): 
    """
    Similar to runNestedCV() but plot ROC curves as well. 

    Params
    ------
      estimator
        classifier, param_grid = SVMModel['rbf_svm']
      param_grid 

      <optional>
      estimator_name 

    """
    # from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    if estimator is None or param_grid is None: 
        print('runNestedCV> Use RBF SVM by default ...')
        estimator, param_grid = selectEstimator(kargs.get('estimator_name', 'linear_svm'))
    if n_trials is None: n_trials = 1
    
    # GridSearch CV metric 
    scoring = kargs('scoring', 'roc_auc')

    # Arrays to store scores
    non_nested_scores = np.zeros(n_trials)
    nested_scores = np.zeros(n_trials)
    best_params = np.zeros(n_trials)

    # Loop for each trial
    for i in range(n_trials):
 
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
        inner_cv = KFold(n_splits=nfold_inner, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=nfold_outer, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_cv, scoring=scoring)
        clf.fit(X, y)
        best_params[i], non_nested_scores[i] = clf.best_params_, clf.best_score_

        # A. use scikit-learn's looping
        # Nested CV with parameter optimization
        # nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        # nested_scores[i] = nested_score.mean

        # customized looping
    score_difference = non_nested_scores - nested_scores
    print("Average difference of {0:6f} with std. dev. of {1:6f}."
          .format(score_difference.mean(), score_difference.std()))

    return (non_nested_scores, nested_scores, score_difference)

def runCVROCMulticlass2(ts_params, **kargs):  
    """
    Generalize ROC curves to multiclass settings. 
    Load data from File. 

    In order to plot ROC, y has to be binarized. 

    Params
    ------
    a. parameters to load training data 
        ts_params is a class object containing the following parameters

        - file name
            cohort
            d2v_method 
            seq_ptype
            index

        - prefix directory 
           (cohort) given by a)
           dir_type
       

    Output
    ------
    ROC plots 
    a. micro-averaged + std
        'roc-microAvg_std-%s.tif' % kargs.get('identifier', 'generic')
    b. per-class, micro, macro 
        'roc-avg-%s.tif' % kargs.get('identifier', 'generic')

    Memo
    ----
    1. potential parameters needed 
       a. a map from numeric class labels to canonical labels 

    """
    def choose_classifier(): 
        pass 
    def test_input(): 
        assert X.shape[0] == y.shape[0], "N=%d <> nL=%d" % (X.shape[0], y.shape[0])
        print('  + training data dimension: %d' % X.shape[1])
        ys = random.sample(y, 5)
        print('  + (n_classes=%d) example y:\n%s\n' % (n_classes, str(ys)))
        return
    def search_label(y_subset, label): 
        # find the positions of 'label' in 'y_subset'
        idx = np.where(np.array(y_subset) == label)[0]
        return idx
    def encode_labels(): 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        lb.fit(class_labels)
        label_pairs = zip(lb.transform(class_labels), class_labels)
        print('label_encoder> labels vs numeric labels ...')
        for l, cl in label_pairs: 
            print('  + %s ~> %s' % (l, cl))
        lookup = dict([(tuple(l), cl) for l, cl in label_pairs])
        return (lb, lookup)
    def load_from_csv(cv_id):  # [params] cohort_name, d2v_method
        cohort_name = ts_params.cohort 
        d2v_method = ts_params.d2v_method
        seq_ptype = ts_params.seq_ptype
        # dir_type = ts_params.dir_type

        ts = TSet.load(cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, index=cv_id, dir_type='cv_train')
        X_train, y_train = TSet.toXY(ts)
        ts = TSet.load(cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, index=cv_id, dir_type='cv_test')
        X_test, y_test = TSet.toXY(ts)
        return (X_train, X_test, y_train, y_test)  # use scikit-learn's convention
    def resolve_output():
        outputdir = os.getcwd()
        for dr in ('outputdir', 'prefix', ):  
            if kargs.has_key(dr): 
                outputdir = kargs[dr]
        assert os.path.exists(outputdir), "Invalid output dir: %s" % outputdir
        return outputdir 

    from scipy import interp
    from itertools import cycle
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from sklearn import preprocessing

    yl = np.unique(y)  # (unique) class labels
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    n_folds = kargs.get('n_folds', 5)
    random_state = np.random.RandomState(0)

    # transform class labels (y) into numerical values + binarize 
    # [memo] binarized labels don't seem to work with cv.split()
    # y, lookup = transform_label()  # y: binarized label; lookup: numeric -> label
    lb, lookup = encode_labels() 
  
    plt.clf() 
    # 1. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 1.5 aggregate all folds
    fpr_cv = {i: [] for i in range(n_classes)}
    tpr_cv = {i: [] for i in range(n_classes)}
    auc_cv = {i: [] for i in range(n_classes)}

    # [test]
    # test_input()

    # 2. Micro-averaged ROC 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    classifier = kargs.get('classifier', selectEstimator('linear_svm')[0])
    test_classifier()
    cv = StratifiedKFold(n_splits=n_folds)
    # for ifold, (train, test) in enumerate(cv.split(X, y)):  # y: binarized labels
    for ifold in range(n_folds):
        X_train, X_test, y_train, y_test = load_from_csv(ifold)

        y_pred = classifier.fit(X_train, y_train).predict_proba(X_test) # y_pred is a 2D array
        y_test = lb.transform(y_test)  # binarize | cf: le.transform(y[test])

        # # per-class FPR and TPR
        # test_class(y_test, y_pred) # i <- random
        for i in range(n_classes):  # this makes sense only when y is binarized y_test[:, i] => ith class
            # idx = search_label(y_test, i)
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            # aggregate values from all folds
            fpr_cv[i].append(fpr[i])
            tpr_cv[i].append(tpr[i])
            auc_cv[i].append(roc_auc[i])

        # Compute micro-average ROC curve and ROC area
        # [memo] ok to use binarized labels in roc_curve()? 
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())  # true label vs prediction
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        tprs.append(interp(mean_fpr, fpr['micro'], tpr['micro']))
        tprs[-1][0] = 0.0
        # roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc["micro"])
        plt.plot(fpr['micro'], tpr['micro'], lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (ifold, roc_auc['micro']))

    # plot micro-average ROC curve as a function of (tprs, aucs, mean_fpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # wrt micro-averaged
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean micro-average ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # standard deviation across k-folds
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    title_msg = kargs.get('title', "ROC curve")
    plt.title(title_msg)
    plt.legend(loc="lower right")
    # plt.show()

    # save plot    
    prefix = resolve_output()
    identifier = kargs.get('identifier', 'generic')
    fname = 'roc-microAvg_std-%s.tif' % identifier
    saveFig(plt, fpath=os.path.join(prefix, fname))

    ### per-class plot + micro + macro 
    
    # # first, expand values from k folds 
    for i in range(n_classes):
        fpr_cv[i] = np.concatenate(fpr_cv[i])
        tpr_cv[i] = np.concatenate(tpr_cv[i])

    # consolidate micro average data across k folds
    fpr_cv['micro'] = mean_fpr
    tpr_cv['micro'] = mean_tpr
    auc_cv["micro"] = mean_auc # auc(fpr_cv["micro"], tpr_cv["micro"])  # "grand" average

    # fpath = 'roc-avg-%s.tif' % kargs.get('identifier', 'generic')
    # title_msg = '?'
    evalMulticlassROCCV(y, fpr_cv, tpr_cv, auc_cv, prefix=prefix, identifier=identifier, class_lookup=lookup) # [params] title

    return

def runCVROCMulticlassNNs(X, y, classifier, **kargs):  
    """
    Generalize ROC curves to multiclass settings in deep learning setting (usually unrealistically slow). 

    In order to plot ROC, y has to be binarized. 

    Output
    ------
    ROC plots 
    
    parameters: identifier, prefix (prefix directory)

    a. micro-averaged + std
        'roc-microAvg_std-%s.tif' % kargs.get('identifier', 'generic')
    b. per-class, micro, macro 
        'roc-macroAvgPerClass-%s.tif' % kargs.get('identifier', 'generic')

    Memo
    ----
    1. potential parameters needed 
       a. a map from numeric class labels to canonical labels 

    """
    def test_input(): 
        assert X.shape[0] == y.shape[0], "N=%d <> nL=%d" % (X.shape[0], y.shape[0])
        print('  + training data dimension: %d' % X.shape[1])
        ys = random.sample(y, 5)
        print('  + (n_classes=%d) example y:\n%s\n' % (n_classes, str(ys)))
        return
    def test_class(y_test, y_pred, i=None): # [params] lookup
        # assert y_pred.shape[0] == y_test.shape[0], "y_test(n=%d) <> y_pred(n=%d)" % (y_test.shape[0], y_pred.shape[0])
        print('  + y_pred dim: %s ~? n_classes=%d' % (str(y_pred.shape), n_classes))
        
        y_subset = y_test
        if i is None: i = random.sample(lookup.keys(), 1)[0] # random.choice(n_classes)
        print('  + y_pred(class=i) dim: %s ~? y_test: %d' % (str(y_pred[:, i].shape), len(y_test)))

        idx = search_label(y_subset, i) # does the class (i) exist
        assert len(idx) > 0, "label=%s is not found in this batch of test points:\n%s\n" % (lookup[i], y_subset)
        print('  +    y_subset         :\n%s\n' % y_subset)  # after y is converted to numerical rep. 
        print('  +    idx              :\n%s\n' % idx)
        print('  +    y_subset(class=i):\n%s\n' % y_subset[idx]) # after y is converted to numerical rep. 
        return
    def search_label(y_subset, label): 
        # find the positions of 'label' in 'y_subset'
        idx = np.where(np.array(y_subset) == label)[0]
        return idx
    def transform_label(): # y 
        # le = preprocessing.LabelEncoder()  # this converts to just regular numbers
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        # class_labels = np.unique(y)
        
        # le.fit(class_labels)  # fit unique labels
        lb.fit(class_labels)
        # yp = le.transform(y)
        yb = lb.transform(y)
        # assert yp.shape == y.shape

        # [test]
        label_pairs = zip(lb.transform(class_labels), class_labels)
        print('transform_label> labels vs numeric labels ...')
        for l, cl in label_pairs: 
            print('  + %s ~> %s' % (l, cl))

        # binarize
        # preprocessing.label_binarize(yp, classes=class_labels)
        lookup = dict([(tuple(l), cl) for l, cl in label_pairs])
        return (yb, lookup) 
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
    def resolve_output():
        outputdir = os.getcwd()
        for dr in ('outputdir', 'prefix', ):  
            if kargs.has_key(dr): 
                outputdir = kargs[dr]
        assert os.path.exists(outputdir), "Invalid output dir: %s" % outputdir
        return outputdir

    from scipy import interp
    from itertools import cycle
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from sklearn import preprocessing

    # yl = np.unique(y)  # (unique) class labels
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    n_folds = kargs.get('n_folds', 5)
    random_state = np.random.RandomState(0)

    # transform class labels (y) into numerical values + binarize 
    # [memo] binarized labels don't seem to work with cv.split()
    # y, lookup = transform_label()  # y: binarized label; lookup: numeric -> label

    # [todo] No need to create this map, use encoder's inverse_transform()
    lb, lookup = encode_labels(y) # turn string labels to integers
  
    plt.clf() 
    # 1. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # [test]
    # test_input()

    # 2. Micro-averaged ROC 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 2a aggregate all folds
    tpr_cv = {i: [] for i in range(n_classes)}
    fpr_cv = {i: mean_fpr for i in range(n_classes)}
    auc_cv = {i: [] for i in range(n_classes)} # used to compute std
    fpr_cv['micro'], tpr_cv['micro'], auc_cv['micro'] = [], [], []

    # 3. Per-class ROC 
    tprs_pc = []

    # Keras parameters
    my_callbacks = [EarlyStopping(monitor='auc_roc', patience=kargs.get('patience', 300), verbose=1, mode='max')]
    shuffle = kargs.get('shuffle', True)
    batch_size = kargs.get('batch_size', 32)
    epochs = kargs.get('epochs', 200)

    cv = StratifiedKFold(n_splits=n_folds)
    for ifold, (train, test) in enumerate(cv.split(X, y)):  # y: binarized labels
        try: 
            # customize this 
            y_train = lb.transform(y[train])
            classifier.fit(X[train], y_train,
                    validation_split=0.0,  # set to 0 here since we have a separate model evaluation
                    shuffle=shuffle,
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    callbacks=my_callbacks)
            y_pred = classifier.predict_proba(X[test]) # y_pred is a 2D array
        except Exception, e: 
            # print('+ error: %s' % e)
            print('   + type(X):%s' % type(X))
            print('   + type(train): %s, examples: %s' % (type(train), train))
            raise ValueError, e 

        y_test = lb.transform(y[test])  # binarize | cf: le.transform(y[test])

        # # per-class FPR and TPR
        # test_class(y_test, y_pred) # i <- random
        for i in range(n_classes):  # this makes sense only when y is binarized y_test[:, i] => ith class
            # idx = search_label(y_test, i)
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i]) # customize this
            roc_auc[i] = auc(fpr[i], tpr[i])  # area under the curve

            tpr_cv[i].append(interp(mean_fpr, fpr[i], tpr[i]))
            tpr_cv[i][-1][0] = 0.0
            auc_cv[i].append(roc_auc[i])

        # Compute micro-average ROC curve and ROC area
        # [memo] ok to use binarized labels in roc_curve()? 
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())  # true label vs prediction
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        tprs.append(interp(mean_fpr, fpr['micro'], tpr['micro']))
        tprs[-1][0] = 0.0

        # roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc["micro"])
        plt.plot(fpr['micro'], tpr['micro'], lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (ifold, roc_auc['micro']))

        # per-class 
        # tprs_pc[i].append(interp(mean_fpr, fpr[i], tpr[i]))

    # plot micro-average ROC curve as a function of (tprs, aucs, mean_fpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # wrt micro-averaged
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean micro-average ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # standard deviation across k-folds
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    title_msg = kargs.get('title', "ROC curve")
    plt.title(title_msg)
    plt.legend(loc="lower right")
    # plt.show()

    # save plot    
    # fpath = kargs.get('fpath', None)
    prefix = resolve_output()
    identifier = kargs.get('identifier', 'generic')
    fname = 'roc-microAvg_std-%s.tif' % identifier
    # if fpath is None: fpath = 'roc-microAvg_std-%s.tif' % kargs.get('identifier', 'generic')
    saveFig(plt, fpath=os.path.join(prefix, fname))

    ### per-class plot + micro + macro 
    
    # test_auc_cv_metrics() # fpr_cv, trp_cv, auc_cv
    # first, merge values from k folds 
    mean_fpr_cv = fpr_cv
    mean_tpr_cv = {i: [] for i in range(n_classes)}
    mean_auc_cv = {i: auc_cv[i] for i in range(n_classes)}
    for i in range(n_classes):
        # fpr_cv[i] = np.linspace(0, 1, 100) # np.concatenate(fpr_cv[i]) 
        mean_tpr_cv[i] = np.mean(tpr_cv[i], axis=0) # mean_tpr of ith-class; np.concatenate(tpr_cv[i])
        mean_tpr_cv[i][-1] = 1.0
        mean_auc_cv[i] = auc(mean_fpr_cv[i], mean_tpr_cv[i])  # this is just a list of AUCs

    # consolidate micro average data across k folds
    mean_fpr_cv['micro'] = mean_fpr
    mean_tpr_cv['micro'] = mean_tpr
    mean_auc_cv["micro"] = mean_auc # auc(fpr_cv["micro"], tpr_cv["micro"])  # "grand" average

    # fpath = 'roc-avg-%s.tif' % kargs.get('identifier', 'generic')
    # title_msg = '?'
    # [params] title
    res = evalMulticlassROCCV(y, fpr=mean_fpr_cv, tpr=mean_tpr_cv, roc_auc=mean_auc_cv, 
            prefix=prefix, identifier=identifier, class_lookup=lookup, 
            target_labels=kargs.get('target_labels', [])) 

    return res 

def modelEvaluate(X, y, **kargs): 
    tCV = kargs.get('use_cv', True)

    # 1. evaluation by ROC 
    modelEvaluateFunc = runCVROCMulticlass if tCV else runROCMulticlass
    res = modelEvaluateFunc(X, y, **kargs)  # classifier, model

    # [post-condition]
    # tag other evaluation metrics (e.g. accuracy) 
    default_metrics = ['neg_log_loss', 'accuracy', 'f1_micro', 'f1_macro', ] # "roc_auc" does not support multiclass (use plotROC)

    # 2. evaluation by other metrics such as accuracy
    #    [todo] non-CV version: runMulticlass
    if not kargs.has_key('n_folds'): kargs['n_folds'] = 5 
    if not kargs.has_key('metrics'): kargs['metrics'] = default_metrics
    res0 = runCVMulticlass(X, y, **kargs)  # metrics=kargs.get('metrics', default_metrics), n_folds=n_folds
    res.update(res0)

    return res

def runCVMulticlass(X, y, **kargs):
    """
    Evaluate the given classifier using the input data (X, y). 

    The counterpart of dnn_utils.modelEvaluate but takes in "classcial" classifiers (i.e. non-NN-based)
    """ 
    def normalized_metrics(): 
        # [note] 1. in dnn_utils, this corresponds to ['loss', 'acc', 'auc_roc']
        #        2. "roc_auc" does not support multiclass 
        metrics = kargs.get('metrics', ['neg_log_loss', 'accuracy', 'f1_micro', 'f1_macro', ])  # 'roc_auc'
        for i, metric in enumerate(metrics): 
            if metric.startswith('los'):
                metrics[i] = 'neg_log_loss'
            elif metric.find('auc') >= 0: 
                metrics[i] = 'roc_auc'
        return metrics
    def post_normalize_metrics(res): 
        name_map = {'neg_log_loss': 'loss', 
                    'accuracy': 'acc', }   # ... so that the names become consistent with those from Keras' model.evaluate()
        for metric, score in res.items(): 
            if metric in name_map: 
                res[name_map[metric]] = score  # make synonyms
        return res
    def resolve_model(): 
        classifier = None

        # classifier is reserved for "classical classifiers" (e.g. SVM) while model usually refers to NNS-based models, which is unlikely used here
        for opt in ['classifier', 'model', ]:  
            if kargs.has_key(opt):
                classifier = kargs[opt]
                break 
        if classifier is None: 
            print('runCVMulticlass> Warning: No classifier provided. Use default ...')
            classifier = selectEstimator('linear_svm')[0]  # default 
        return classifier

    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    # from sklearn import preprocessing
    from sklearn.metrics import accuracy_score

    res = {}

    class_labels = np.unique(y)
    n_classes = len(class_labels)
    print('modelEvaluate> dim(X): %s, n_classes: %d' % (str(X.shape), n_classes))
    
    n_folds = kargs.get('n_folds', 5)
    myCV = StratifiedKFold(n_splits=n_folds)
    classifier = resolve_model()
    for metric in normalized_metrics():  # 'roc_auc'
        scores = cross_val_score(classifier, X, y, cv=myCV, scoring=metric)
        # accuracy = accuracy_score(labels, y_pred)
        print('modelEvaluate> under metric: %s\n ... scores(n_folds=%d):\n%s\n' % (metric, n_folds, scores))

        if kargs.get('take_average', True): 
            res[metric] = np.mean(scores)
        else:  
            res[metric] = list(scores)
    post_normalize_metrics(res)
    return res

def runCVROCMulticlass(X, y, **kargs):  
    """
    Generalize ROC curves to multiclass settings. 

    In order to plot ROC, y has to be binarized. 

    Output
    ------
    ROC plots 
    
    parameters: identifier, prefix (prefix directory)

    a. micro-averaged + std
        'roc-microAvg_std-%s.tif' % kargs.get('identifier', 'generic')
    b. per-class, micro, macro 
        'roc-macroAvgPerClass-%s.tif' % kargs.get('identifier', 'generic')

    Memo
    ----
    1. potential parameters needed 
       a. a map from numeric class labels to canonical labels 

    2. model evaluation metrics 
       scoring: http://scikit-learn.org/stable/modules/model_evaluation.html

    """
    def choose_classifier(): 
        pass 
    def test_input(): 
        assert X.shape[0] == y.shape[0], "N=%d <> nL=%d" % (X.shape[0], y.shape[0])
        print('  + training data dimension: %d' % X.shape[1])
        ys = random.sample(y, 5)
        print('  + (n_classes=%d) example y:\n%s\n' % (n_classes, str(ys)))
        return
    def test_class(y_test, y_pred, i=None): # [params] lookup
        # assert y_pred.shape[0] == y_test.shape[0], "y_test(n=%d) <> y_pred(n=%d)" % (y_test.shape[0], y_pred.shape[0])
        print('  + y_pred dim: %s ~? n_classes=%d' % (str(y_pred.shape), n_classes))
        
        y_subset = y_test
        if i is None: i = random.sample(lookup.keys(), 1)[0] # random.choice(n_classes)
        print('  + y_pred(class=i) dim: %s ~? y_test: %d' % (str(y_pred[:, i].shape), len(y_test)))

        idx = search_label(y_subset, i) # does the class (i) exist
        assert len(idx) > 0, "label=%s is not found in this batch of test points:\n%s\n" % (lookup[i], y_subset)
        print('  +    y_subset         :\n%s\n' % y_subset)  # after y is converted to numerical rep. 
        print('  +    idx              :\n%s\n' % idx)
        print('  +    y_subset(class=i):\n%s\n' % y_subset[idx]) # after y is converted to numerical rep. 
        return
    def test_classifier(): 
        print('  + input classifier: %s' % classifier)
        return
    def test_auc_cv_metrics(): # fpr_cv, trp_cv, auc_cv
        return
    def search_label(y_subset, label): 
        # find the positions of 'label' in 'y_subset'
        idx = np.where(np.array(y_subset) == label)[0]
        return idx
    def transform_label(): # y 
        # le = preprocessing.LabelEncoder()  # this converts to just regular numbers
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        # class_labels = np.unique(y)
        
        # le.fit(class_labels)  # fit unique labels
        lb.fit(class_labels)
        # yp = le.transform(y)
        yb = lb.transform(y)
        # assert yp.shape == y.shape

        # [test]
        label_pairs = zip(lb.transform(class_labels), class_labels)
        print('transform_label> labels vs numeric labels ...')
        for l, cl in label_pairs: 
            print('  + %s ~> %s' % (l, cl))

        # binarize
        # preprocessing.label_binarize(yp, classes=class_labels)
        lookup = dict([(tuple(l), cl) for l, cl in label_pairs])
        return (yb, lookup) 
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
    def resolve_model(): 
        classifier = None

        # classifier is reserved for "classical classifiers" (e.g. SVM) while model usually refers to NNS-based models, which is unlikely used here
        for opt in ['classifier', 'model', ]:  
            if kargs.has_key(opt):
                classifier = kargs[opt]
                break 
        if classifier is None: 
            print('runCVROCMulticlass> Warning: No classifier provided. Use default ...')
            classifier = selectEstimator('linear_svm')[0]  # default 
        return classifier
    def resolve_output():
        outputdir = os.getcwd()
        for dr in ('outputdir', 'prefix', ):  
            if kargs.has_key(dr): 
                outputdir = kargs[dr]
        assert os.path.exists(outputdir), "Invalid output dir: %s" % outputdir
        return outputdir  

    from scipy import interp
    from itertools import cycle
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn import preprocessing
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import average_precision_score

    # font size
    # font = {'family' : 'normal',
    #     'weight' : 'bold',
    #     'size'   : 22}
    # plt.rc('font', **font)

    # yl = np.unique(y)  # (unique) class labels
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    n_folds = kargs.get('n_folds', 5)
    random_state = np.random.RandomState(0)

    # transform class labels (y) into numerical values + binarize 
    # [memo] binarized labels don't seem to work with cv.split()
    # y, lookup = transform_label()  # y: binarized label; lookup: numeric -> label

    # [todo] No need to create this map, use encoder's inverse_transform()
    lb, lookup = encode_labels(y) # turn string labels to integers
  
    plt.clf() 
    # 1. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # [test]
    # test_input()

    # 2. Micro-averaged ROC 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 2a aggregate all folds
    tpr_cv = {i: [] for i in range(n_classes)}
    fpr_cv = {i: mean_fpr for i in range(n_classes)}
    auc_cv = {i: [] for i in range(n_classes)} # used to compute std
    fpr_cv['micro'], tpr_cv['micro'], auc_cv['micro'] = [], [], []

    # 3. Per-class ROC 
    tprs_pc = []

    classifier = resolve_model() 
    # test_classifier()

    myCV = StratifiedKFold(n_splits=n_folds)
    average_precisions = {c: [] for c in ['micro', 'macrio'] + range(n_classes)} 
    for ifold, (train, test) in enumerate(myCV.split(X, y)):  # y: binarized labels
        try: 
            # customize this 
            classifier.fit(X[train], y[train])
            y_pred = classifier.predict_proba(X[test]) # y_pred is a 2D array

            # y_pred_label = classifier.predict(X[test])
            # scores.append(classifier.score(X[test], y[test]))  # average accuracy
        except Exception, e: 
            # print('+ error: %s' % e)
            print('   + type(X):%s' % type(X))
            print('   + type(train): %s, examples: %s' % (type(train), train))
            raise ValueError, e 

        y_test = lb.transform(y[test])  # binarize | cf: le.transform(y[test])

        # # per-class FPR and TPR
        # test_class(y_test, y_pred) # i <- random
        for i in range(n_classes):  # this makes sense only when y is binarized y_test[:, i] => ith class
            # idx = search_label(y_test, i)
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i]) # customize this
            roc_auc[i] = auc(fpr[i], tpr[i])  # area under the curve

            # aggregate values from all folds
            # fpr_cv[i].append(fpr[i])
            # tpr_cv[i].append(tpr[i])
            # auc_cv[i].append(roc_auc[i])

            tpr_cv[i].append(interp(mean_fpr, fpr[i], tpr[i]))
            tpr_cv[i][-1][0] = 0.0
            auc_cv[i].append(roc_auc[i])

        # Compute micro-average ROC curve and ROC area
        # [memo] ok to use binarized labels in roc_curve()? 
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())  # true label vs prediction

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # fpr_cv["micro"].extend(fpr["micro"]); tpr_cv["micro"].extend(tpr["micro"])
        # auc_cv["micro"].append(roc_auc["micro"])

        tprs.append(interp(mean_fpr, fpr['micro'], tpr['micro']))
        tprs[-1][0] = 0.0

        average_precisions['micro'].append(average_precision_score(y_test.ravel(), y_pred.ravel()))  # micro-averaged precision score in i-th fold

        # roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc["micro"])
        plt.plot(fpr['micro'], tpr['micro'], lw=1, alpha=0.3,
                    label='ROC fold %d (AUC = %0.2f)' % (ifold, roc_auc['micro']))

        # per-class 
        # tprs_pc[i].append(interp(mean_fpr, fpr[i], tpr[i]))

    # plot micro-average ROC curve as a function of (tprs, aucs, mean_fpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # wrt micro-averaged
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean micro-average ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # standard deviation across k-folds
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    title_msg = kargs.get('title', "ROC curve")
    plt.title(title_msg)
    plt.legend(loc="lower right")
    # plt.show()

    # save plot    
    # fpath = kargs.get('fpath', None)
    outputdir = resolve_output()
    identifier = kargs.get('identifier', 'generic')
    fname = 'roc-microAvg_std-%s.tif' % identifier
    # if fpath is None: fpath = 'roc-microAvg_std-%s.tif' % kargs.get('identifier', 'generic')
    saveFig(plt, fpath=os.path.join(outputdir, fname))

    ### per-class plot + micro + macro 
    
    # test_auc_cv_metrics() # fpr_cv, trp_cv, auc_cv
    # first, merge values from k folds 
    mean_fpr_cv = fpr_cv
    mean_tpr_cv = {i: [] for i in range(n_classes)}
    mean_auc_cv = {i: auc_cv[i] for i in range(n_classes)}
    for i in range(n_classes):
        # fpr_cv[i] = np.linspace(0, 1, 100) # np.concatenate(fpr_cv[i]) 
        mean_tpr_cv[i] = np.mean(tpr_cv[i], axis=0) # mean_tpr of ith-class; np.concatenate(tpr_cv[i])
        mean_tpr_cv[i][-1] = 1.0
        mean_auc_cv[i] = auc(mean_fpr_cv[i], mean_tpr_cv[i])  # this is just a list of AUCs

    # consolidate micro average data across k folds
    mean_fpr_cv['micro'] = mean_fpr
    mean_tpr_cv['micro'] = mean_tpr
    mean_auc_cv["micro"] = mean_auc # auc(fpr_cv["micro"], tpr_cv["micro"])  # "grand" average

    # fpath = 'roc-avg-%s.tif' % kargs.get('identifier', 'generic')
    # title_msg = '?'
    # [params] title

    # evalMulticlassROCCV is similar to evalMulticlassROCCurve in runROCMulticlass()
    res = evalMulticlassROCCV(y, fpr=mean_fpr_cv, tpr=mean_tpr_cv, roc_auc=mean_auc_cv, 
            outputdir=outputdir, identifier=identifier, class_lookup=lookup, 

            # if plot_selected_classes is set to True, then only show ROC curves of the target_labels
            target_labels=kargs.get('target_labels', []), 

            plt_style=kargs.get('plt_style', None), 
            plot_selected_classes=kargs.get('plot_selected_classes', False)) 
    
    # averaged micro-averaged precision score
    res['average_precision'] = res['precision'] = np.mean(average_precisions['micro'])

    return res

def evalMulticlassROCCV(y, fpr, tpr, roc_auc, **kargs):
    """
    Params
    ------
    label: a particular class labels

    (fpr, tpr, roc_auc) are dictionaries with (numeric) class labels as keys
    fpr: 
    tpr
    roc_auc

    fpath: path to the output file 
    identifier: 
    title: title text 

    target_labels: if specified, only show ROC curve for these classes

    Output
    ------
    1. (averaged) ROC plot
    'roc-avg-%s.tif' % kargs.get('identifier', 'generic')

    Memo
    ----
    1. multiple redundant curves in per-class ROC plot

    """ 
    def sample_classes(n=3): 
        if n_classes > 3: 
            intervals = ranges(n_classes, n)
        return [int(interval[1]) for interval in intervals]

    def ranges(N, nb):
        step = N / nb
        return [(round(step*i), round(step*(i+1)))for i in range(nb)]
    
    def binarize_labels(y_unique): 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        # class_labels = np.unique(y)
        lb.fit(y_unique)
        # yp = le.transform(y)
        label_pairs = zip(lb.transform(y_unique), y_unique)
        lookup = dict([(tuple(l), cl) for l, cl in label_pairs])
        return dict(lookup)
    def to_class_name(i):
        c = [0] * n_classes
        c[i] = 1 
        # print('  + lookup:\n%s\n' % lookup)
        return lookup[tuple(c)]
    def index_class_names(): 
        labels = kargs.get('target_labels', [])
        indices = []
        if not labels: 
            pass
        else: 
            adict = {}
            for i in range(n_classes): 
                adict[to_class_name(i)] = i

            indices = []
            for label in labels: 
                try: 
                    indices.append(adict[label])
                except: 
                    raise ValueError, "Could not find label %s among:\n%s\n" % (label, ' '.join(str(cl) for cl in adict.keys()))
        return indices
    def test_class_names(): 
        for i in range(n_classes): 
            print("  + %d-th class: %s" % (i, to_class_name(i)))
        return
    def test_auc_cv_metrics(): # fpr_cv, trp_cv, auc_cv
        print('>>> test_auc_cv_metrics >>>')
        print('  + dimension test ')

        class_n = 1
        # print('  + fpr: %s (dim=%s)' % (fpr[class_n], str(fpr[class_n].shape)))
        # print('  + tpr: %s (dim=%s)' % (tpr[class_n], str(tpr[class_n].shape)))
        print('  + len(roc_auc):%d =?= n_classes (+{micro, macro}):%d+2' % (len(roc_auc), n_classes))
        print('  + roc_auc (classes+macro+micro):\n%s\n' % roc_auc)

        colors = cycle([['aqua', 'turquoise', 'cornflowerblue', 'teal', 'green', 'darkorange']])
        class_to_aucs = {}
        for i, color in zip(range(n_classes), colors):
            mean_auc = roc_auc[i]
            class_to_aucs[i] = mean_auc
            print("  + class name: %s, mean_auc: %f" % (to_class_name(i), mean_auc))
            print('  + ROC curve of class {0} (area = {1:0.2f})'.format( to_class_name(i), mean_auc ))
        
        # div(message='test_auc_cv_metrics> summary of AUCs vs classes ...', symbol="%")
        print("test_auc_cv_metrics> summary of AUCs vs classes ...")
        # auc_values = [class_to_aucs[i] for i in range(n_classes)]
        # min_auc, max_auc = min(auc_values), max(auc_values)
        ranked_auc = sorted([(to_class_name(i), auc) for i, auc in class_to_aucs.items()], key=lambda x:x[1], reverse=False)
        
        res = {}
        print('  + min | class=%s, auc=%f' % (ranked_auc[0][0], ranked_auc[0][1]))
        print('  + max | class=%s, auc=%f' % (ranked_auc[-1][0], ranked_auc[-1][1]))
        res = {'min': (ranked_auc[0][0], ranked_auc[0][1]), 
               'max': (ranked_auc[-1][0], ranked_auc[-1][1])}
        
        print('  + Ranked AUC scores: ')
        for cl, auc in ranked_auc: 
            print("    + class=%s, auc=%f" % (cl, auc))
        ### plot 
        # [todo] continue to work plotUtils.py

        print("  + micro vs macro? %f: %f | use micro-averaging as the value for 'roc_auc'" % (roc_auc["micro"], roc_auc["macro"]))
        res['roc_auc'] = res['auc_roc'] = roc_auc["micro"] 
        res['macro'] = roc_auc["macro"]
        res['micro'] = roc_auc["micro"]

        return res

    from itertools import cycle
    from system.utils import div
    from sklearn import preprocessing
    from sklearn.metrics import precision_recall_curve

    # plot style
    plt_style = kargs.get('plt_style', None)
    if plt_style is not None: 
        print('runCVROCMulticlass> Plot ROC curves using style: %s' % plt_style)
        try: 
            plt.style.use(plt_style)  # 'seaborn'
        except: 
            print('runCVROCMulticlass> Warning: Invalid plot style %s. Using ggplot by default' % plt_style)
            plt.style.use('ggplot')

    class_labels = np.unique(y)
    n_classes = len(class_labels)
    lookup = kargs.get('class_lookup', binarize_labels(class_labels)) # lookup: binarized labels -> canonical labels (string)
    assert len(lookup.keys()[0]) == n_classes, "n_one_hot: %d but n_classes: %d" % (len(lookup.keys()[0]), n_classes)
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.clf()
    plt.figure()

    # micro
    # [memo] below is not correct, need to apply interp()
    # if not roc_auc.has_key("micro"): roc_auc["micro"] = auc(fpr["micro"], tpr["micro"]) 
    # % format(n_classes, roc_auc["micro"]),
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-averaging (area = {0:0.2f})'  # alternatively, 'ROC curve via micro-averaging (area = {0:0.2f})
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # macro
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-averaging (area = {0:0.2f})'  # 'ROC curve via macro-averagig (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # test_class_names()
    res = test_auc_cv_metrics()  # resturn max performance and min performance; key: 'min': label -> auc, 'max': label -> auc

    # per class 
    lw = 2
    colors = cycle(['aqua', 'turquoise', 'cornflowerblue', 'teal', 'green', 'darkorange']) # 'darkorange'

    tPlotSelectedClasses = kargs.get('plot_selected_classes', False)
    if tPlotSelectedClasses and n_classes > 3:
        # specify or just pick any three classes
        cids = index_class_names() # option: target_labels 
        if not cids: 
            cids = sorted(random.sample(range(n_classes), min(n_classes, 3))) # sample_classes(n=3)
            print('evalMulticlassROCCV> Warning: plot random classes:\n%s\n' % [to_class_name(cid) for cid in cids])
        n_iter = 0
        for i, color in zip(cids, colors):
            mean_auc = roc_auc[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='Class {0} (area = {1:0.2f})'.format(to_class_name(i), mean_auc))
            n_iter += 1
            print('  + complete curve (class=%d, n_iter=%d)' % (i, n_iter))  
    else: 
        # roc_auc[i] has auc from k folds => a list
        for i, color in zip(range(n_classes), colors):
            mean_auc = roc_auc[i] # auc(sorted(fpr[i]), sorted(tpr[i]))
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='Class {0} (area = {1:0.2f})'.format(to_class_name(i), mean_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    
    title_msg = kargs.get('title', 'Extension of ROC to multiclass CKD stages (averaged over %d classes)' % n_classes)
    plt.title(title_msg)
    plt.legend(loc="lower right")
    # plt.show()

    # save plot    
    prefix = os.getcwd()
    for opt in ('prefix', 'outputdir', ):  # compatibility
        if kargs.has_key(opt): 
            prefix = kargs[opt]
            break  
    fname_default = 'roc-multiclass-cv-%s.tif' % kargs.get('identifier', 'generic')
    fname = fname_default # kargs.get('outputfile', fname_default)
    saveFig(plt, fpath=os.path.join(prefix, fname))

    return res

def plotROCPerClass(label, fpr, tpr, roc_auc, **kargs): 
    """
    Params
    ------
    label: a particular class labels

    (fpr, tpr, roc_auc) are dictionaries with (numeric) class labels as keys
    fpr: 
    tpr
    roc_auc

    fpath: path to the output file 
    identifier: 
    title: title text 

    """
    plt.clf()  # clear previous plots

    plt.figure()
    lw = 2
    plt.plot(fpr[label], tpr[label], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[label])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    title_msg = kargs.get('title', "ROC curve (class=%s)" % label)
    plt.title(title_msg)
    plt.legend(loc="lower right")

    fpath = kargs.get('fpath', None)
    if fpath is None: fpath = 'roc-%s-L%s.tif' % (kargs.get('identifier', 'generic'), label)
    saveFig(plt, fpath=fpath)

    return 

def runCVROC(X, y, **kargs): 
    """

    Params
    ------
    classifier: assuming that model selection is completed (so that the input classifier has the optimal hyperparams)

    """
    def resolve_output():
        outputdir = os.getcwd()
        for dr in ('outputdir', 'prefix', ):  
            if kargs.has_key(dr): 
                outputdir = kargs[dr]
        assert os.path.exists(outputdir), "Invalid output dir: %s" % outputdir
        return outputdir
    from scipy import interp
    from itertools import cycle
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    # from sklearn.neural_network import MLPClassifier

    n_folds = kargs.get('n_folds', 5)
    random_state = np.random.RandomState(0)

    classifier = kargs.get('classifier', None)
    if classifier is None: 
        classifier = svm.SVC(kernel='linear', probability=True,
                             random_state=random_state)
    # condition: the classifier has the optimal hyperparameter setting
    print('runCVROC> input classifier:\n%s\n' % classifier)

    plt.clf()  # clear previous plots

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(n_splits=n_folds)
    for i, (train, test) in enumerate(cv.split(X, y)):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (%s)' % kargs.get('title', 'CKD'))  # (): something add to title
    plt.legend(loc="lower right")
    # plt.show()
    
    # output dir
    # 1. provide the path directly (fpath)
    # 2. provide prefix (directory) and an identifier
    prefix = resolve_output()
    identifier = kargs.get('identifier', 'generic')
    fpath = kargs.get('fpath', os.path.join(prefix, 'roc-%s.tif' % identifier))
    # if fpath is None: fpath = 'roc-%s.tif' % kargs.get('identifier', 'generic')
    saveFig(plt, fpath=fpath)

    return

# helper function 
def format_list(alist, mode='h', sep=', ', n_pad_space=0):  # horizontally (h) or vertially (v) display a list 
    if mode == 'h': 
        s = sep.join([str(e) for e in alist])  
    else: 
        s = ''
        spaces = ' ' * n_pad_space
        for e in alist: 
            s += '%s%s\n' % (spaces, e)
    return s

def deep_classify(model, data, **kargs):
    """

    Input
    -----
    model: compiled deep NN model
    data: input data dictionary 
          keys: 
             X, y | 
             X_train, y_train, X_test, y_test

    Memo
    ----
    
    Float between 0 and 1. Fraction of the training data to be used as validation data. 
    The model will set apart this fraction of the training data, will not train on it, 
    and will evaluate the loss and any model metrics on this data at the end of each epoch. 
    The validation data is selected from the last samples in the x and y data provided, before shuffling.
    """
    from keras.utils import np_utils
    from keras.callbacks import Callback, EarlyStopping
    from dnn_utils import auc_roc
    
    my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]

    if data.has_key('X') and data.has_key('y'):     
        X, y = data['X'], data['y']
        model.fit(X, y,
              validation_split=kargs.get('validation_split', 0.3),
              shuffle=kargs.get('shuffle', False),
              batch_size=kargs.get('batch_size', 10), epochs=kargs.get('epochs', 10), verbose=1,
              callbacks=my_callbacks) 
    else: 
        X_train, y_train = data['X_train'], data['y_train']
        if data.has_key('X_test') and data.has_key('y_test'): 
            X_test, y_test = data['X_test'], data['y_test']
            model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=kargs.get('batch_size', 10), epochs=kargs.get('epochs', 10), verbose=1,
                    callbacks=my_callbacks)
    return 

def analyze_performance(scores, n_resampled=100, ci_low=0.05, ci_high=0.95):
    from sampler import sampling
    # import collections
    finalRes = {}
    for mt, values in scores.items():  # foreach metric and its performance scores (over multiple trials)

        # min, max computed wrt AUC scores
        if mt in ('min', 'max'):   # stage, score
            # find out majoriy
            counter = collections.Counter([label for label, _ in values])
            print('analyze_perf> %s => \n%s\n' % (mt, counter))
            target = counter.most_common(1)[0][0]
            
            target_values = [s for label, s in values if label == target]
            assert len(target_values) > 0, "No scores found for label: %s" % target
            bootstrapped = sampling.bootstrap_resample(target_values, n=n_resampled)
            finalRes[mt] = (target, np.mean(bootstrapped))

            ret = sampling.ci4(bootstrapped, low=0.05, high=0.95)  # keys: ci_low, ci_high, mean, median, se/error
            finalRes['%s_err' % mt] = (ret['ci_low'], ret['ci_high'])
        else: 

            bootstrapped = sampling.bootstrap_resample(values, n=n_resampled)
            finalRes[mt] = np.mean(bootstrapped)

            ret = sampling.ci4(bootstrapped, low=ci_low, high=ci_high)  # keys: ci_low, ci_high, mean, median, se/error
            finalRes['%s_err' % mt] = (ret['ci_low'], ret['ci_high']) 
    return finalRes # keys: X in {min, max, micro, macro, loss/neg_log_loss, acc/accuracy, auc_roc/roc_auc, f1_micro, f1_macro} and X_err 

def modelEvaluateBatch(X, y, **kargs): 
    """
    Batch version of and modelEvaluate(). 

    Memo
    ----
    1. Analogous to dnn_utils.modelEvaluateBatch()
       in which the return value is a dictionary of the following keys: 
       
       X and X_err, where X in {min, max, micro, macro, loss/neg_log_loss, acc/accuracy, auc_roc/roc_auc, f1_micro, f1_macro} 

    """

    from sampler import sampling

    nTrials = kargs.get('n_trials', 1)
    if not kargs.has_key('use_cv'): kargs['use_cv'] = True # cross validation by default

    scores = {}
    for i in range(nTrials): 
    
        # similar to dnn_utils.modelEvaluate(X, y, **kargs)  # a dictionary with keys: {min, max, micro, macro, loss, acc, auc_roc} 
        res = modelEvaluate(X, y, **kargs)  # must provide either trained_model or model (untrained)
        for mt, score in res.items(): 
            if not scores.has_key(mt): scores[mt] = []
            scores[mt].append(score)

    n_resampled = 100
    res = analyze_performance(scores, n_resampled=n_resampled)

    return res

def runROCMulticlass(X, y, **kargs):  # no CV
    """

    **kargs
    -------
    classifier
    random_state 
    identifier: used for file ID

    ratios 
    test_size
    

    Reference
    ---------
    1. tpheno/demo/plot_roc.py

    Memo
    ----
    1. tentative use: 
       runROCMulticlass(X, y, classifier=classifier, prefix=outputdir, identifier=identifier, target_labels=target_labels)

    Related
    -------
    runCVROCMulticlass()    # with cross validation

    """
    def binarize_label(y): 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        lb.fit(y)
        # yp = le.transform(y)
        # label_pairs = zip(lb.transform(class_labels), class_labels)
        # lookup = dict([(tuple(l), cl) for l, cl in label_pairs])

        return lb # dict(lookup)
    def train_dev_test_split(X, y, ratios=[0.8, 0.1, 0.1, ]): 
        assert len(ratios) in (2, 3, )

        n0 = X.shape[0]
        ridx = sampling.splitDataPerClass(y, ratios=ratios)  # works by permutating data index via y
        assert len(ridx) == 3

        X, y = X[ridx[0]], y[ridx[0]]
        X_dev, y_dev = X[ridx[1]], y[ridx[0]]
        X_test, y_test = X[ridx[2]], y[ridx[0]]

        n1 = X.shape[0]
        print('split> sample size %d (r=%s)=> (train: %d, dev: %d, test: %d)' % (n0, ratios, n1, X_dev.shape[0], X_test.shape[0]))
        assert X.shape[0]+X_dev.shape[0]+X_test[0] == X.shape[0]

        return [(X, y), (X_dev, y_dev), (X_test, y_test)]
    def describe_classifier(clf):
        myName = '?'
        try: 
            myName = clf.__class__.__name__ 
        except: 
            print('warning> Could not access __class__.__name__ ...')
            myName = str(clf)
        print('... classifier name: %s' % myName)
        return
    def det_test_size(): # used for train_test_split (without dev set)
        r = 0.3
        if kargs.has_key('test_size'): 
            r = kargs['test_size']
        else: 
            ratios = kargs.get('ratios', [0.7, ])  # training set ratio 
            r = 1-sum(ratios)
            assert r > 0.0, "test_size ratio < 0.0: %f" % r
        print('runROCMulticlass> test set ratio: %f' % r)
        return r
    def is_binarized(y):
        return len(y.shape) >=2 and y.shape[-1] == n_classes 
    def get_model_dir(dir_type='nns_model'): 
        
        # alternatively, use seqConfig.tsHanlder.get_nns_model_dir()
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('runROCMulticlass> model dir: %s' % modelPath)
        return modelPath  
    def resolve_model(): 
        classifier = None
        for opt in ['model', 'classifier', ]: 
            if kargs.has_key(opt):
                classifier = kargs[opt]
                break 
        if classifier is None: 
            print('runROCMulticlass> Warning: No classifier provided. Use default ...')
            classifier = svm.SVC(kernel='linear', probability=True, random_state=0)  # default 
        return classifier

    from sampler import sampling
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # import seqConfig

    y_uniq = np.unique(y)
    n_classes = len(y_uniq)
    print('runROCMulticlass> find %d unique labels:\n%s\n' % (n_classes, format_list(y_uniq)))
    lb = binarize_label(y_uniq) 
  
    # NNS-based model has its own fit() and is expected to have been trained separately prior to this subroutine
    trained_model = kargs.get('trained_model', None)  
    X_test = y_test = y_score = None
    if trained_model is None: 
        # shuffle and split training and test sets
        random_state = kargs.get('random_state', 53)
        # ratios = kargs.get('ratios', [0.7, ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=det_test_size(), random_state=random_state)

        # samples = train_dev_test_split(X, y, ratios=kargs.get('ratios', [0.8, 0.1, 0.1, ]))
        # X_train, y_train = samples[0]  # training examples
        # X_dev, y_dev, X_test, y_test = samples[1], samples[2]  # dev and test sets 

        # untrained classifier
        classifier = resolve_model()  # precedence: model, classifier, default
        try: 
            # customize this 
            # y_train = lb.transform(y_train) 
            # print('diagnosis> y_train:%s' % y_train[:5])
            if kargs.get('binarize_label', False): y_train = lb.transform(y_train)

            classifier.fit(X_train, y_train)   # y_train is not binarized
            # err_train = classifier.score(X_train, y_train)

            y_score = classifier.predict_proba(X_test) # y_score  is a 2D array
            
            # predict_mine = np.where(rf_predict_probabilities > 0.21, 1, 0)
            y_pred = classifier.predict(X_test)
            print('diagnosis> dim(y_score):%s' % str(y_score.shape))
        except Exception, e: 
            # print('+ error: %s' % e)
            print('diagnosis> dim(X_train):%s, dim(y_train):%s' % (str(X_train.shape), str(y_train.shape)))
            describe_classifier(classifier)
            raise ValueError, e 
    else:   # pre-trained model
        X_test, y_test = X, y  
        # fit() has been completed prior ot the call
        print('info> given pre-trained model ...')
        y_score = trained_model.predict_proba(X_test)
        
        y_pred = trained_model.predict(X_test)  # threshold 0.5 by default
        print('diagnosis> dim(y_score):%s' % str(y_score.shape))

    # if y_test has been binarized, then don't repeat it
    if not is_binarized(y_test): y_test = lb.transform(y_test)

    # evalMulticlassROCCurve() is similar to evalMulticlassROCCV() in runCVROCMulticlass()
    res = evalMulticlassROCCurve(y_test, y_score, classes=y_uniq, binarizer=lb, **kargs)  # outputfile, outputdir, identifier

    # also compute accuracy
    # res['acc'] = accuracy_score(y_test, y_pred)  # args: y_true, y_pred

    # if kargs.get('general_evaluation', True):   
    #     default_metrics = ['neg_log_loss', 'accuracy', 'f1_micro', 'f1_macro', ] # "roc_auc" does not support multiclass (use plotROC)

    #     # [todo] use train-test split
    #     #        other params: take_average
    #     res0 = modelEvaluate(X, y, classifier, metrics=kargs.get('metrics', default_metrics), n_folds=5)
    #     res.update(res0)

    # micro-averaged precision score
    res.update(evalMulticlassPredisionRecall(y_test, y_score, classes=y_uniq, **kargs))

    return res

def evalMulticlassPredisionRecall(y_test, y_score, classes, **kargs):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    res = {}  # output
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    n_classes = len(classes)
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(n_classes, average_precision["micro"]))

    res['average_precision'] = res['precision'] = average_precision["micro"]

    ### plot micro-averaged precision recall curve 
    plt.clf()
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                    color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    n_stages = 5 # [hardcode]
    plt.title('Average precision score, micro-averaged over all classes (%d CKD stages + Control): AP={0:0.2f}'
        .format(n_stages, average_precision["micro"]))

    # save plot  
    prefix = os.getcwd() # outputdir 
    for opt in ('outputdir', 'prefix', ): # compatibility
        if kargs.has_key(opt):  
            prefix = kargs[opt]
            break
    fname_default = 'precision_recall-multiclass-%s.tif' % kargs.get('identifier', 'generic')
    fname = fname_default # kargs.get('outputfile', fname_default)
    saveFig(plt, fpath=os.path.join(prefix, fname))    
    return res

def evalMulticlassROCCurve(y_test, y_score, classes, **kargs): 
    """

    Input
    -----
    y_test/y_true: binarized y from the test set
    y_score/y_predict: predicted values of y
    classes: unique classes (so that we can fit a LabelBinarizer)

    """
    def sample_classes(n=3): 
        if n_classes > 3: 
            intervals = ranges(n_classes, n)
        return [int(interval[1]) for interval in intervals]
    def ranges(N, nb):
        step = N / nb
        return [(round(step*i), round(step*(i+1)))for i in range(nb)]
    
    def binarize_label0(): 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        # class_labels = np.unique(y)
        lb.fit(class_labels)
        # yp = le.transform(y)
        label_pairs = zip(lb.transform(class_labels), class_labels)
        lookup = dict([(tuple(l), cl) for l, cl in label_pairs])
        return dict(lookup)
    def to_class_name0(i):
        c = [0] * n_classes
        c[i] = 1 
        # print('  + lookup:\n%s\n' % lookup)
        return lookup[tuple(c)]

    def binarize_label(y_uniq): 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        class_labels = np.unique(y_uniq)
        lb.fit_transform(class_labels)
        # print('binarizer> fit %d labels' % len(class_labels))
        return lb # dict(lookup)
    def to_class_name(i, binarizer):
        
        # one hot encode
        c = [0] * n_classes  # one hot encode
        c[i] = 1 
        # print('  + class #%d => %s' % (i, np.array(c)))
        # print('  + inv: %s' % binarizer.inverse_transform(np.array([c])))
        # return lookup[tuple(c)]
        return binarizer.inverse_transform(np.array([c]))[0]
    def index_class_names(binarizer): 
        labels = kargs.get('target_labels', [])
        indices = []
        if not labels: 
            pass
        else: 
            adict = {}
            for i in range(n_classes): 
                adict[to_class_name(i, binarizer)] = i

            indices = []
            for label in labels: 
                try: 
                    indices.append(adict[label])
                except: 
                    raise ValueError, "Could not find label %s among:\n%s\n" % (label, ' '.join(str(cl) for cl in adict.keys()))
        return indices
    def test_class_names(binarizer): 
        for i in range(n_classes): 
            print("  + %d-th class: %s" % (i, to_class_name(i, binarizer)))
        return
    def test_auc_cv_metrics(binarizer): # fpr_cv, trp_cv, auc_cv
        print('>>> test_auc_cv_metrics >>>')
        print('  + dimension test ')

        class_n = 1

        res = {}  # output

        # print('  + fpr: %s (dim=%s)' % (fpr[class_n], str(fpr[class_n].shape)))
        # print('  + tpr: %s (dim=%s)' % (tpr[class_n], str(tpr[class_n].shape)))
        print('  + len(roc_auc):%d =?= n_classes (+{micro, macro}):%d+2' % (len(roc_auc), n_classes))
        # print('  + roc_auc:\n%s\n' % roc_auc)

        class_to_aucs = {}
        for i in range(n_classes):
            class_name = to_class_name(i, binarizer)
            mean_auc = roc_auc[i]
            class_to_aucs[i] = mean_auc
            print("  + class name: %s, mean_auc: %f" % (class_name, mean_auc))
            print('  + ROC curve of class {0} (area = {1:0.2f})'.format( to_class_name(i, binarizer), mean_auc ))
        
        # div(message='test_auc_cv_metrics> summary of AUCs vs classes ...', symbol="%")
        print("test_auc_cv_metrics> summary of AUCs vs classes ...")
        # auc_values = [class_to_aucs[i] for i in range(n_classes)]
        # min_auc, max_auc = min(auc_values), max(auc_values)
        ranked_auc = sorted([(to_class_name(i, binarizer), auc) for i, auc in class_to_aucs.items()], key=lambda x:x[1], reverse=False)
        
        
        print('  + min | class=%s, auc=%f' % (ranked_auc[0][0], ranked_auc[0][1]))
        print('  + max | class=%s, auc=%f' % (ranked_auc[-1][0], ranked_auc[-1][1]))
        print('  + micro auc=%f | macro auc=%f' % (roc_auc['micro'], roc_auc['macro']))

        res = {'min': (ranked_auc[0][0], ranked_auc[0][1]),      
               'max': (ranked_auc[-1][0], ranked_auc[-1][1])}   # (class, score)
        res['micro'], res['macro'] = roc_auc['micro'], roc_auc['macro']
        res['roc_auc'] = res['auc_roc'] = res['micro']
        # res['auc'] = res['micro']  # use micro-averaged auc as the standard (which takes into account unbalanced labels better)
        
        print('  + Ranked AUC scores: ')
        for cl, auc in ranked_auc: 
            print("    + class=%s, auc=%f" % (cl, auc))
        print('  => %s' % ranked_auc)
        ### plot 
        # [todo] continue to work plotUtils.py

        return res

    from sklearn import preprocessing
    from itertools import cycle
    from system.utils import div

    # import plotly.plotly as py
    # import plotly.graph_objs as go

    n_classes = y_test.shape[1]
    lb = kargs.get('binarizer', binarize_label(classes)) # lookup: binarized labels -> canonical labels (string)

    y_true, y_pred = y_test, y_score
    if n_classes is None: 
        n_classes = y_true.shape[1]
        print('evalMulticlassROCCurve> %d classes detected.' % n_classes)

    # Compute ROC curve and ROC area for each class
    plt.clf()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 1. Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plotting 
    # Plot ROC curves for the multiclass problem

    # 2. Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Return max performance and min performance; key: 'min': label -> auc, 'max': label -> auc
    res = test_auc_cv_metrics(lb)  

    # Plot all ROC curves
    lw = 2
    # cycledColors: ['aqua', 'turquoise', 'cornflowerblue', 'teal', 'green', 'darkorange']
    colors = cycle(['aqua', 'turquoise', 'cornflowerblue', 'teal', 'green', 'darkorange']) # 'darkorange', 'navy'

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(to_class_name(i, lb), roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()

    tPlotSelectedClasses = kargs.get('plot_selected_classes', False)
    if tPlotSelectedClasses and n_classes > 3:
        # specify or just pick any three classes (to prevent from cluttering the graphic)
        cids = index_class_names(lb) # option: target_labels 
        if not cids: 
            cids = sorted(random.sample(range(n_classes), min(n_classes, 3))) # sample_classes(n=3)
            print('evalMulticlassROCCurve> Warning: plot random classes:\n%s\n' % [to_class_name(cid, lb) for cid in cids])
        n_iter = 0
        for i, color in zip(cids, colors):
            mean_auc = roc_auc[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(to_class_name(i, lb), mean_auc))
            n_iter += 1
            print('  + complete curve (class=%d, n_iter=%d)' % (i, n_iter))  
    else: 
        # roc_auc[i] has auc from k folds => a list
        # cids = index_class_names(lb) # option: target_labels
        for i, color in zip(range(n_classes), colors):
            mean_auc = roc_auc[i] # auc(sorted(fpr[i]), sorted(tpr[i]))
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(to_class_name(i, lb), mean_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    
    title_msg = kargs.get('title', 'Multiclass ROC Evaluating 5 CKD stages + Control (n_classes=%d)' % n_classes)
    plt.title(title_msg)
    plt.legend(loc="lower right")
    # plt.show()

    # save plot  
    prefix = os.getcwd() # outputdir 
    for opt in ('outputdir', 'prefix', ): # compatibility
        if kargs.has_key(opt):  
            prefix = kargs[opt]
            break
    fname_default = 'roc-multiclass-%s.tif' % kargs.get('identifier', 'generic')
    fname = fname_default # kargs.get('outputfile', fname_default)
    saveFig(plt, fpath=os.path.join(prefix, fname))

    return res

def example_pipeline(): 
    X, y = load_data()  # load data
    y_uniq = np.unique(y)

    X, y, lb = process_data(X, y)  # binarize labels
    y_test, y_score = classify(X, y)  # classify data | input: (X, binarized y)

    outputfile = 'multiclass_roc-iris.tif'  # full path or just the file name

    # evaluation via ROC
    # classes: original class labels prior to any encoding operations
    evalMulticlassROCCurve(y_test, y_score, classes=y_uniq, outputfile=outputfile)

    return

# data generator prototype
def load_data(): 
    from sklearn import datasets
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    return (X, y)  

# data processing prototype
def process_data(X=None, y=None, classes=None, random_state=0): 
    def binarize_label(y): 
        lb = preprocessing.LabelBinarizer()  # this converts to binary codes
        class_labels = np.unique(y)
        lb.fit(class_labels)
        # yp = le.transform(y)
        # label_pairs = zip(lb.transform(class_labels), class_labels)
        # lookup = dict([(tuple(l), cl) for l, cl in label_pairs])

        return lb # dict(lookup)
    def to_class_name(i):
        c = [0] * n_classes
        c[i] = 1 
        # print('  + lookup:\n%s\n' % lookup)
        return lookup[tuple(c)]

    from sklearn import preprocessing
    from sklearn.preprocessing import label_binarize

    # Import some data to play with
    if X is None or y is None: 
        X, y = load_data()
    if classes is None: 
        n_classes = len(np.unique(y))
        classes = range(n_classes)

    # Binarize the output
    
    # y = label_binarize(y, classes=classes)
    lb = binarize_label(y)
    y = lb.transform(y)
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    return (X, y, lb)

# classifier prototype
def classify(X, y, random_state=0): 
    # from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))

    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    return (y_test, y_score)

def t_roc_multiclass(**kargs):
    def test_input(): 
        assert X.shape[0] == y.shape[0], "N=%d <> nL=%d" % (X.shape[0], y.shape[0])
        print('  + training data dimension: %d' % X.shape[1])
        ys = random.sample(y, 5)
        print('  + (n_classes=%d) example y:\n%s\n' % (n_classes, str(ys)))
        return
    def test_class(y_test, y_pred, i=None): # [params] lookup
        # assert y_pred.shape[0] == y_test.shape[0], "y_test(n=%d) <> y_pred(n=%d)" % (y_test.shape[0], y_pred.shape[0])
        print('  + y_pred dim: %s ~? n_classes=%d' % (str(y_pred.shape), n_classes))
        
        y_subset = y_test
        if i is None: i = random.sample(lookup.keys(), 1)[0] # random.choice(n_classes)
        print('  + y_pred(class=i) dim: %s ~? y_test: %d' % (str(y_pred[:, i].shape), len(y_test)))

        idx = search_label(y_subset, i) # does the class (i) exist
        assert len(idx) > 0, "label=%s is not found in this batch of test points:\n%s\n" % (lookup[i], y_subset)
        print('  +    y_subset         :\n%s\n' % y_subset)  # after y is converted to numerical rep. 
        print('  +    idx              :\n%s\n' % idx)
        print('  +    y_subset(class=i):\n%s\n' % y_subset[idx]) # after y is converted to numerical rep. 
        return
    def get_clf_name(classifier): # can be called before or after model selection (<- select_classifier)
        # classifier = kargs.get('classifier', None)
        if classifier is not None: 
            try: 
                name = classifier.__name__
            except: 
                print('info> infer classifier name from class name ...')
                # name = str(estimator).split('(')[0]
                name = classifier.__class__.__name__
        else: 
            name = kargs.get('classifier_name', None) 
            assert name is not None
        return name
    def roc_cv(X, y, classifier):    
        # fname = Graphic.getName(prefix=prefix, cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, ext_plot='tif')
        # identifier, _ = os.path.splitext(fpath)

        cohort_name = kargs.get('cohort', 'CKD')
        identifier = seqparams.makeID(params=[get_clf_name(classifier), cohort_name]) # d2v_method, seq_ptype
        
        # data/<cohort>/plot
        outputdir = Graphic.getPath(cohort=cohort_name, dir_type='plot', create_dir=True)
        # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        print('  + outputdir: %s' % outputdir)
        # [note] can use nested CV but too expensive
        # evaluation
        # ms.runCVROC(X, y, classifier=classifier, fpath=fpath)   # remember to call plt.clf() for each call (o.w. plots overlap)
        runCVROCMulticlass(X, y, classifier=classifier, prefix=outputdir, identifier=identifier)
        return 
    def search_label(y_subset, label): 
        # find the positions of 'label' in 'y_subset'
        idx = np.where(np.array(y_subset) == label)[0]
        return idx
    def transform_label(): # y 
        le = preprocessing.LabelEncoder() 
        class_labels = np.unique(y)
        le.fit(class_labels)  # fit unique labels
        yp = le.transform(y)
        assert yp.shape == y.shape

        # [test]
        nToL = zip(le.transform(class_labels), class_labels)
        print('transform_label> labels vs numeric labels ...')
        for l, cl in nToL: 
            print('  + %s ~> %s' % (l, cl))
        return (yp, dict(nToL))    

    from itertools import cycle
    import seqparams
    from seqparams import Graphic

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # inspect_labels()

    # Binarize the output
    # y = label_binarize(y, classes=[0, 1, 2])
    # n_classes = y.shape[1]

    n_classes = len(set(y))
    print('t_roc_multiclass> number of classes: %d' % n_classes)

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # input: (X, y)
    random_state = np.random.RandomState(0)
    classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    # runCVROCMulticlass(X, y, classifier=clf, prefix='roc-iris-multiclass')
    # roc_cv(X, y, classifier=classifier)  # [note] do not binarize the 'y' as an input

    # select estimator by name 
    classifier = selectOptimalEstimator('l1_logistic', X, y, **kargs)
    print('> classifier: %s (type: %s)' % (get_clf_name(classifier), classifier.__class__))
    print('> opt params: %s' % classifier.get_params())

    return

def t_roc_multiclass2(**kargs):
    X, y = load_data()

    classifier = selectOptimalEstimator('l1_logistic', X, y, **kargs)
    runROCMulticlass(X, y, model=classifier, identifier='l1log-iris', ratios=[0.7, ], target_labels=[0, 2]) 

    return

def test(**kargs): 
    # t_roc_multiclass(**kargs)

    # evaluation without CV
    t_roc_multiclass2(**kargs)

    return 

if __name__ == "__main__":
    test() 