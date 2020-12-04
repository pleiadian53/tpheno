# encoding: utf-8
"""
=========================================
Nested versus non-nested cross-validation
=========================================

This example compares non-nested and nested cross-validation strategies on a
classifier of the iris data set. Nested cross-validation (CV) is often used to
train a model in which hyperparameters also need to be optimized. Nested CV
estimates the generalization error of the underlying model and its
(hyper)parameter search. Choosing the parameters that maximize non-nested CV
biases the model to the dataset, yielding an overly-optimistic score.

Model selection without nested CV uses the same data to tune model parameters
and evaluate model performance. Information may thus "leak" into the model
and overfit the data. The magnitude of this effect is primarily dependent on
the size of the dataset and the stability of the model. See Cawley and Talbot
[1]_ for an analysis of these issues.

To avoid this problem, nested CV effectively uses a series of
train/validation/test set splits. In the inner loop (here executed by
:class:`GridSearchCV <sklearn.model_selection.GridSearchCV>`), the score is
approximately maximized by fitting a model to each training set, and then
directly maximized in selecting (hyper)parameters over the validation set. In
the outer loop (here in :func:`cross_val_score
<sklearn.model_selection.cross_val_score>`), generalization error is estimated
by averaging test set scores over several dataset splits.

The example below uses a support vector classifier with a non-linear kernel to
build a model with optimized hyperparameters by grid search. We compare the
performance of non-nested and nested CV strategies by taking the difference
between their scores.

.. topic:: See Also:

    - :ref:`cross_validation`
    - :ref:`grid_search`

.. topic:: References:

    .. [1] `Cawley, G.C.; Talbot, N.L.C. On over-fitting in model selection and
     subsequent selection bias in performance evaluation.
     J. Mach. Learn. Res 2010,11, 2079-2107.
     <http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf>`_

"""
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
# import matplotlib.cm as cm  # silhouette test
# import seaborn as sns

from sklearn.datasets import load_iris
# from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import os

print(__doc__)


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

#######################################################################################################################################


NUM_TRIALS = 30

# SVM
gammax = np.logspace(-5, 4, 10) # gamma: inverse of bandwidth
degreex = range(2, 10)
regC = np.logspace(-3, 3, 7) 
isProba = False

# Basic SVM options
SVM_Models = {'rbf_svm': [SVC(kernel='rbf', probability=isProba, class_weight='balanced'), {"C": regC, "gamma": gammax}], 
             'rbf_svm_auto': [SVC(kernel='rbf', gamma='auto', probability=isProba, class_weight='balanced'), {"C": regC, }], 
             'linear_svm': [SVC(kernel='linear', probability=isProba, class_weight='balanced'), {"C": regC, }], 
             'poly_svm': [SVC(kernel='poly', probability=isProba, class_weight='balanced'), {"C": regC, "degree": degreex}], 
          }

def runNestedCV(X, y, estimator=None, param_grid=None, nfold_inner=5, nfold_outer=5, n_trials=None): 
    """
    
    Params
    ------
      estimator
        classifier, param_grid = SVMModel['rbf_svm']

    """
    # from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    if estimator is None or param_grid is None: 
        print('runNestedCV> Use RBF SVM by default ...')
        estimator, param_grid = SVM_Models['rbf_svm']
    if n_trials is None: n_trials = NUM_TRIALS

    # Arrays to store scores
    non_nested_scores = np.zeros(n_trials)
    nested_scores = np.zeros(n_trials)

    # Loop for each trial
    for i in range(n_trials):
 
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
        inner_cv = KFold(n_splits=nfold_inner, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=nfold_outer, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_cv)
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    score_difference = non_nested_scores - nested_scores
    print("Average difference of {0:6f} with std. dev. of {1:6f}."
          .format(score_difference.mean(), score_difference.std()))

    return (non_nested_scores, nested_scores, score_difference)

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

def test(**kargs): 
    # Number of random trials
    # NUM_TRIALS = 30

    # Load the dataset
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    # Set up possible values of parameters to optimize over
    p_grid = {"C": [1, 10, 100],
              "gamma": [.01, .1]}
    # We will use a Support Vector Classifier with "rbf" kernel
    svm = SVC(kernel="rbf")

    non_nested_scores, nested_scores, score_difference = \
        runNestedCV(X_iris, y_iris, estimator=svm, param_grid=p_grid)
    plotComparison(non_nested_scores, nested_scores, score_difference, fpath=None)

    return

if __name__ == "__main__": 
    test()



