"""
=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

Multiclass settings
-------------------

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-class classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`.

"""
# print(__doc__)

import os
import numpy as np

# plotting 
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
# plt.style.use('ggplot')
plt.style.use('seaborn')   # better default font sizes 

# import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def load_data(): 
    from sklearn import datasets
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    return (X, y)  

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

def plotROCCurve(fpr, tpr, label=2): 
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2

    plt.plot(fpr[label], tpr[label], color='darkorange',
              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[label])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return

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

def evalMulticlassROCCurve(y_test, y_score, classes, outpath=None, **kargs): 
    """

    Input
    -----
    y_test: binarized y from the test set
    y_score: predicted values of y
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

    from sklearn import preprocessing
    # import plotly.plotly as py
    # import plotly.graph_objs as go

    n_classes = y_test.shape[1]
    lb = kargs.get('binarizer', binarize_label(classes)) # lookup: binarized labels -> canonical labels (string)

    y_true, y_pred = y_test, y_score
    if n_classes is None: 
        n_classes = y_true.shape[1]
        print('evalMulticlassROCCurve> %d classes detected.' % n_classes)

    # Compute ROC curve and ROC area for each class
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

    # Plot all ROC curves
    lw = 2
    colors = cycle(['aqua', 'turquoise', 'cornflowerblue']) # 'darkorange'

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

    if n_classes > 3:
        # specify or just pick any three classes
        cids = index_class_names() # option: target_labels 
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
        for i, color in zip(range(n_classes), colors):
            mean_auc = roc_auc[i] # auc(sorted(fpr[i]), sorted(tpr[i]))
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(to_class_name(i, lb), mean_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    
    title_msg = kargs.get('title', 'Extension of ROC to multiclass setting (averaged over %d classes)' % n_classes)
    plt.title(title_msg)
    plt.legend(loc="lower right")
    # plt.show()

    # save plot    
    prefix = kargs.get('prefix', os.getcwd())
    fname = 'roc-MacroAvgPerClass-%s.tif' % kargs.get('identifier', 'generic')
    saveFig(plt, fpath=os.path.join(prefix, fname))

    return

def test(**kargs): 

    X, y = load_data()  # load data
    y_uniq = np.unique(y)

    X, y, lb = process_data(X, y)  # binarize
    y_test, y_score = classify(X, y)

    outpath = 'multiclass_roc-iris.tif'  # full path or just the file name
    evalMulticlassROCCurve(y_test, y_score, classes=y_uniq, outpath=outpath)

    return 

if __name__ == "__main__": 
    test()
