# print(__doc__)

import plotly.plotly as py
import plotly.graph_objs as go

import os
import numpy as np
from itertools import cycle

# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp


def load_data(): 
    from sklearn import datasets
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    return (X, y)  

def process_data(X=None, y=None, classes=None, random_state=0): 
    from sklearn.preprocessing import label_binarize

    # Import some data to play with
    if X is None or y is None: 
    	X, y = load_data()
    if classes is None: 
    	n_classes = len(np.unique(y))
    	classes = range(n_classes)

    # Binarize the output
    y = label_binarize(y, classes=classes)
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    return (X, y)

def classify(X, y, random_state=0): 
    # from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from scipy import interp
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))

    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    return (y_test, y_score)

def evalMulticlassROCCurve(y_test, y_score, n_classes=None, outpath=None, **kargs): 
    import plotly.plotly as py
    import plotly.graph_objs as go

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

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plotting 
    lw = 2
    trace1 = go.Scatter(x=fpr[2], y=tpr[2], 
                    mode='lines', 
                    line=dict(color='darkorange', width=lw),
                    name='ROC curve (area = %0.2f)' % roc_auc[2]
                   )
    trace2 = go.Scatter(x=[0, 1], y=[0, 1], 
                    mode='lines', 
                    line=dict(color='navy', width=lw, dash='dash'),
                    showlegend=False)

    layout = go.Layout(title='Receiver operating characteristic example',
                   xaxis=dict(title='False Positive Rate'),
                   yaxis=dict(title='True Positive Rate'))

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    if kargs.get('save_', True): 
        # save file
        fname = 'roc.tif'
        if outpath is not None: 
            base, fname = os.path.dirname(outpath), os.path.basename(outpath)
            if not base: base = os.getcwd()
            outpath = os.path.join(base, fname)
        else: 
        	assert os.path.exists(outpath), "Invalid output path:\n%s\n" % outpath

        # (@) Send to Plotly and show in notebook
        # py.iplot(fig, filename=fname)
        # (@) Send to broswer 
        plot_url = py.plot(fig, filename=fname)
        py.image.save_as({'data': data}, outpath)
    else: 
    	py.iplot(fig)

    return 

def test(**kargs): 

    # load data 
    X, y = load_data()
    X, y = process_data(X, y)
    y_test, y_score = classify(X, y)

    # basedir = '/Users/pleiades/Documents/work/tpheno/seqmaker/data/CKD/plot'
    outpath = 'multiclass_roc-iris.tif'  # full path or just the file name
    evalMulticlassROCCurve(y_test, y_score, outpath=outpath)

    return

if __name__ == "__main__": 
    test()


