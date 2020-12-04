from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import collections

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification


from sklearn import preprocessing

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def eval_weights(y):  # compute class weights and sample weights 
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

    # Instantiate the label encoder
    y_uniq = np.unique(y)
    le = preprocessing.LabelEncoder()
    le.fit(y_uniq) # Fit the label encoder to our label series
    print('  + lookup:\n%s\n' % dict(zip(y_uniq, le.transform(y_uniq))) )

    y_int = le.transform(y)   # Create integer based labels Series
    labelToInteger = dict(zip(y, y_int)) # Create dict of labels : integer representation

    class_weight = compute_class_weight('balanced', np.unique(y_int), y_int)
    sample_weight = compute_sample_weight('balanced', y_int)
    class_weight_dict = dict(zip(le.transform(list(le.classes_)), class_weight))

    return class_weight_dict

def sample(X, y, n_samples=None, verbose=True):
    """

    Memo
    ----
    This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. 
    The folds are made by preserving the percentage of samples for each class. 

    """
    def show_proportion(train_index): 
        classes = collections.Counter(y[train_index])
        nc = sum(classes[c] for c in classes)
        print('> %s (sum=%d)' % (classes, nc))
        return 

    from sklearn.model_selection import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=53)
    if n_samples is not None: 
        train_index, test_index = list(cv.split(X, y))[0]
        if verbose: show_proportion(train_index)
        return (X[train_index], y[train_index])

        # for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    	    # classes = collections.Counter(y[train_index])
    	    # nc = sum(classes[c] for c in classes)
    	    # if verbose: print('[split #%d] %s (sum=%d)' % (i, classes, nc))
    return (X, y)

def sampleByLabels(y, n_samples=None, verbose=True, sort_index=False, random_state=53):
    def show_proportion(train_index): 
        classes = collections.Counter(y[train_index])
        nc = sum(classes[c] for c in classes)
        print('> %s (sum=%d)' % (classes, nc))
        return
    from sklearn.model_selection import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)

    if n_samples is not None: 
        X = np.array([0] * len(y)) # dummy data
        y = np.array(y)
        train_index, test_index = list(cv.split(X, y))[0]
        if verbose: show_proportion(train_index)
        if sort_index: 
        	return np.sort(train_index)
        return train_index
    return np.array(range(0, len(y)))

def sampleDataframe(df, col=None, **kargs):
    """
    
    Params
    ------
    **kargs
      n
      frac
      replace 
      weights

    Memo
    ----
    a. subsetting a datatframe according to a condition
        c_code = df['icd9']==code
        c_id = df['mrn']==mrn
        rows = df.loc[c_code & c_id] 

    """
    if col is None:  
        return df.sample(**kargs)  # params: n, frac, replace, weights 

    ### sample according to the strata defined by 'col'
    # assert col in df.columns
    
    # A. ensure each stratum gets at most N samples
    random_state = kargs.get('random_state', 53)
    tSortIndex = kargs.get('sort_index', False)
    if kargs.has_key('n_per_class'): 
        strata = df[col].unique()
        n_per_class = kargs['n_per_class']
        print('sampleDataframe> n_classes=%d, n_per_class=%d' % (len(strata), n_per_class))
        dfx = []
        for i, stratum in enumerate(strata): 
            dfp = df.loc[df[col]==stratum]
            n0 = dfp.shape[0]  # n_per_class cannot exceed n0
            dfx.append(dfp.sample(n=min(n_per_class, n0), random_state=random_state))
        df = pd.concat(dfx, ignore_index=True)
        
        assert df.shape[0] <= len(strata) * n_per_class
        return df 
    else: 
        raise NotImplementedError, "sampleDataframe> Unknown sampling mode."
    
    if tSortIndex: df = df.sort_index()
    return df

def samplePerClass(y, n_per_class=1000, verbose=True, sort_index=False, random_state=53): 
    """
    Subsample a maximum of N samples for each class. 
    Return selected indices. 
    """    
    yS = Series(y)
    labelSet = yS.unique()
    max_samples = n_per_class
    # adict = {label: [] for label in labelSet}
    idx = [] # np.array([])
    for i, label in enumerate(labelSet):
    	ySL = yS.loc[yS==label]  # a sub-Series where elements == label
    	nT = ySL.shape[0]

    	# n0 = len(idx)  # [test]
    	# idx = np.hstack((idx, ySL.sample(n=min(nT, max_samples), random_state=random_state).index.values))
        idx.extend(list(ySL.sample(n=min(nT, max_samples), random_state=random_state).index.values))
        # n1 = len(idx)   
        # assert n1 > n0
    idx = np.array(idx)
    print('samplePerClass> idx | type: %s, example:\n%s\n' % (type(idx), idx))
    if sort_index: return np.sort(idx)
    return idx

def selectPerClass(X, y, n_per_class=1000, policy='random', **kargs): 
    """
    Subsample a maximum of N samples for each class. 
    Return selected indices. 
    """    
    if policy.startswith('rand'):
        return samplePerClass(y, n_per_class=n_per_class, **kargs)

    else: # 'longest'
        yS = Series(y)
        labelSet = yS.unique()
        max_samples = n_per_class
    
        idx = [] # np.array([])
        for i, label in enumerate(labelSet):
    	    ySL = yS.loc[yS==label]  # a sub-Series where elements == label
    	    # nT = ySL.shape[0]

            sorted_idx = sorted([(j, len(X[j])) for j in ySL.index], key=lambda x:x[1], reverse=True)[:max_samples]
            idx.extend([j for j, _ in sorted_idx])

        idx = np.array(idx)
        print('selectPerClass> policy=%s | idx type: %s, example:\n%s\n' % (policy, type(idx), idx))
        if kargs.get('sort_index', True): return np.sort(idx)
    return idx

def t_subsampling(): 
    """
 
    Memo
    ----
    1. Sample a subset of the data (X, y) while retaining proportionality of the class labels (y)


    """
    from sklearn.model_selection import ShuffleSplit
    
    X, y = make_classification(n_samples=500, n_features=20, n_classes=1, n_redundant=0, n_informative=10,
    	                   # weights= [0.1, 0.1, 0.5, 0.2, 0.1], 
                           random_state=1, n_clusters_per_class=1)

    # rs = ShuffleSplit(n_splits=2, train_size=100, random_state=0)
    # i = 0
    # for train_index, test_index in rs.split(y):
    # 	classes = collections.Counter(y[train_index])
    # 	print('[split #%d] %s' % (i, classes))
    #     i += 1


    # Xp = np.array(['a'] * len(X))
    sample(X, y, n_samples=200)
    print type(sampleByLabels(y, n_samples=200, sort_index=True)) # numpy.ndarray
    sampleByLabels(list(y), n_samples=200, sort_index=True)


    return 

def test(**kargs): 

    # sampling dataset
    t_subsampling()


    return

if __name__ == "__main__": 
    test()

