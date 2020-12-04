# needed imports
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import os, sys, collections, re
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import pandas as pd 
from pandas import DataFrame

# temporal sequence module
from config import seq_maker_config, sys_config
from seqmaker import seqCluster as sc
from seqmaker import seqparams


######################################################################################
#
# Reference 
# ---------
#   1. SciPy dendrogram: 
#      https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
#
#
######################################################################################

# some setting for this notebook to actually show the graphs inline, you probably won't need this
# %matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

def g_data():  # g: generate
    # generate two clusters: a with 100 points, b with 50:
    np.random.seed(4711)  # for repeatability of this tutorial
    a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
    X = np.concatenate((a, b),)
    print X.shape  # 150 samples with 2 dimensions
    plt.scatter(X[:,0], X[:,1])
    # plt.show()
    y = None

    return (X, y)

def apply_ward(X, y=None): 
    """
    

    Output
    ------
    Z: The linkage matrix encoding the hierarchical clustering to render as a dendrogram

    1. Each row of Z has the format: 
          [idx1, idx2, dist, sample_count].
    """

    # generate the linkage matrix
    Z = linkage(X, 'ward')    

    return Z 

def p_test(X, y=None):

    # color specific points 
    idxs = [33, 68, 62]
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1])  # plot all points
    plt.scatter(X[idxs,0], X[idxs,1], c='r')  # plot interesting points in red again 
    idxs = [15, 69, 41]
    plt.scatter(X[idxs,0], X[idxs,1], c='y')
    
    plt.show()

    return

def fancy_dendrogram(*args, **kwargs):
    """
    Dendrogram + annotating heights at split points

    """
    max_d = kwargs.pop('max_d', None)
    print('fancy_dendrogram> final max_d: %s' % max_d)    
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')

    return ddata 

def p_dendrogram(Z, y=None): # p: plot 
    
    basedir = os.path.join(os.getcwd(), 'images') # sys_config.read('')
    if not os.path.exists(basedir): os.mkdir(basedir)
    plt.clf()
    
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    dendrogram(
            Z,
            labels = y, 
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )

    # plt.show()
    
    ext = 'tif'
    fname = 'dentrogram_test.%s' % ext 
    fpath = os.path.join(basedir, fname)
    plt.savefig(fpath, bbox_inches='tight') 
    plt.close()
    
    return

def p_truncated_dendrogram(Z, **kargs):
    # [params]
    basedir = os.path.join(os.getcwd(), 'test') # sys_config.read('')
    if not os.path.exists(basedir): os.mkdir(basedir)
    last_n_merge = kargs.get('p', kargs.get('last_n_merge', 12))

    show_leaf_counts = True
    show_cluster_size = kargs.get('show_cluster_size', show_leaf_counts)
    y = kargs.get('y', kargs.get('labels', None))

    plt.clf()

    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        labels = y, 
        truncate_mode='lastp',  # show only the last p merged clusters
        p=last_n_merge,  # show only the last p merged clusters
        show_leaf_counts=show_cluster_size,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    # plt.show()
    
    graph_mode = 'dentrogram_truncated'
    descriptor = kargs.get('meta', kargs.get('descriptor', 'test'))
    ext = 'tif'
    fname = seqparams.name_cluster_file(descriptor, **kargs) # '%s_%s.%s' % (descriptor, graph_mode, ext) 
    fpath = os.path.join(basedir, fname)

    plt.savefig(fpath, bbox_inches='tight') 
    plt.close()

    return  

def e_cophenetic(X, Z):  # e: evaluate clusters
    """

    Cophenetic Correlation Coefficient
       This (very very briefly) compares (correlates) the actual pairwise distances of all your samples 
       to those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering 
       preserves the original distances. 
    """
    c, coph_dists = cophenet(Z, pdist(X)) 
    print('info> Cophenetic Correlation Coefficient: %f' % c)
    return (c, coph_dists)
def evaluate(X, Z):
    """
    Evaluate cluster quality. 

    Metrics
    -------
    1. Cophenetic Correlation Coefficient
       This (very very briefly) compares (correlates) the actual pairwise distances of all your samples 
       to those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering 
       preserves the original distances

    """
    c, coph_dists = e_cophenetic(X, Z) 

    return

def distance_matrix(X): # pdist 
    """

    Memo
    ----
    1. metric: 
       cosine, hamming, jaccard, mahalanobis, minkowsk, correlation, 
       seuclidean: standardized euclidean

    Reference 
    ---------
    1. scipy.spatial.distance.pdist
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """
    metric = 'euclidean'  # cosine, hamming, jaccard, mahalanobis, minkowsk, correlation, seuclidean (standardized euclidean)
    M = pdist(X, metric=metric)

    return M
    

def p_dendrogram2(Z, **kargs):
    """

    Input 
    ----- 
    y: labels

    Memo
    ----
    1. meant to subsume p_dendrogram
    2. call fancy_dendrogram instead (wit height annotations)


    """
    basedir = os.path.join(os.getcwd(), 'test') # sys_config.read('')
    if not os.path.exists(basedir): os.mkdir(basedir)
    plt.clf()

    last_n_merge = kargs.get('p', kargs.get('last_n_merge', 12))
    show_leaf_counts = False
    show_cluster_size = kargs.get('show_cluster_size', show_leaf_counts)
    y = kargs.get('y', kargs.get('labels', None))

    # determine number of clusters 
    # e.g. set cut-off to 50
    max_distance = 50
    max_d = kargs.get('max_d', max_distance)  # max_d as in max_distance

    fancy_dendrogram(
        Z,
        labels = y, 
        truncate_mode='lastp',
        p=last_n_merge,
        show_leaf_counts=show_cluster_size,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
        max_d = max_d,      # plot a horizontal cut-off line
    )
    # plt.show()
    
    graph_mode = 'dentrogram'
    descriptor = kargs.get('meta', kargs.get('descriptor', '%s_%s' % ('hc', graph_mode)))
    ext = 'tif'
    fname = seqparams.name_cluster_file(descriptor, **kargs) # '%s_%s.%s' % (descriptor, graph_mode, ext) 
    fpath = os.path.join(basedir, fname)
    print('output> saving plot to %s' % fpath)
    plt.savefig(os.path.join(basedir, fname), bbox_inches='tight') 
    plt.close()

    return 

def retrieve_cluster(Z, **kargs): 
    from scipy.cluster.hierarchy import fcluster

    n_clusters = kargs.pop('n_clusters', 2)

    # now apply a given policy to define clusters 
    criterion = kargs.get('criterion', kargs.get('policy', 'distance'))
    height_cutoff = kargs.get('max_d', 50)  # let's say this gives as opt # of clusters

    # assuming that n_clusters is determined ...
    # [output] clusters example: array([2, 2, 2, 1, 1, 2 ... 1, 2], dtype=int32) 
    if criterion.startswith('dist'): 
        clusters = retrieve_cluster(Z, max_d=height_cutoff, criterion='distance')  
    elif criterion.startswith('ncluster'): # knowing number of clusters 
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
    elif criterion.startswith('inconsist'): # inconsistency method (cluster merge's h - average height)/ std
        clusters = fcluster(Z, 8, depth=10)  # [todo]


    return clusters # mapping data points to cluster ids

def p_dendrogram_plotly(X=None, y=None, Z=None, **kargs):
    """

    Memo
    ----
    1. https://community.plot.ly/t/how-does-create-dendrogram-defined-similarity/2470/2

       default setting 

       d = scs.distance.pdist(X)
       Z = sch.linkage(d, method='complete')
       P = sch.dendrogram(Z, orientation=self.orientation, labels=self.labels, no_plot=True)

    """

    import plotly.plotly as py
    import plotly.figure_factory as ff

    basedir = sys_config.read('DataExpRoot')  
    d2v_method = 'PVDM'

    graph_mode = 'dentrogram_labeled'
    descriptor = kargs.get('meta', 'test')
    ext = 'tif'
    fname = '%s_%s.%s' % (descriptor, graph_mode, ext) 

    if X is None: X = np.random.rand(10, 10)
    if y is None: 
    	names = ['Jack', 'Oxana', 'John', 'Chelsea', 'Mark', 'Alice', 'Charlie', 'Rob', 'Lisa', 'Lily']
    else: 
    	names = y

    if Z is None: 
    	Z = apply_ward(X, y=y) 
    
    fig = ff.create_dendrogram(X, orientation='left', labels=names)
    fig['layout'].update({'width':1000, 'height':1000})  # demo: 800

    fpath = os.path.join(basedir, fname)
    py.iplot(fig, filename=fpath)

    return

def t_dendrogram(**kargs):
    X, y = g_data() 
    Z = apply_ward(X, y)
    # p_dendrogram(Z)
    # p_truncated_dendrogram(Z)

    p_dendrogram2(Z)

    return

def t_hc(**kargs): 
    X, y = g_data() 
    Z = apply_ward(X, y)
    # p_dendrogram(Z)
    # p_truncated_dendrogram(Z)

    p_dendrogram2(Z)

    # now apply a given policy to define clusters 
    criterion = kargs.get('criterion', kargs.get('policy', 'distance'))

    height_cutoff = 50  # let's say this gives as opt # of clusters

    # assuming that n_clusters is determined ... 
    clusters = retrieve_cluster(Z, criterion=criterion, max_d=height_cutoff)

    # visualize clusters 
    t_visualize(clusters, **kargs)    

    return

def t_visualize(clusters, **kargs): 
    basedir = sys_config.read('DataExpRoot')  
    d2v_method = 'PVDM'

    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
    # plt.show()

    graph_mode = 'hc'
    descriptor = kargs.get('meta', d2v_method)
    ext = 'tif'
    fname = '%s_%s.%s' % (descriptor, graph_mode, ext) 
    plt.savefig(os.path.join(basedir, fname), bbox_inches='tight') 
    plt.close()

    return

# [app]
def run_hc(X, y, D, **kargs): 

    metric = 'euclidean'  # cosine, hamming, jaccard, mahalanobis, minkowsk, correlation, seuclidean (standardized euclidean)
    metrics = ['euclidean', 'seuclidean', 'cosine', 'correlation', ]
    linkage_policy = 'average'

	n_doc0 = len(D)
    assert X.shape[0] == n_doc0

    labels = set(y)
    n_labels = len(labels)

    # labels map [todo]
    lmap = kargs.get('label_map', {0: 'type 1', 1: 'type 2', 2: 'birth', })

    # verifying the data
    print('run_hc> X (dim: %s), y (dim: %s), D (dim: %s)' % (str(X.shape), str(y.shape), n_doc0))  
    print('verify> canonical labels:\n%s\n' % [lmap.get(l, 'unknown') for l in labels])

    # Z = apply_ward(X_subset, y=y_subset) 

    for metric in metrics: 

        d = pdist(X, metric=metric)
        Z = linkage(d, method=linkage_policy) # single (min), complete (max), average, weighted, centroid, ward
        print('verify> X: %s, y: %s > Z: %s' % (str(X.shape), str(y.shape), str(Z.shape)))  
        kargs['descriptor'] = 'hc_%s' % metric  # cluster method + similarity metric

        try: 
    	    p_dendrogram2(Z, y=y, last_n_merge=100, max_d=None, **kargs)
            # p_truncated_dendrogram(Z, labels=y_subset, meta=d2v_method) # [log] too many data points at 430,000
            # p_dendrogram_plotly(X, y, Z, meta=d2v_method) # only support euclidean now
        except Exception, e: 
    	    print('debug> %s' % e)
    	    print('diagnosis> running distance function wrt %s' % metric)
            d = pdist(X_subset, metric=metric)
            print('diagnosis> distance function passed ... size of d: %d' % len(d))
            Z = linkage(d, method='single')
            print('diagnosis> linkage definition passed ... dim of Z: %s' % str(Z.shape))
            p_truncated_dendrogram(Z, labels=y, **kargs) # [log] too many data points at 430,000

    return 

def t_conditon_diag(**kargs):
    """
    HC on doc vectors.
    """
    def mlabel_to_slable(y): 
    	lsep = '_'
        yp = [mlabel.split(lsep)[0] for mlabel in y] # take most frequent (sub-)label as the label
        print('verify> example single labels:\n%s\n' % yp[:10])
        return yp

    from seqmaker import seqSampling as ss
    from seqmaker import seqCluster as sc 

    basedir = sys_config.read('DataExpRoot')  
    d2v_method = 'PVDM'

    metric = 'euclidean'  # cosine, hamming, jaccard, mahalanobis, minkowsk, correlation, seuclidean (standardized euclidean)
    linkage_policy = 'complete'
    meta = kargs.get('meta', '%s-D%s_L%s' % (d2v_method, metric, linkage_policy))

    redo_clustering = True
    n_sample = 200  # still got memory error with plotly
    print('params> desired sample size: %d' % n_sample)
    
    # [input] PV-DM feature set  or just sause build_data_matrix2
    # ifile = 'fset_%s_432000_labeled.csv' % d2v_method
    # fpath = os.path.join(basedir, ifile) 
    # df = pd.read_csv(fpath, sep=',', header=None, index_col=0, error_bad_lines=True)
    # print('data> %s data dim: %s' % (d2v_method, str(df.shape))) # [log] PVDM data dim: (432000, 200)
    # labels = df.index.values
    # print('verify> example labels:\n%s\n' % labels[:10]) # [log] ['V70.0_401.9_199.1' '746.86_426.0_V72.19' '251.2_365.44_369.00' ...]

    X, y, D = sc.build_data_matrix2(**kargs)
    n_unique_mlabels = len(set(y))

    # verifying the data
    print('t_condition_diag> X (dim: %s), y (dim: %s), D (dim: %s)' % (str(X.shape), str(y.shape), len(D)))  
    print('verify> example composite labels:\n%s\n' % y[:10])
    yp = mlabel_to_slable(y)
    n_unique_slabels = len(set(yp))
    print('verify> total number of labels: %d > n_multilabels: %d > n_slabels: %d' % \
    	(len(y), n_unique_mlabels, n_unique_slabels))
    # X, y = df.values, labels

    # sample subset 
    candidates = ss.get_representative(docs=D, n_sample=n_sample, n_doc=X.shape[0], policy='rand')
    # candidates = ss.get_cluster_representative(n_sample=n_sample, n_doc=X.shape[0], force_clustering=redo_clustering)  # [dependency] ss <- sc
    assert len(candidates) == n_sample, "sample size %d != specified %d" % (len(candidates), n_sample)

    # if redo_clustering: X, y, D = sc.build_data_matrix2()
    X_subset, y_subset = X[candidates], y[candidates]
    # Z = apply_ward(X_subset, y=y_subset) 
    d = pdist(X_subset, metric=metric)
    Z = linkage(d, method=linkage_policy)
    print('verify> X_subset: %s, y_subset: %s > Z: %s' % (str(X_subset.shape), str(y_subset.shape), str(Z.shape)))  

    try: 
    	p_dendrogram2(Z, y=y_subset, meta=meta, last_n_merge=100, max_d=None)
        # p_truncated_dendrogram(Z, labels=y_subset, meta=d2v_method) # [log] too many data points at 430,000
        # p_dendrogram_plotly(X, y, Z, meta=d2v_method) # only support euclidean now
    except Exception, e: 
    	print('debug> %s' % e)
    	print('diagnosis> running distance function wrt %s' % metric)
        d = pdist(X_subset, metric=metric)
        print('diagnosis> distance function passed ... size of d: %d' % len(d))
        Z = linkage(d, method='single')
        print('diagnosis> linkage definition passed ... dim of Z: %s' % str(Z.shape))
        p_truncated_dendrogram(Z, labels=y_subset, meta=d2v_method) # [log] too many data points at 430,000

    return 

def t_condition_diag2(**kargs): 
    # load data 
    X, y = sc.build_data_matrix()
    n_unique_mlabels = len(set(y))
    print('verify> example composite labels:\n%s\n' % y[:10])

    lsep = '_'
    yp = [mlabel.split(lsep)[0] for mlabel in y] # take most frequent (sub-)label as the label
    print('verify> example single labels:\n%s\n' % yp[:10])
    n_unique_slabels = len(set(yp))

    # [log] number of unique multilabels: 264650 > unique single labels: 6234
    #       n_doc: 432000, fdim: 200
    print('verify> total number of labels: %d > number of unique multilabels: %d > unique single labels: %d' % \
    	(len(y), n_unique_mlabels, n_unique_slabels))

    return

def test(**kargs): 
    # plot dendrogram
    # t_dendrogram(**kargs)

    # run HC and plot dendrogram
    # print('> cluster the doc vec first ...')
    # sc.t_analysis()
    print('> now run HC using the generated data (X, y) ...')
    t_conditon_diag() 

    return 

if __name__ == "__main__": 
    test()