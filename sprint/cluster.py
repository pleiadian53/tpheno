# encoding: utf-8

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from collections import namedtuple
import collections

import csv
import re
import string

import sys, os, random 

from pandas import DataFrame, Series
import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

### clustering algorithms 
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN
# from sklearn.cluster import KMeans

# from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# from sklearn.cluster import AffinityPropagation

# evaluation 
from sklearn import metrics
from scipy.spatial import distance

import scipy.spatial.distance as ssd
from batchpheno import utils
import sampling_utils as sampling  # refactored from seqmaker.sequtils

# module wise configuration
import config_sprint
import learn_manifold

def map_clusters(cluster_labels, item):
    """
    Map cluster/class labels to cluster IDs. 

    Read the labels array and clusters label and return the set of words in each cluster

    Input
    -----
    cluster_labels: cluster IDs ~ oredering the training instances (X)
    item: any of inputs (X, y, D) consistent with the labeling

    Output
    ------
    A hashtable: cluster IDs -> labels
   
    e.g. 

    data = [x1, x2, x3, x4 ]  say each x_i is a 100-D feature vector
    labels = [0, 1, 0, 0]  # ground truth labels
    clusters = [0, 1, 1, 0]   say there are only two clusters 
    => cluster_to_docs: 
       {0: [0, 0], 1: [1, 0]}
    """
    cluster_to_docs = utils.autovivify_list()

    # mapping i-th label to cluster cid
    for i, cid in enumerate(cluster_labels):  # cluster_labels is listed in the order of (sequences ~ labels)
        cluster_to_docs[cid].append( item[i] )
    return cluster_to_docs

def eval_cluster(clusters, labels, cluster_to_labels=None, **kargs): 
    """
    Run intrinsic cluster evaluation under the condition that ground truths/labels are available

    Input
    -----
    clusters: list of cluster labels/IDs (whose size is the same as X, i.e. the data from which clusters
              were derived)   
    labels (y): ground-truth labels in the order of the original data set (X, y) from which clusters were 
              derived. 
    cluster_to_labels: generated by applying map_cluster()
    """
    def hvol(tb): # volume of hashtable
        return sum(len(v) for v in tb.values())
        
    y = labels
    assert y is not None, "Could not evaluate purity without ground truths given."
        
    # cluster_to_labels = map_clusters(clusters, y)
    # if 'cluster_to_labels' in locals():
    if cluster_to_labels is None: 
        cluster_to_labels = map_clusters(clusters, y)

    N = n_total = hvol(cluster_to_labels)
        
    # [output] purity_score/score, cluster_labels/clabels, ratios, fractions, topn_ratios
    res = {}

    ulabels = sorted(set(y))
    n_labels = len(ulabels)

    res['unique_labels'] = res['ulabels'] = ulabels

    maxx = []
    clabels = {}  # cluster/class label by majority vote
    for cid, labels in cluster_to_labels.items():
        counts = collections.Counter(labels)
        l, cnt = counts.most_common(1)[0]  # [policy]
        clabels[cid] = l            
        maxx.append(max(counts[ulabel] for ulabel in ulabels))

    res['purity_score'] = res['score'] = sum(maxx)/(n_total+0.0)
    res['cluster_labels'] = res['clabels'] = clabels
        
    # cluster ratios for each (unique) label 
    ratios = {ulabel:[] for ulabel in ulabels}
    fractions = {ulabel:[] for ulabel in ulabels}
    for ulabel in ulabels: # foreach unique label 
        for cid, labels in cluster_to_labels.items(): # foreach cluster (id)
            counts = collections.Counter(labels)
            r = counts[ulabel]/(len(labels)+0.0) # given a (true) label, find the ratio of that label in a cluster
            rf = (counts[ulabel], len(labels)) # fraction format
            ratios[ulabel].append((cid, r))
            fractions[ulabel].append((cid, rf))
    res['ratios'] = ratios # cluster purity ratio for each label
    res['fractions'] = fractions

    # [todo] ratios of label determined by majority votes 
    ratios_max_votes = {}  # cid -> label -> ratio
    for cid, lmax in clabels.items():  # cid, label of cluster by max vote
        ratios_max_votes[cid] = dict(res['ratios'][lmax])[cid]
    res['ratios_max_votes'] = ratios_max_votes

    # rank purest clusters for each label and find which clusters to study
    ranked_ratios = {}
    if topn_clusters is not None: 
        for ulabel in ulabels: 
            ranked_ratios[ulabel] = sorted(ratios[ulabel], key=lambda x: x[1], reverse=True)[:topn_clusters]  # {(cid, r)}
    else: 
        for ulabel in ulabels: 
            ranked_ratios[ulabel] = sorted(ratios[ulabel], key=lambda x: x[1], reverse=True) # {(cid, r)}            
    res['ranked_ratios'] = ranked_ratios
    
    return res # ['unique_labels', 'purity_score', 'cluster_labels', 'ratios', 'fractions', 'ratios_max_votes', 'ranked_ratios', ]

def merge_cluster(clusters, cids=None): 
    assert isinstance(clusters, dict)
    member = clusters.itervalues().next()
    assert hasattr(member, '__iter__')

    data = []
    if cids is None: # merge all 
        for cid, members in clusters.items(): # members can be a list or list of lists
            data.extend(members)
    else: 
        for cid in cids: 
            try: 
                members = clusters[cid]
                data.extend(members)
            except: 
                pass
    if not data:
        assert cids is not None 
        print('warning> No data selected given cids:\n%s\n' % cids)
    return data

def summarize_cluster(cluster_labels, X, y=None):  
    """
    
    Input
    -----
    cluster_labels: cluster IDs resulted from running a clustering algorithm 
    y: ground truths that come with the data X (for intrinsic evaluations)

    """
    # import scipy.spatial.distance as ssd

    clusters = map_clusters(cluster_labels, X)  # id -> {X[i]}
        
    res = {}
    # compute mean and medians
    cluster_means, cluster_medians, cluster_medoids = {}, {}, {}
    for cid, points in clusters.items(): 
        cluster_means[cid] = np.mean(points, axis=0)
        cluster_medians[cid] = np.median(points, axis=0)

    res['cluster_means'] = cluster_means
    res['cluster_medians'] = cluster_medians

    nrows = len(cluster_labels)
    assert X.shape[0] == nrows
    c2p = map_clusters(cluster_labels, range(nrows))  # cluster to position i.e. id -> indices 
        
    # k-nearest neighbors wrt mean given a distance metric 
    # [params]
    topk = 10
    dmetric = ssd.cosine # choose an appropriate metric 

    representatives = {'mean': cluster_means, 'median': cluster_medians, }
    for rtype, cluster_repr in representatives.items(): 
        cluster_knn = eval_knn(cluster_repr, metric=dmetric, topk=topk)
        res['cluster_%s' % rtype] = cluster_knn

    # save statistics 
    # res['cluster_knn'] = cluster_knn
    return res

def cluster_sampling(X, cluster_labels, n_sample=None, **kargs):
    if n_sample is None: n_sample = max(1, X.shape[0]/10)

    Xsub, cids = sampling.sample_class2(X, y=None, n_sample=n_sample **kargs) 

    return (Xsub, cids)

def eval_knn(cluster_repr, metric=None, topk=10): 
    """
    Input
    -----
    cluster_repr: cluster representatives in terms of a dictionary: cid -> representative point (e.g. centroid)
    topk: k in knn
 
    """
    if metric is None: metric = ssd.cosine
    cluster_knn = {}
    for cid, rpt in cluster_repr.items(): # foreach cluster (id) and its centroid
        idx = c2p[cid]  # idx: all data indices in cluster cid 
        rankedDist = sorted([(i, metric(X[i], rpt)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
        # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
        cluster_knn[cid] = [i for i, d in rankedDist]
    return cluster_knn

def cluster_analysis(ts=None, **kargs): 
    def cluster_distribution(ldmap): # label to distribution map
        # [params] testdir
        # plot histogram of distancesto the origin for all document vectors
        for ulabel, distr in ldmap.items(): 
            plt.clf() 
           
            canonical_label = lmap.get(ulabel, 'unknown')
            
            f = plt.figure(figsize=(8, 8))
            sns.set(rc={"figure.figsize": (8, 8)})
            
            intervals = [i*0.1 for i in range(10+1)]
            sns.distplot(distr, bins=intervals)

            # n, bins, patches = plt.hist(distr)
            # print('verfiy_norm> Label: %s > seq_ptype: %s, d2v: %s > n: %s, n_bins: %s, n_patches: %s' % \
            #      (canonical_label, seq_ptype, d2v_method, n, len(bins), len(patches)))

            identifier_distr = 'L%s-P%s' % (canonical_label, identifier)
            fpath = os.path.join(plotdir, 'cluster_distribution-%s.tif' % identifier_distr)
            print('output> saving cluster distribution to %s' % fpath)
            plt.savefig(fpath)

        return
    def summarize_cluster(cluster_labels):  # [refactor] also see cluster.analyzer
        clusters = map_clusters(cluster_labels, X)  # id -> {X[i]}
        
        res = {}
        # compute mean and medians
        cluster_means, cluster_medians, cluster_medoids = {}, {}, {}
        for cid, points in clusters.items(): 
            cluster_means[cid] = np.mean(points, axis=0)
            cluster_medians[cid] = np.median(points, axis=0)

        nrows = len(cluster_labels)
        assert X.shape[0] == nrows
        c2p = map_clusters(cluster_labels, range(nrows))  # cluster to position i.e. id -> indices 
        
        # k-nearest neighbors wrt mean given a distance metric 
        # [params]
        topk = 10
        dmetric = ssd.cosine # choose an appropriate metric 

        cluster_knn = {}
        for cid, mpoint in cluster_means.items(): # foreach cluster (id) and its centroid
            idx = c2p[cid]  # idx: all data indices in cluster cid 
            rankedDist = sorted([(i, dmetric(X[i], mpoint)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
            # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
            cluster_knn[cid] = [i for i, d in rankedDist]

        cluster_knn_median = {}
        for cid, mpoint in cluster_medians.items(): # foreach cluster (id) and its centroid
            idx = c2p[cid]  # idx: all data indices in cluster cid 
            rankedDist = sorted([(i, dmetric(X[i], mpoint)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
            # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
            cluster_knn_median[cid] = [i for i, d in rankedDist]

        # save statistics 
        res['cluster_means'] = cluster_means
        res['cluster_medians'] = cluster_medians
        res['cluster_knn'] = cluster_knn   # knn wrt mean
        res['cluster_knn_median'] = cluster_knn_median  # knn wrt median

        return res


    def evaluate_cluster(labels_cluster): 
        div(message='clustering algorithm: %s' % cluster_method)

        labels_true, labels = y, labels_cluster

        n_clusters_free = ('db', 'aff')  # these do not require specifying n_clusters
        if cluster_method.startswith(n_clusters_free): 
            print('(Estimated) number of clusters: %d' % n_clusters_est)
        else: 
            print('(Specified) number of clusters: %d' % n_clusters)

        mdict = {}

        if labels_true is not None: 
            mdict['homogeneity'] = metrics.homogeneity_score(labels_true, labels)
            print("Homogeneity: %0.3f" % mdict['homogeneity'])

            mdict['completeness'] = metrics.completeness_score(labels_true, labels)
            print("Completeness: %0.3f" % mdict['completeness'])

            mdict['v_measure'] = metrics.v_measure_score(labels_true, labels)
            print("V-measure: %0.3f" % mdict['v_measure'])

            mdict['ARI'] = metrics.adjusted_rand_score(labels_true, labels)
            print("Adjusted Rand Index: %0.3f" % mdict['ARI'])

            mdict['AMI'] = metrics.adjusted_mutual_info_score(labels_true, labels)
            print("Adjusted Mutual Information: %0.3f" % mdict['AMI'])
        else: 

            # only silhouette coeff doesn't require ground truths (expensive!)
            # add sampling method
            n_sample = 1000 
            try: 
                Xsub = sampling.sample_class2(X, y=labels, n_sample=n_sample, replace=False) # [todo] without replacemet => replacement upon exceptions
            except: 
                print('evaluate> could not sample X without replacement wrt cluster labels (dim X: %s while n_cluster: %d)' % \
                    (str(X.shape), len(set(labels)) ))
                Xsub = sampling.sample_class2(X, y=labels, n_sample=n_sample, replace=True)

            try: 
                mdict['silhouette'] = metrics.silhouette_score(Xsub, np.array(labels), metric='sqeuclidean')
                print("Silhouette Coefficient: %0.3f" % mdict['silhouette'])
            except Exception, e: 
                print('evaluate> Could not compute silhouette score: %s' % e)
            
        return mdict 

    # from seqmaker.seqparams import TSet 
    from config_sprint import TSet 
    from seqmaker import evaluate
    # from sklearn.cluster import AgglomerativeClustering
    # from sklearn.neighbors import kneighbors_graph
    # from sklearn.cluster import AffinityPropagation
    # from sklearn import metrics
    # rom scipy.spatial import distance
    
    # from seqmaker import seqUtils as su
    # import sampling

    # [params]
    experiment_id = 'sprint'
    
    # [params]
    n_clusters = kargs.get('n_clusters', 10)
    n_classes = kargs.get('n_classes', 1)  # default 1, assuming that ground truths are not available
    n_clusters_est = -1

    cluster_method = kargs.get('cluster_method', kargs.get('c_method', 'kmeans'))

    inputdir = datadir = os.path.join(os.getcwd(), 'data')
    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'data')) # sys_config.read('DataExpRoot')
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory
    save_cluster = kargs.get('save_', True)  # save cluster labels? 

    plotdir = os.path.join(os.getcwd(), 'plot')

    lmap = kargs.get('class_label_names', {0: 'TD1', 1: 'TD2', 2: 'Gest', })  # mapping from class label to meanings

    # [params] identifier for d2v models
    seq_ptype = kargs.get('seq_ptype', None) # sequence pattern type: regular, random, diag, med, lab
    d2v_method = kargs.get('d2v_method', kargs.get('doc2vec_method', None))
    run_tsne = kargs.get('run_tsne', False)

    identifier = kargs.get('identifier', 'E%s-C%s-nC%s-nL%s' % (experiment_id, cluster_method, n_clusters, n_classes)) \
        if (seq_ptype is None or d2v_method is None) else 'E%s-C%s-nC%s-nL%s-S%s-D2V%s' % \
           (experiment_id, cluster_method, n_clusters, n_classes, seq_ptype, d2v_method)
    if run_tsne: 
        identifier += 'Mtsne'
    # map_label_only = kargs.get('map_label_only', True) # if False, then cluster map file will keep original docs in its data field

    doc_labeled = False
    X, y = kargs.get('X', None), kargs.get('y', None)
    seqx = D = kargs.get('D', None)
    meta_fields = kargs.get('meta_fields', config_sprint.TSet.meta_fields)

    ts_indices = None
    if ts is None: 
        assert X is not None, "No training data set given either in the form of matrix (X) or dataframe (ts)"
        ts_indices = np.array(range(0, X.shape[0]))
    else: 
        ts_indices = ts[TSet.index_field].values  # ID field of the training data (e.g. person IDs)
        X, y = evaluate.transform(ts, standardize_=kargs.get('std_method', 'minmax'), meta_fields=meta_fields) # default: minmax

    if y is None: y = np.repeat(1, X.shape[0]) # unlabeled data
    n_classes_verified = len(set(y))
    if n_classes is not None: assert n_classes == n_classes_verified
    if n_classes_verified > 1: 
        doc_labeled = True # each document is represented by its (surrogate) labels
    if D is not None: X.shape[0] == len(D)

    # labels_true = y
    print('verify> dim(X): %s | n_clusters=%d, n_classes=%d' % (str(X.shape), n_clusters, n_classes))


    # preprocess with T-SNE or other manifold learning methods? 
    if run_tsne: 
        Xp = learn_manifold.tsne(X)
    else: 
        Xp = X
 
    # [params] output 
    # cluster labels 
    load_path = os.path.join(outputdir, 'cluster-%s.csv' % identifier)  
    load_cluster = kargs.get('load_', True) and os.path.exists(load_path)

    ### Run Cluster Analysis ### 
    model = n_cluster_est = None   # [todo] model persistance
    if not load_cluster: 
        model, n_clusters_est = run_cluster_analysis(Xp, y, **kargs)
        cluster_labels = model.labels_
        # cluster_inertia   = model.inertia_
    else: 
        print('io> loading pre-computed cluster labels (%s, %s, %s) ...' % (cluster_method, seq_ptype, d2v_method))
        df = pd.read_csv(load_path, sep=',', header=0, index_col=False, error_bad_lines=True)
        cluster_labels = df['cluster_id'].values
        assert Xp.shape[0] == len(cluster_labels), "nrow of X: %d while n_cluster_ids: %d" % (Xp.shape[0], len(cluster_labels))

    print('status> completed clustering with %s' % cluster_method)

    res_metrics = {}
    if y is not None and n_classes > 1: 
        res_metrics = evaluate_cluster(cluster_labels)
        doc_labeled = True # each document is represented by its (surrogate) labels
    res_summary = summarize_cluster(cluster_labels)

    # [note] alternatively use map_clusters2() to map cluster IDs to appropriate data point repr (including labels)
    # cluster_to_docs  = map_clusters(cluster_labels, seqx)

    ### Test ### 
    if D is None: # label only for 'data'
        cluster_to_docs  = map_clusters(cluster_labels, y)  # y: pre-computed labels (e.g. heuristic labels)
    else: 
        cluster_to_docs  = map_clusters(cluster_labels, D)
    print('info> resulted n_cluster: %d =?= %d' % (len(cluster_to_docs), n_clusters))  # n_cluster == 2

    ### save clustering result 
    if save_cluster: 
            
        header = ['id', 'cluster_id', ]
        adict = {h:[] for h in header}
        fpath = os.path.join(outputdir, 'cluster_id-%s.csv' % identifier)

        for i, cl in enumerate(cluster_labels):
            adict['id'].append(ts_indices[i])
            adict['cluster_id'].append(cl)

        df = DataFrame(adict, columns=header) 
        if doc_labeled:
            header = ['id', 'cluster_id', 'label', ]
            df['label'] = y

        print('output> saving cluster map (id -> cluster id) to %s' % fpath)
        df.to_csv(fpath, sep='|', index=False, header=True)  

        # [output]
        if cluster_to_docs is not None: 
            fpath = os.path.join(outputdir, 'cluster_map-%s.csv' % identifier)  
            header = ['cluster_id', 'data', ] # in general, this should include 'id'
            adict = {h:[] for h in header}
            size_cluster = 0
            for cid, content in cluster_to_docs.items():
                size_cluster += len(content)
            size_avg = size_cluster/(len(cluster_to_docs)+0.0)
            print('verify> averaged %s-cluster size: %f' % (cluster_method, size_avg))
            
            if D is not None: # save only 
                rid = random.sample(cluster_to_docs.keys(), 1)[0]  # [test]
                for cid, content in cluster_to_docs.items():
                    if cid == rid: 
                        print "log> cid: %s => %s\n" % (cid, str(cluster_to_docs[cid]))
                    # store 
                    adict['cluster_id'].append(cid)
                    adict['data'].append(content[0]) # label, sequence, etc. # [todo]
                df =  DataFrame(adict, columns=header)
                print('output> saving cluster content map (cid -> contents) to %s' % fpath)

        # [output] save knn 
        if res_summary: 
            header = ['cluster_id', 'knn', ]
            cluster_knn_map = res_summary['cluster_knn']  # wrt mean, cid -> [knn_id]
            sep_knn = ','
            adict = {h: [] for h in header}
            for cid, knn in cluster_knn_map.items(): 
                adict['cluster_id'].append(cid) 
                adict['knn'].append(sep_knn.join([str(e) for e in knn]))
        
            fpath = os.path.join(outputdir, 'cluster_knnmap-%s.csv' % identifier)
            print('output> saving knn-to-centriod map (cid -> knn wrt centroid) to %s' % fpath)
            df = DataFrame(adict, columns=header)
            df.to_csv(fpath, sep='|', index=False, header=True)

    # optimal number of clusters?
    # step 2: Evalution 
    # compute silhouette scores (n_clusters vs silhouette scores)
    if doc_labeled: 
        div(message='Testing cluster consistency with surrogate labels (%s) ...' % cluster_method)
        labels_true = y
        ulabels = set(labels_true)
        n_labels = len(ulabels)
        ratios = {l:[] for l in ulabels}
        for ulabel in ulabels: 
            for cid, alist in cluster_to_docs.items():
                counts = collections.Counter(alist)
                r = counts[ulabel]/(len(alist)+0.0) # given a (true) label, find the ratio of that label in a cluster
                ratios[ulabel].append(r)
        print('result> cluster-label distribution (method: %s):\n%s\n' % (cluster_method, ratios))
        cluster_distribution(ratios)

    return (cluster_labels, res_metrics)  # cluster id to document members (labels or content)

def run_cluster_analysis(X, y=None, **kargs):  
    """
    Main routine for running cluster analysis. 


    Related
    -------
    cluster_analysis()
    run_silhouette_analysis() for kmeans

    """
    # from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN

    cluster_method = kargs.get('cluster_method', 'kmeans').lower()
    n_clusters = kargs.get('n_clusters', 10)
    n_clusters_est = None
    print('run_cluster_analysis> requested %d clusters' % n_clusters)
    if cluster_method in ('kmeans', 'k-means', ): 
        if kargs.get('optimize_k', False): 
            n_clusters_silh = run_silhouette_analysis(X, y=y, **kargs)  # external args: outputdir
            print('status> best n_clusters (by silhouette test): %d vs requested %d' % (n_clusters_silh, n_clusters))
            n_clusters = n_clusters_silh

        model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        model.fit(X)
    elif cluster_method in ('minibatch', 'minibatchkmeans'):
        model = MiniBatchKMeans(n_clusters=n_clusters)  # init='k-means++', n_init=3 * batch_size, batch_size=100
        model.fit(X)
    elif cluster_method.startswith('spec'):
        model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors") 
        model.fit(X)
    elif cluster_method.startswith('agg'):
        knn_graph = kneighbors_graph(X, 30, include_self=False)  # n_neighbors: Number of neighbors for each sample.

        # [params] AgglomerativeClustering
        # connectivity: Connectivity matrix. Defines for each sample the neighboring samples following a given structure of the data. 
        #               This can be a connectivity matrix itself or a callable that transforms the data into a connectivity matrix
        # linkage:  The linkage criterion determines which distance to use between sets of observation. 
        #           The algorithm will merge the pairs of cluster that minimize this criterion.
        #           average, complete, ward (which implies the affinity is euclidean)
        # affinity: Metric used to compute the linkage
        #           euclidean, l1, l2, manhattan, cosine, or precomputed
        #           If linkage is ward, only euclidean is accepted.
        connectivity = knn_graph # or None
        linkage = kargs.get('linkage', 'average')
        model = AgglomerativeClustering(linkage=linkage,  
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
        model.fit(X)
    elif cluster_method.startswith('aff'): # affinity propogation
        damping = kargs.get('damping', 0.9)
        preference = kargs.get('preference', -50)

        # expensive, need subsampling
        model = AffinityPropagation(damping=damping, preference=preference) 
        model.fit(X)
        
        cluster_centers_indices = model.cluster_centers_indices_
        n_clusters_est = len(cluster_centers_indices)
        print('affinityprop> method: %s (damping: %f, preference: %f) > est. n_clusters: %d' % (cluster_method, damping, preference, n_clusters)) 

    elif cluster_method.startswith('db'): # DBSCAN: density-based 
        # first estimate eps 
        n_sample_max = 500
        metric = kargs.get('metric', 'euclidean')

        # [note] 
        # eps : float, optional
        #       The maximum distance between two samples for them to be considered as in the same neighborhood.
        # min_samples : int, optional
        #       The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
        #       This includes the point itself.
        
        eps = kargs.get('eps', None)
        if eps is None:      
            X_subset = X[np.random.choice(X.shape[0], n_sample_max, replace=False)] if X.shape[0] > n_sample_max else X

            # pair-wise distances
            dmat = distance.cdist(X_subset, X_subset, metric)
            off_diag = ~np.eye(dmat.shape[0],dtype=bool)  # don't include distances to selves
            dx = dmat[off_diag]
            sim_median = np.median(dx)
            sim_min = np.min(dx)
            eps = (sim_median+sim_min)/2.

        print('dbscan> method: %s > eps: %f' % (cluster_method, eps))
        model = DBSCAN(eps=eps, min_samples=10, metric=metric) 
        model.fit(X)

        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True

        cluster_labels = model.labels_
        n_clusters_est = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    else: 
        raise NotImplementedError, "Cluster method %s is not yet supported" % cluster_method 

    return (model, n_clusters_est)

def run_silhouette_analysis(X, y=None, **kargs):
    from sklearn.metrics import silhouette_samples, silhouette_score
    # import matplotlib.cm as cm

    # [params]
    range_n_clusters = kargs.get('range_n_clusters', [2, 3, 4, 5, 6, 10, 15, 20])
    n_clusters_requested = kargs.get('n_clusters', None)
    if n_clusters_requested is not None: 
        if not n_clusters_requested in range_n_clusters: 
            range_n_clusters.append(n_clusters_requested)
    print('param> input n_clusters (requested): %d > range_n_clusters: %s' % (n_clusters_requested, range_n_clusters))

    # identifier 
    identifier = kargs.get('identifier', 'nR%s-%s' % (min(range_n_clusters), max(range_n_clusters)))
    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    ranked_scores = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        ranked_scores.append((n_clusters, silhouette_avg))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                     marker='o', c="white", alpha=1, s=200)
   
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        # plt.show()
        graph_ext = 'tif'
        fpath = os.path.join(outputdir, 'silhouette_test-%s-nC%s.%s' % (identifier, n_clusters, graph_ext))
        print('output> saving silhouette test result to %s' % fpath)
        plt.savefig(fpath)
    ### end range of n_clusters 

    ranked_scores = sorted(ranked_scores, key=lambda x: abs(x[1]), reverse=False) # reverse=False => ascending 
    print('output> ranked scores (n_clusters vs average score):\n%s\n' % ranked_scores)

    return ranked_scores[0][0]

def t_cluster(**kargs): 
    import gap, cluster_utils, tsne_test
    import itertools
    
    prefix = os.path.join(kargs.get('prefix', os.getcwd()), 'data')
    fpath = os.path.join(prefix, 'tset-Esprint-Tbaseline.csv')
    ts = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded data of dim %s from:\n%s\n' % (str(ts.shape), fpath)) # [log] dim: (8808, 16)

    meta_fields = config_sprint.TSet.meta_fields

    ## find optimal k
    # paramset = [{'cluster_method': 'kmeans', 'n_clusters': 10, 'optimize_k': True, 
    #              'run_tsne': True, 
    #              'range_n_clusters': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50]}, 
    #              ]

    ## loop 
    # cluster methods
    cluster_methods = ['kmeans', 'minibatch', 'spectral', ]
    n_clusterx = [5, 10, 20, 50, 100, 150, 200]
    cluster_methods_kless = ['dbscan']
    paramset = []
    
    # methods with k or n_clusters 
    for params in itertools.product(*[cluster_methods, n_clusterx]): 
        paramset.append({'cluster_method': params[0], 'n_clusters': params[1]})
    # methods without k or n_clusters 
    for params in itertools.product(*[cluster_methods_kless, ]):    
        paramset.append({'cluster_method': params[0], })

    ## parameter setting examples from above
    # paramset = [{'cluster_method': 'kmeans', 'n_clusters': 10}, 
    #              {'cluster_method': 'kmeans', 'n_clusters': 20}, 
    #              {'cluster_method': 'kmeans', 'n_clusters': 50},
    #              {'cluster_method': 'kmeans', 'n_clusters': 100},
    #              {'cluster_method': 'minibatch', 'n_clusters': 10},
    #              {'cluster_method': 'minibatch', 'n_clusters': 20}, 
    #              {'cluster_method': 'minibatch', 'n_clusters': 50},
    #              {'cluster_method': 'minibatch', 'n_clusters': 100}, 
    #              {'cluster_method': 'spectral', 'n_clusters': 10},
    #               {'cluster_method': 'spectral', 'n_clusters': 20},
    #               {'cluster_method': 'spectral', 'n_clusters': 50},
    #               {'cluster_method': 'spectral', 'n_clusters': 100},
    #               {'cluster_method': 'dbscan'},
    #              ]

    # X0, X1 = tsne_test.run_tsne(X, **kargs)

    for params in paramset: 
        cluster_analysis(ts=ts, **params)

    ## gap statistics 
    # use gap.py or gap2.py (which uses pip installed python package)

    return 

def t_tsne(**kargs): 
    import tsne_test
    prefix = os.path.join(kargs.get('prefix', os.getcwd()), 'data')
    fpath = os.path.join(prefix, 'tset-Esprint-Tbaseline.csv')
    ts = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded data of dim %s from:\n%s\n' % (str(ts.shape), fpath)) # [log] dim: (8808, 16)
    meta_fields = config_sprint.TSet.meta_fields

    X = np.asarray(X, dtype=np.float64)
    # y = np.asarray(y, dtype=np.float64)

    print('run_tsne> X dim: %s | X[1][:10]: %s' % (str(X.shape), X[1][:10]))

    # do SVD first? 

    # perform t-SNE embedding
    # vis_data = bh_sne(X)
    X0, X1 = tsne_test.run_tsne(X, **kargs)

    return

def test(**kargs): 
    import load_data as ld
   
    ## step one: make training data 
    # ts = ld.make_tset() 

    ## cluster analysis loop
    t_cluster(**kargs) 

    return 


if __name__ == "__main__": 
    test() 


