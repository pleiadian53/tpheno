
# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20170322

# We'll use matplotlib for graphics.
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# %matplotlib inline

import random

import pandas as pd 
from pandas import DataFrame

# temporal sequence modules
from config import seq_maker_config, sys_config
from seqmaker import seqCluster as sc


# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy


digits = load_digits()
digits.data.shape

print(digits['DESCR'])


def load_data(): 
    # from sklearn.datasets import load_digits
    digits = load_digits()
    print digits.data.shape

    print(digits['DESCR'])

    nrows, ncols = 2, 5
    plt.figure(figsize=(6,3))
    plt.gray()
    for i in range(ncols * nrows):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.matshow(digits.images[i,...])
        plt.xticks([]); plt.yticks([])
        plt.title(digits.target[i])
    plt.savefig('images/digits-generated.png', dpi=300)

    # We first reorder the data points according to the handwritten numbers.
    X = np.vstack([digits.data[digits.target==i]
                   for i in range(10)])
    y = np.hstack([digits.target[digits.target==i]
                    for i in range(10)])

    print('verify> X dim: %s, y dim: %s' % (str(X.shape), str(y.shape))) #  dim: (1797, 64), y dim: (1797,)
    print('+ X:\n%s\n' % X[0])

    return (X, y)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def scatter2(x, labels):
    """

    Memo
    ----
    1. circular color system: 
        http://seaborn.pydata.org/tutorial/color_palettes.html

    """
    example_label = labels[random.sample(labels, 1)[0]]
    print('verify> label: %s' % example_label)

    ulabels = set(labels) # unique labels
    n_colors = len(set(labels))

    lcmap = {}  # label to color id: assign a color to each unique label 
    for ul in ulabels: 
    	if not lcmap.has_key(ul): lcmap[ul] = len(lcmap)
    clmap = {c: l for l, c in lcmap.items()}  # reverse map: color id => label
    
    # from labels to colors
    colors = np.zeros((len(labels), ), dtype='int') # [0] * n_colors
    for i, l in enumerate(labels): 
        colors[i] = lcmap[l]  # label => color id

    palette = np.array(sns.color_palette("hls", n_colors)) # [m1] circular color system

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(n_colors): # foreach color
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0) # find all fvec of i-th color
        txt = ax.text(xtext, ytext, str(clmap[i]), fontsize=18)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts 
    

def run_tsne(X, y=None, n_components=2): 
    X_proj = TSNE(n_components=n_components, random_state=RS).fit_transform(X)
 
    return X_proj 


def t_digits():
    X, y = load_data()
    digits_proj = run_tsne(X)
    
    print('verify> X_proj dim: %s' % str(digits_proj.shape)) #  

    scatter(digits_proj, y)
    plt.savefig('images/digits_tsne-generated.png', dpi=120)

    return  

def t_tsne2(**kargs): 
    def mlabel_to_slabel(mlabels):  # or use seqparams.lsep 
    	labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
        print('verify> labels (dtype=%s): %s' % (type(labels), labels[:10]))
        return labels
    def v_lsep(y, n=10): 
        got_multilabel =True
        for i, e in enumerate(y):
            if i >= n: break 
            if len(e.split(lsep)) < 2: 
        		got_multilabel = False 
        		break 
        return got_multilabel

    from seqmaker import seqSampling as ss
    from seqmaker import seqCluster as sc 

    basedir = sys_config.read('DataExpRoot')  
    d2v_method = 'PVDM'
    vis_method = 'tsne'
    meta = kargs.get('meta', '%s-V%s' % (d2v_method, vis_method))

    redo_clustering = True
    n_sample = 20000  # still got memory error with plotly
    print('params> desired sample size: %d' % n_sample)

    lsep = '_'

    ### load data
    X, y, D = sc.build_data_matrix2(**kargs)
    n_unique_mlabels = len(set(y))

    # verifying the data
    print('t_condition_diag> X (dim: %s), y (dim: %s), D (dim: %s)' % (str(X.shape), str(y.shape), len(D)))  
    print('verify> example composite labels:\n%s\n' % y[:10])
    print('verify> got multiple label with lsep=%s? %s' % (lsep, v_lsep(y)))

    yp = mlabel_to_slabel(y)
    n_unique_slabels = len(set(yp))
    print('verify> total number of labels: %d > n_multilabels: %d > n_slabels: %d' % \
    	(len(y), n_unique_mlabels, n_unique_slabels))

    # sample subset 
    candidates = ss.get_representative(docs=D, n_sample=n_sample, n_doc=X.shape[0], policy='rand')
    # if redo_clustering: X, y, D = sc.build_data_matrix2()
    X_subset, y_subset = X[candidates], y[candidates]

    print('input> prior to tsne > X (dim: %s), y (dim: %s)' % (str(X_subset.shape), str(y_subset)) )

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(X_subset).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))
    y_data = y_subset

    X_proj = run_tsne(X=x_data, y=y_data, n_components=2)  
    # 2D: X_proj[:, 0], X_proj[:, 1]
    print('output> X_proj (dim: %s)' % str(X_proj.shape))  

    plt.clf()
    scatter2(X_proj, y)
    plt.savefig('images/%s_s%d.png' % (meta, n_sample), dpi=300)
    plt.close()

    return

def test(**kargs): 
    # t_digits()
    t_tsne2(**kargs)

    return

if __name__ == "__main__": 
    test()












