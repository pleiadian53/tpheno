import numpy as np
from scipy import stats, integrate

import random, os, sys, re
from collections import Counter

import pandas as pd 
from pandas import DataFrame

### Plotting ###

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

BaseDir = os.path.join(os.getcwd(), 'test')

# T1D instance ratios for all clusters (size=50)
t1d_ratios = [0.051041666666666666, 0.035224153705397984, 0.06823420931304748, 0.020106221547799695, 0.0015360983102918587, 0.08211678832116788, 
 0.3080168776371308, 0.06937799043062201, 0.016602809706257982, 0.06314993382491964, 0.012404149751917006, 0.07775061124694377, 
 0.07809449433814916, 0.04503152206544581, 0.9830699774266366, 0.0, 0.04781704781704782, 0.04249815225424981, 0.09749920861031972, 
 0.011132519803040034, 0.1439153439153439, 0.08227401129943503, 0.037360890302066775, 0.0077938947824204375, 0.04245709123757904, 
 0.06310679611650485, 0.08936651583710407, 0.03532008830022075, 0.018963144211653158, 0.02, 0.044671824307461944, 0.3157894736842105, 
 0.050314465408805034, 0.296667649660867, 0.03769140164899882, 0.018266475644699142, 0.033937170373767485, 0.08412698412698413, 
 0.02937062937062937, 0.2880080280983442, 0.19477006311992787, 0.09057750759878419, 0.07345575959933222, 0.00718132854578097, 
 0.014360313315926894, 0.15438247011952191, 0.03509487859620486, 0.05945604048070841, 0.11780455153949129, 0.025338253382533826]

t2d_ratios = [0.9489583333333333, 0.964775846294602, 0.9317657906869525, 0.9798937784522003, 0.9984639016897081, 0.9178832116788321, 
  0.6919831223628692, 0.930622009569378, 0.9833971902937421, 0.9368500661750804, 0.9875958502480829, 0.9222493887530563, 0.9219055056618508, 
  0.9549684779345542, 0.016930022573363433, 1.0, 0.9521829521829522, 0.9575018477457502, 0.9025007913896803, 0.9888674801969599, 0.8560846560846561, 
  0.917725988700565, 0.9626391096979332, 0.9922061052175796, 0.957542908762421, 0.9368932038834952, 0.9106334841628959, 0.9646799116997793, 
  0.9810368557883469, 0.98, 0.955328175692538, 0.6842105263157895, 0.949685534591195, 0.703332350339133, 0.9623085983510011, 0.9817335243553008, 
  0.9660628296262325, 0.9158730158730158, 0.9706293706293706, 0.7119919719016558, 0.8052299368800722, 0.9094224924012158, 0.9265442404006677, 
  0.992818671454219, 0.9856396866840731, 0.8456175298804781, 0.9649051214037951, 0.9405439595192916, 0.8821954484605087, 0.9746617466174662]


def histogram(iterable, low, high, bins=10):
    '''Count elements from the iterable into evenly spaced bins

        >>> scores = [82, 85, 90, 91, 70, 87, 45]
        >>> histogram(scores, 0, 100, 10)
        [0, 0, 0, 0, 1, 0, 0, 1, 3, 2]

    '''
    # from collections import Counter
    step = (high - low + 0.0) / bins
    dist = Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)] 

def t_facet(**kargs): 
    sns.set(style="darkgrid")

    tips = sns.load_dataset("tips") # header: total_bill tip sex smoker day time  size
    print('input> tips:\n%s\ndim:%s\n' % (tips, str(tips.shape)))  # dim:(244, 7)

    g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
    bins = np.linspace(0, 60, 15)  # foreach facet
    g.map(plt.hist, "total_bill", color="steelblue", bins=bins, lw=0)
 
    # fname = '%s-%s.%s' % (prefix, identifier, ext)
    fname = 'hist_facet-t1.tif'
    fpath = os.path.join(BaseDir, fname)
    plt.savefig(fpath)

    return

def t_distplot(x, **kargs): 

    # [params]
    prefix = kargs.get('prefix', 'hist')
    identifier = kargs.get('identifier', 'test')
    ext = 'tif'

    fig = plt.figure(figsize=(8, 8))

    sns.set(rc={"figure.figsize": (8, 8)})
     
    # np.arange(0, 1.1, 0.1) # rightmost exclusive
    # np.linspace(0, 1.0, 11) # 11 ticks => 10 intervals
    intervals = [i*0.1 for i in range(10+1)]
    
    # sns.distplot(x, bins=intervals)
    # sns.distplot(x);

    ### rug plot: adding small vertical ticks 
    sns.distplot(x, kde=False, rug=True)  # count will not be normalized when kdf is turned off

    # n, bins, patches = plt.hist(distr)
    
    fpath = os.path.join(BaseDir, '%s-%s.%s' % (prefix, identifier, ext))
    plt.savefig(fpath)


    return 


def t_overlay(x1, x2): 
    
    return

def test(**kargs): 
    
    # case 1: fake data
    # rx = np.hstack([np.repeat(0.1, 5), np.repeat(0.3, 6), np.repeat(0.9, 2)])
    # np.random.shuffle(rx)
    # print('stats> %s' % histogram(rx, 0.0, 1.0, bins=10)) # count the number of occurrences in each bin
    # t_distplot(rx)

    # case 2: cluster ratios from tpheno 
    # t_distplot(t1d_ratios, identifier='T1D_kmeans_ratios')
    # print('stats> %s' % histogram(t1d_ratios, 0.0, 1.0, bins=10)) # count the number of occurrences in each bin
    # t_distplot(t2d_ratios, identifier='T2D_kmeans_ratios')
    # print('stats> %s' % histogram(t2d_ratios, 0.0, 1.0, bins=10)) # count the number of occurrences in each bin

    # case 3: facetting histogram (case-by-case histograms)
    t_facet()
    
    return

if __name__ == "__main__": 
    test()