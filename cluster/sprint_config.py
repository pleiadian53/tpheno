# encoding: utf-8

import pandas as pd
from pandas import DataFrame, Series

# import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
# from matplotlib import pyplot as plt

import os, sys, collections, re, glob
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

ProjDir = '/Users/pleiades/Documents/work/sprint'
DataDir = os.path.join(ProjDir, 'data')


class TSet(object):
    index_field = 'MASKID'
    date_field = 'date'
    target_field = 'target'  # usually surrogate labels
    annotated_field = 'annotated'
    content_field = 'content'  # representative sequence elements 
    label_field = 'mlabels'  # multiple label repr of the underlying sequence (e.g. most frequent codes as constituent labels)

    meta_fields = [target_field, index_field, ]