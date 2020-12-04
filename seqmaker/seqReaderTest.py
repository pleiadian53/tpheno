import pandas as pd
import os, gc, sys
from os import getenv 
import time, re

# local modules 
from batchpheno import icd9utils
from config import seq_maker_config, sys_config

from pattern import medcode as pmed

import seqReader

# word2vec modules
import gensim
from gensim.models.lsimodel import stochastic_svd as svd

##########################################################################
#
#  A test module for building seqReader which focuses mainly on 
#  gensim and TensorFlow
#
#
##########################################################################



default_root = sys_config.read('DataIn')  # input data directory
assert os.path.exists(default_root)

def iter_documents(top_directory, ext='.dat'):
    """
    Generator: iterate over all relevant documents, yielding one
    document (=list of utf8 tokens) at a time.

    A "friend function" to TxtSubdirsCorpus
    """
    # find all .txt documents, no matter how deep under top_directory
    for root, dirs, files in os.walk(top_directory):
        for fname in filter(lambda fname: fname.endswith(ext), files):
            # read each document as one big string
            document = open(os.path.join(root, fname)).read()
            # break document into utf8 tokens
            yield gensim.utils.tokenize(document, lower=True, errors='ignore')


class TxtSubdirsCorpus(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, top_dir, doc_ext='.dat'):
        self.top_dir = top_dir
        self.doc_ext = doc_ext

        # create dictionary = mapping for documents => sparse vectors
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir, ext=self.doc_ext))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for tokens in iter_documents(self.top_dir):
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)

def load_doc(**kargs):
    """

    Note
    ----
    1. Give a hint to the stochastic SVD algo with chunksize=5000 to process its input stream in groups of 5,000 vectors

    """

    # [params]
    basedir = kargs.get('top_dir', default_root)
    ext = '.txt'
    verify_ = True 

    if verify_: 
        doc_files = [d for d in os.listdir(basedir) if d.find(ext) > 0]
        print('info> number of %s file: %d' % (ext, len(doc_files)))
        for d in doc_files: 
            print('  + %s' % d)

    corpus = TxtSubdirsCorpus(basedir, doc_ext=ext)

    if verify_: 
        print "info> show the corpus vectors:\n%s\n" % corpus.dictionary
        for vector in corpus:
            print vector

    # or run truncated Singular Value Decomposition (SVD) on the streamed corpus
    u, s = svd(corpus, rank=200, num_terms=len(corpus.dictionary), chunksize=5000) # [1]

    return

def test(**kargs):
    load_doc(**kargs)

    return 

if __name__ == "__main__": 
    test()







