# encoding: utf-8

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import numpy as np 
from operator import itemgetter
from collections import OrderedDict
import multiprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

import random, os, gc, sys
from pandas import DataFrame
from batchpheno import sampling
from batchpheno.utils import div
from config import sys_config
import seqReader as sr
import seqAlgo    # sequence algorithm library (specialized version of algorithms)
import seqparams  # all algorithm variables and settings
import labeling

# word2vec and doc2vec modules
import gensim
from gensim.models import Word2Vec, Doc2Vec

# Tensorflow

####################################################################################################
#
#
#  Usage Note
#  ----------
#  This module subsumes analyzer.py in creating feature vectors, word vectors, and document vectors. 
#
#  - configure w2v, d2v parameters via W2V and D2V 
#  - 
#
#  Reference 
#  ---------
#  1. https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors
# 
# 
#
# 
#
####################################################################################################


class W2V(seqparams.W2V):  # refactored from seqparams
    """
    

    Params
    ------
    window: is the maximum distance between the predicted word and context words used for prediction within a document.
    min_count: ignore all words with total frequency lower than this.


    Note
    ----
    Notes on Parameter choices: 

    1. 100-dimensional vectors, as the 400d vectors of the paper don't seem to offer much benefit on this task; 
       similarly, frequent word subsampling seems to decrease sentiment-prediction accuracy, so it's left out

    2. cbow=0 means skip-gram which is equivalent to the paper's 'PV-DBOW' mode, 
       matched in gensim with dm=0; 
       added to that DBOW model are two DM models, one which averages context vectors (dm_mean) and 
       one which concatenates them (dm_concat, resulting in a much larger, slower, more data-hungry model)

    3. a min_count=2 saves quite a bit of model memory, discarding only words that appear in a single doc 
      (and are thus no more expressive than the unique-to-each doc vectors themselves)

    """

    # [params] example parameters already defined in seqparams
    # n_features = 100
    # window = 5
    # min_count = 2  # set to >=2, so that symbols that only occur in one doc don't count
    
    # n_cores = multiprocessing.cpu_count()
    # # print('info> number of cores: %d' % n_cores)
    # n_workers = max(n_cores-20, 15)

    # word2vec_method = 'SG'  # skipgram 
    # doc2vec_method = 'PVDM'  # default: distributed memory
    # read_mode = 'doc' 
    
    supported_methods = ['sg', 'cbow', ]
    w2v_method = 'sg'
    
    @staticmethod 
    def isSupported(w2v_method): 
        return w2v_method in supported_methods

### end class W2V 

class D2V(seqparams.D2V): # refactored from seqparams
    """

    Params
    ------
    dm: defines the training algorithm. By default (dm=1), 'distributed memory' (PV-DM) is used. 
        Otherwise, distributed bag of words (PV-DBOW) is employed.

    dm_mean: if 0 (default), use the sum of the context word vectors. 
             If 1, use the mean. Only applies when dm is used in non-concatenative mode.


    (*) choosing between hierarchical softmax and negative sampling 

    hs = if 1, hierarchical softmax will be used for model training. 
               i.e. for each node in the tree, there's a classifier that tells apart the target being either in the left or in right partitions
                    of words 
         If set to 0 (default), and negative is non-zero, negative sampling will be used.

        (*) how many negative samples? 

            negative:  if > 0, negative sampling will be used, the int for negative specifies how many 
                       “noise words” should be drawn (usually between 5-20). Default is 5. 
                       If set to 0, no negative samping is used.

    (*) number of iterations (controls both the optimization, reducing cost function J, and model complexity)
    iter: number of iterations (epochs) over the corpus. The default inherited from Word2Vec is 5, 
          but values of 10 or 20 are common in published ‘Paragraph Vector’ experiments.

    References 
    ----------
    1. gensim
       https://radimrehurek.com/gensim/models/doc2vec.html
    2. 
    """
    # dm = 1  # PV-DM
    # dm_concat = 0  # don't concatentate
    # dm_mean = 1 # take average of the word vectors as the document vector

    # negative = 5 
    # hs = 0  # use negative sampling 
    
    # simple_models = [
    #     # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    #     Doc2Vec(dm=1, dm_concat=1, size=100, window=W2V.window, negative=5, hs=0, min_count=W2V.min_count, workers=W2V.n_cores),
    #     # PV-DBOW 
    #     Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=W2V.min_count, workers=W2V.n_cores),
    #     # PV-DM w/average
    #     Doc2Vec(dm=1, dm_mean=1, size=100, window=W2V.window, negative=5, hs=0, min_count=W2V.min_count, workers=W2V.n_cores),
    # ]
    supported_methods = ['pv-dm2', 'pv-dm', 'pv-dbow', 'tf-idf', 'tfidf', ]
    d2v_method = 'pv-dm2' # default if not changed
    w2v_based_embedding = ['average', 'tfidf', 'tf-idf', ] # tfidfavg

    # document label attributes 
    label_attributes = ['words', 'tags', ]  

    @staticmethod 
    def isSupported(d2v_method): 
        return d2v_method in supported_methods

    @staticmethod
    def isWordVectorBased(d2v_method): # document vectors are computed externally based on word vectors (e.g. taking average)
        return d2v_method in D2V.w2v_based_embedding
    @staticmethod
    def isWVBased(d2v_method):
        return d2v_method in D2V.w2v_based_embedding 

    @staticmethod
    def show_params(): 
        div(message='Doc2Vec Parameters', symbol='%')
        msg  = '  + current d2v method: %s\n' % D2V.d2v_method  # pv-dm2: pv-dm + pv-dbow
        msg += '  + number of features: %d\n' % D2V.n_features
        msg += '  + window size: %d\n' % D2V.window 
        msg += '  + ignore tokens with total freq less than %d\n' % D2V.min_count
        msg += '  + number of epochs: %d\n' % D2V.n_iter
        
        prob_approx_method = D2V.prob_approx_method

        msg += '  + training method: %s' % prob_approx_method
        if D2V.hs == 0: 
            msg += '  + number of negative samples: %d' % D2V.negative
        
        msg += '  + use concatenation of context vectors? %s\n' % str(D2V.dm_concat == 1)

        print msg 
        
        return

    @staticmethod
    def getName(**kargs): 
        pass 
    @staticmethod
    def getPath(**kargs): # [params] d2v_method, outputdir
        # outputdir_default = TSet.getPath(cohort=cohort_name, dir_type=dir_type, create_dir=True)
        # tUseTSetDir = False
        # try: 
        #     from tset import TSet
        #     tUseTSetDir = True
        # except: 
        #     print('vector.D2V> Could not import tset.TSet...')

        outputdir = kargs.get('outputdir', os.getcwd())  # seqparams.TSet.getPath(); try to use different output dirs ~ cohort, content type
        
        # [note] sometimes a single algorithm may consist of more than one method, cannot simply rely on D2V.d2v_method
        #        e.g. pv-dm2 = pv-dm + pv-dbow
        d2v_method = kargs.get('d2v_method', D2V.d2v_method) 

        # file ID 
        identifier = kargs.get('identifier', 'Pf%dw%di%d' % (D2V.n_features, D2V.window, D2V.n_iter))  # P: parameters

        # suffix ~ meta in getDocVecPV() whichholds meta data such as model ID
        suffix = kargs.get('suffix', None) # e.g. model ID: used to distinguish models with the same parameter setting
        ofile_default = '%s.%s' % (identifier, d2v_method) if suffix is None else '%s-%s.%s' % (identifier, suffix, d2v_method)
        ofile = kargs.get('outputfile', ofile_default)

        fpath = os.path.join(outputdir, ofile)

        return fpath

    @staticmethod
    def load(**kargs):
        """

        Params
        ------
        suffix: corresponds to 'meta' in getDocVec() which serves as a model ID

        """
        tLoadSuccess = False
        modelPath = D2V.getPath(**kargs) # [params] d2v_method, outputdir 
        print('D2V.load> Info: model path:\n%s\n' % modelPath)

        model = None 
        if os.path.exists(modelPath): 
            try: 
                model = Doc2Vec.load(modelPath)  # can continue training with the loaded model! 
                if len(model.docvecs) > 0: 
                    tLoadSuccess = True 
                else: 
                    print('D2V.load> Warning: Loaded empty model from %s' % modelPath)
                    tLoadSuccess = False
            except: 
                print('D2V.load> Error: Could not load model from %s' % modelPath)
                tLoadSuccess = False 
        return model  # a valid model or None
    @staticmethod
    def save(model, **kargs):
        modelPath = D2V.getPath(**kargs) # [params] d2v_method, outputdir, identifier 
        model.save(modelPath)  # the model needs to support save()
        print('D2V.save> Info: saved model (method=%s):\n%s\n' % (kargs.get('d2v_method', '?'), modelPath))
        return

### end class D2V

def config(**kargs): 
    # configure model paramters via W2V and D2V (helper) classes 
    pass
def resolve(**kargs): 
    # resolve w2v_method, d2v_method
    pass

def word2index(model):
    """
    Extract word2index mapping from Word2Vec model.
    """
    word2index = OrderedDict((v, k) for k, v in sorted(model.index2word, key=itemgetter(1)))
    return word2index


# def makeFeatureVec(words, model, n_features=None): 
#     return getAverageVec(words, model, n_features=n_features)

def getAverageVec(tokens, model):
    """
    Function to average all of the word vectors in a given
    paragraph. 

    Input: tokens/words from a document (ordered albeith ordering not important)
    Output: document vector in terms of the average of all word vectors

    Related
    -------
    weightedAvg() | getWeightedAvgVec()

    """
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    n_features = model.vector_size # W2V.n_features 
        
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((n_features,), dtype="float32")
    #
    n_tokens = 0 
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for token in tokens:
        if token in index2word_set: 
            featureVec = np.add(featureVec,model.wv[token])  # [note] model[word] doesn't seem to work
            n_tokens = n_tokens + 1.

    msg = "None of the input words were indexed by the input model!"
    # assert n_tokens > 0, msg
    if n_tokens == 0: 
        print "warning> %s | input:\n%s\n" % (msg, str(tokens))
        return featureVec # return the zero vectors        
    return np.divide(featureVec, n_tokens)

def vectorizeByAvg(docs, model, **kargs):
    """
    Given a set of temporal docs (each one being a list of medical codes), calculate 
    the average feature vector for each one and return a 2D numpy array 
    """
    return byAvg(docs, model, **kargs)
def byAvg(docs, model, test_=True, w2v_method='sg'):  # formerly, getAvgFeatureVecs()
    """
    Vectorize each input document (given w2v or d2v model) by 
    taking an average of individual word vector. 

    Unlike vectorize2, this does not require 'docs' to be labeled. 

    Input: documents 
    Output: document vectors (based on averaging word vectors) 

    Related 
    -------
    vectorize() | getWordVec()
        based on w2v model: sg or cbow 
    vectorize2() | vecorizeDoc() | getDocVec()
        based on d2v model: pv-dm, pv-dbow, tfidf, average (which is the same as w2v)

    byAvg | vectorizeByAvg | vectorizeByAveraging

    byTfidf | vectorizeByTfidf 

    """
    n_features = model.vector_size # W2V.n_features
    assert model is not None, "model cannot be null"
    # if model is None: 
    #     print('status> computing w2v model ...')
    #     model = evalWordVec(docs, w2v_method=w2v_method)  # w2v_method: 'sg', 'cbow'

    docFeatureVecs = np.zeros((len(docs), model.vector_size), dtype="float32")
    # Loop through the (patient) docs
    n_doc = len(docs)
    for i, doc in enumerate(docs):
       if test_ and (i % 2000 == 0):
           print "+ computing doc #%d of %d via averaging" % (i, n_doc)
       docFeatureVecs[i] = getAverageVec(doc, model)
    return docFeatureVecs

def toStrings(docs, sep=' '): 
    """
    Input: a list of documents 'docs' each of which consists of a list of words/tokens 
           i.e. 'docs' is a list of lists 
    Output: a list of documents, each of which becomes a string connected via 'sep'
    """
    n_doc = len(docs)
    rid = random.randint(0, n_doc-1)
    assert isinstance(docs[rid], list)

    docx = []
    for i, doc in enumerate(docs): 
        ds = sep.join(doc)
        if i == rid: 
            dst = sep.join(doc[:50])
            print('verify> doc:\n%s\n => reconstructed doc:\n%s\n' % (doc[:50], dst))
        docx.append(ds)
    return docx

# [deprecated] use labeling directly
def labelDocByFreqDiag(seqx, **kargs):  # refactored from seqAnalyzer
    """
    Label each input document in 'seqx' by the most frequent diagnostic code. 
    """
    # import labeling
    return labeling.labelDocByFreqDiag(seqx, **kargs)

def makeD2VLabels(sequences, **kargs):   # refactored from seqAnalzyer
    """
    Label sequences/sentences for the purpose of using Doc2Vec. 

    Adapted from seqCluster.makeD2VLabels()

    New version available in labeling as labelize()

    Related 
    -------
    Base function of the wrapper labelDocuments()

    Memo
    ----
    The input to Doc2Vec is an iterator of LabeledSentence objects. Each such object represents a single sentence, 
    and consists of two simple lists: a list of words and a list of labels:
        
        sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])

    """ 
    # import labeling
    return labeling.makeD2VLabels(sequences, **kargs)

def labelize(docs, **kargs):
    """
    Label sequences/sentences for the purpose of using Doc2Vec. 

    Input: documents 
    Output: labeled documents

    Params
    ------
    label_type 
    class_labels: class labels

    Memo
    ----

    1. TaggedDocument (& deprecated LabeledSentence) ... 10.23.17

    a. The input to Doc2Vec is an iterator of LabeledSentence objects. Each such object represents a single sentence, 
    and consists of two simple lists: a list of words and a list of labels:
        
        sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])

    b. LabeledSentence is an older, deprecated name for the same simple object-type to encapsulate a text-example that is now called TaggedDocument. 
       Any objects that have words and tags properties, each a list, will do. (words is always a list of strings; 
       tags can be a mix of integers and strings, but in the common and most-efficient case, is just a list with a single id integer, 
       starting at 0.)

    """
    if isLabeled(docs): 
        print('vector.labelize> input already labeled!')
        return docs
    return labeling.labelize(docs, **kargs) # [params] label_type, class_labels, offset

def labelDocuments(docs, class_labels=[], label_type='doc', overwrite_=False, offset=0): 
    """
    Label the input documents; in particular, ensure that the document is labeled if not already. 
    Input: documents 
    Output: labeled documents
    No-op if 'docs' are already labeled and 'labels' is None (i.e. no need to assign new labels)

    Params
    ------ 
    labels: class labels (generally not the same as document labels)

    """
    def is_labeled():
        # is the document labeled?
        tval = False
        r = random.randint(0, len(docs)-1)
        try: 
            docs[r].tags  # assuming that the attribute name for labels is 'tags'
            tval = True
        except: 
            tval = False
        return tval
    def check_format(n_test=10): # ensure that each input doc is in list-of-tokens format  # [refactor] unit test
        idx = np.random.randint(len(docs), size=n_test)
        for i, r in enumerate(idx): 
            thisDoc = docs[r]
            if isinstance(thisDoc, gensim.models.doc2vec.TaggedDocument): 
                if i == 0: print('labelDocuments> input document is tagged')
                continue
            # if not isinstance(thisDoc, list): 
            if not hasattr(thisDoc, '__iter__'): 
                # could be np.ndarray
                raise ValueError, "Input document is not in list-of-tokens format:\n%s\n" % str(thisDoc)
            if len(thisDoc) == 0: 
                raise ValueError, "Input document (%d-th) is empty!" % r
        return
    def strip(): 
        unlabeledDocs = []
        for i, doc in enumerate(docs): 
            unlabeledDocs.append(doc.words)
        return unlabeldDocs
    
    check_format() # condition: docs are unlabeled, must be in list-of-tokens format

    if isLabeled(docs):  # statement won't work for np.array 
        nL = len(class_labels)
        if nL == 0: 
            print('labelDocuments> Noop: input already tagged.')
            return docs  # no-op
        else: # nL > 0 and overwrite_:  # overwrie labels? 
            if overwrite_: 
                print('labelDocuments> overwriting existing labeling via user-provided labels ...')
                docs = strip()
                assert nL == len(docs)
                # => labelize
            else: 
                print('labelDocuments> Noop: input documents already tagged > ignoring input labels ...')
                return docs
    else: 
        pass 
        # print('labelDocuments> input has not been labeled.')

    labeled_docs = docs
    if len(class_labels) > 0: 
        # labeling via input labels
        
        # the class-label generation part of this routine is not used now ... 10.26.17
        
        # labeled_docs = makeD2VLabels(sequences=docs, labels=labels) # can pass precomputed labels via 'labels'
        labeled_docs = labelize(docs, label_type=label_type, class_labels=class_labels)

    else: 
        # automatic labeling
        labeled_docs = labelize(docs, label_type=label_type)
    assert len(labeled_docs) > 0 

    return labeled_docs

def evalDocVec(docs, **kargs): # [output] model
    """
    
    Input: a list of docuemnts 'docs'
    Output: d2v model (not the vectorized document, which is given by getDocVec)

    Params
    ------
    d2v_method 

        Use D2V to configure the document embedding-specific paramters 
        
            n_features 
            window 
            n_workers 
            min_count
            n_iter: number of iterations over input documents (corpus)   

        If d2v_method in {average, tfidf, }, then only train w2v model

    Related
    -------
    evalWordVec: compate word vector 
    toDocVec: compute document vectors given model

    Memo
    ----
    1. Essentially, the vocabulary is a dictionary (accessible via model.wv.vocab) of all of the unique words extracted from 
    the training corpus along with the count (e.g., model.wv.vocab['penalty'].count for counts for the word penalty).

    2. To avoid common mistakes around the model’s ability to do multiple training passes itself, 
       an explicit epochs argument MUST be provided. In the common and recommended case, 
       where train() is only called once, the model’s cached iter value should be supplied as epochs value.

    """
    def show_params(): 
        msg = "Parameter Setting (d2v):\n"
        msg += '1.  number of features: %d\n' % n_features
        msg += '2.  window length: %d > total: 2 * window: %d\n' % (window, 2*window)
        msg += '3.  min count: %d\n' % min_count
        msg += '4.  PV method: %s\n' % d2v_method # "PV-DM" if dm == 1 else "PV-DBOW"
        msg += '4a. concat? %s\n' % str(dm_concat == 1)
        div(message=msg, symbol='%')
        return 

    # ensure that 'docs' are labeled (required to use gensim.Doc2Vec)  [note] should have been labeled at getDocVec level
    # labeled_docs = labelDocuments(docs, labels=kargs.get('labels', []), label_type=kargs.get('label_type', 'doc')) # labels? check_labeling?  

    # [params] d2v
    d2v_method = kargs.get('d2v_method', D2V.d2v_method) # 'pv-dm', 'pv-dbow'
    n_features, window = D2V.n_features, D2V.window
    n_workers = D2V.n_workers
    min_count = D2V.min_count
    n_iter = D2V.n_iter  # default from Word2Vec is 5 
    D2V.show_params()

    nDoc = len(labeled_docs); assert nDoc == len(docs)
    labeled_docs = np.array(labeled_docs) # need to use its indexing 

    model = None
    if d2v_method == 'pv-dm': # pv-dm only; distributed memory; each doc is treated as a token for which vector is to be derived
        dm = 1  
        # tUseD2V = True 
        # labeled_docs = makeD2VLabels(sequences=sequences) # labels: use labelDocByFreqDiag() to generate by default

        # PV-DM w/average
        model = Doc2Vec(dm=dm, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=5, 
                           hs=0, min_count=min_count, workers=n_workers, iter=n_iter)
            
        # [memo] 1
        model.build_vocab(labeled_docs) # [error] must sort before initializing vectors/weights
        print('evalDocVec> n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(labeled_docs), model.corpus_count, model.iter, D2V.n_iter))
        # word count: e.g. model.wv.vocab['250.00'].count 

        # model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
        for epoch in range(10):
            perm = np.random.permutation(nDoc)
            model.train(labeled_docs[perm], total_examples=model.corpus_count, epochs=model.iter)  # [memo] 2. epochs: number of passes to the training data
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay    
            
    elif d2v_method == 'pv-dbow': # (based on) bag of words model
        dm = 0
        # tUseD2V = True
        # labeled_docs = makeD2VLabels(sequences=sequences) # labels: use labelDocByFreqDiag() to generate by default

        # PV-DBOW 
        # [params] default dm_mean=0, dm_concat=0? 

        # can also train word vectors simultaneously
        # set 'dbow_words' to 1 => trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training
        # default is 0 (faster training of doc-vectors only).
        model = Doc2Vec(dm=dm, dm_mean=D2V.dm_mean, size=n_features,
                           dbow_words=0,  
                           negative=5, hs=0, 
                           min_count=min_count, 
                           workers=n_workers, 
                           iter=n_iter)
        model.build_vocab(labeled_docs) # [error] must sort before initializing vectors/weights
        print('evalDocVec> n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(labeled_docs), model.corpus_count, model.iter, W2V.n_iter))
        
        for epoch in range(10):
            perm = np.random.permutation(nDoc)
            model.train(labeled_docs[perm], total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay 

    # elif d2v_method in ('tfidf', 'tfidfavg', 'tf-idf', 'average', ): 
    elif D2V.isWordVectorBased(d2v_method): 
        # reversing to w2v model 

        model = evalWordVec(docs, **kargs)
        print('warning> d2v_method=%s > computed only w2v model (try using evalWordVec() instead) ...' % d2v_method)

    else: 
        raise NotImplementedError, "Unknown doc2vec method: %s" % d2v_method

    assert model is not None
    
    return model

def evalDocVecPVDM(docs, **kargs):  # evalDocVec<method>
    """
    Input: docs
    Output: d2v model

    Params
    ------
    sample: threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, values of 1e-5 (or lower) may also be useful, value 0. disable downsampling.

    hs: if 1, hierarchical softmax will be used for model training. 
        If set to 0 (default), and negative is non-zero, negative sampling will be used.

    """
    # from gensim.models import Doc2Vec

    # tUseD2V = True 
    # labeled_docs = makeD2VLabels(sequences=sequences) # labels: use labelDocByFreqDiag() to generate by default
    size = kargs.get('size', D2V.n_features)
    window_size = kargs.get('window', D2V.window)


    # PV-DM w/average
    model = Doc2Vec(dm=1, dm_mean=1, dm_concat=0, size=size, window=window_size, negative=5, 
                       sample=1e-5, 
                       hs=0, min_count=D2V.min_count, workers=D2V.n_workers, iter=D2V.n_iter)
            
    # [memo] 1
    model.build_vocab(labeled_docs) # [error] must sort before initializing vectors/weights
    print('evalDocVec> n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(labeled_docs), model.corpus_count, model.iter, D2V.n_iter))
    # word count: e.g. model.wv.vocab['250.00'].count 

    # model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    for epoch in range(10):
        perm = np.random.permutation(nDoc)
        model.train(labeled_docs[perm], total_examples=model.corpus_count, epochs=model.iter)  # [memo] 2. epochs: number of passes to the training data
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    return model      

def getDocVecPV(docs, **kargs): 
    """
    Compute document vector by combing PVDM and PV-DBOW. 
    All documents are trained at the same time (unlike getDocVecPV2 which separates documents into 
    train and test splits)

    Input: (labeled) documents 
    Output: document vectors (not the d2v model itself)

    kargs
    -----
    segment_by_visit: if True then label_type <- 'v'

    Params
    ------
    dm: training algorithm 
        By default (dm=1), ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.

    sample: threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, values of 1e-5 (or lower) may also be useful, value 0. disable downsampling.

    hs: if 1, hierarchical softmax will be used for model training. 
        If set to 0 (default), and negative is non-zero, negative sampling will be used.

    dm_mean = if 0 (default), use the sum of the context word vectors. 
              If 1, use the mean. Only applies when dm is used in non-concatenative mode.

    dm_concat = if 1, use concatenation of context vectors rather than sum/average; default is 0 (off). 
                Note concatenation results in a much-larger model, as the input is no longer the size of one 
                (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words in the context strung together.

    meta: meta data that serves as a secondary model ID

    n_epoch: number of passesof model training (which consists of multiple iterations determined by model.iter <- D2V.n_iter)

    
   
    Related
    -------
    getDocVecPV2: same but considers a train-test split 
  
    Memo
    ----
    1. train test split should be called prior to this routine. 
       x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    """
    def show_params(): 
        msg = "Parameter Setting (d2v):\n"
        msg += '1.  number of features: %d\n' % n_features
        msg += '2.  window length: %d > total: 2 * window: %d\n' % (window, 2*window)
        msg += '3.  min count: %d\n' % min_count

        d2v_method_eff = 'pv-dm+pv-dbow' if d2v_method == 'pv-dm' else d2v_method

        msg += '4.  PV method: %s\n' % d2v_method_eff # "PV-DM" if dm == 1 else "PV-DBOW"
        msg += '4a. concat? %s\n' % str(dm_concat == 1)
        div(message=msg, symbol='%')
        return 
    def model_type(): 
        # [condition] model is not None and has been well tested

        # PS: can also do isinstance(model, gensim.models.Doc2Vec) but then this won't generalize into non-gensim implementations
        return 'wv' if D2V.isWordVectorBased(d2v_method) else 'dv'
    def build_vocabulary(D):  # D_train, D_test
        
        print("   + (build_vocab) example doc: %s ~ type: %s" % (D[0], type(D[0])))  # np.array(list(), list())
        model_dm.build_vocab(D) 
        print('     + pv-dm>   n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(docs), model_dm.corpus_count, model_dm.iter, D2V.n_iter))
        model_dbow.build_vocab(D) 
        print('    + pv-dbow> n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(docs), model_dbow.corpus_count, model_dbow.iter, D2V.n_iter)) 
        return    
    def train_model(D, n_epoch=10): # given model_dm, model_dbow
        # [note] pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
        # [note] not sure if it's necessary to go through multiple passes of the data

        Xs = []
        # condition: models have been initialized
        print('  + .train_model> n_epochs=%d, epochs(inner): %d' % (n_epoch, model_dm.iter))

        # train over n epoch (necessary?)
        for epoch in range(n_epoch):

            # train DM
            model_dm.train(D, total_examples=model_dm.corpus_count, epochs=model_dm.iter) 
            model_dm.alpha -= 0.002  # decrease the learning rate
            model_dm.min_alpha = model_dm.alpha  # fix the learning rate, no decay   
            

            # train DBOW
            model_dbow.train(D, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)
            model_dbow.alpha -= 0.002 
            model_dbow.min_alpha = model_dbow.alpha
            
        # merge vectors
        Xs.append(toDocVec(D, model_dm))  # vectors from DM model 
        Xs.append(toDocVec(D, model_dbow)) # vectors from DBOW model 
        X = np.hstack(Xs) # joint vectors
         
        return X
    def get_vector(D): # get vectors given models
        Xs = [] 
        Xs.append(toDocVec(D, model_dm))  # vectors from DM model 
        Xs.append(toDocVec(D, model_dbow)) # vectors from DBOW model 
        return np.hstack(Xs) # joint vectors
    def get_model_identifier(): 
        # all the caller (e.g. getDocVec) to take care of this
        # note that if no model ID is given, the only model distinguising parameters would only be model parameters (e.g. n_features)
        # example model ID consitituents: ctype? cohort or group? 
        return kargs.get('meta', 'generic')
    def test_doc_labels(D): 
        docIDs = TDocTag.getDocIDs(D, pos=0)  # use the first element as the label/document ID 
        nuniq = len(np.unique(docIDs))
        ntotal = len(docIDs)
        assert nuniq == ntotal, "getDocVecPV> Prior to training > document IDs are not unique %d vs %d" % (nuniq, ntotal)
        return
    def verify_params(): 
        print('getDocVecPV> Params summary of d2v models (n_workers: %d, n_iter: %d) ...' % (D2V.n_workers, D2V.n_iter))
        msg = '  + n_features: %d, window: %d\n' % (D2V.n_features, D2V.window)
        msg += '  + negative sample? %s\n' % D2V.negative
        msg += '  + min count: %d\n' % D2V.min_count
        print msg

        return
         
    import random
    from labeling import TDocTag

    tTestModel = kargs.get('test_', False) or kargs.get('test_model', False)  # moved to getDocVec()
    tSaveModel = True
    
    tl_docs = isLabeled(docs)
    print('getDocVecPV> prior to labelDocuments, already labeled? %s, example: %s, type: %s' % (tl_docs, docs[0], type(docs[0])))

    # [note] if there's a distinction between train and test set 
    #        D_train = labelDocuments(D_train, label_type='train'); D_test = labelDocuments(D_test, label_type='test')
    labeled_docs = labelDocuments(docs, label_type='doc' if kargs.get('segment_by_visit', False) else 'v') # noop if inputs already labeled
    test_doc_labels(labeled_docs)

    # [params] d2v
    d2v_method = kargs.get('d2v_method', 'pv-dm2') # 'pv-dm2' # 'pv-dm' + 'pv-dbow'
    modelType = 'dv' # model_type()  # 'dv', 'wv', ...

    n_features, window = kargs.get('n_features', D2V.n_features), kargs.get('window', D2V.window)
    D2V.show_params()
    model_id = get_model_identifier() # a string that serves as model ID (that may share the same parameters)
    ### combine all input data 
    # [note] for training, can combine both docs and augmented_docs (e.g. docs have class labels but augmented are the unlabeled)
    
    # [note] D_train should be a list of TaggedDocument's
    
    nDoc = len(labeled_docs)
    print('getDocVecPV> total: %d' % nDoc)

    ### PV-DM + PV-DBOW
    # [note] hs=0 => negative sampling; dm_mean=1 => use mean of context word vectors
    # model_dm = kargs.get('model_dm', None)  # may want to continue on pretrained model (e.g. training set followed by test set)

    # attempt to load first
    # [note] do not simply use D2V.d2v_method because we have two algorithms corresponding to two different models here
    verify_params()
    tNewModel = False  # compute a new model? 
    model_dm = D2V.load(d2v_method='dm', outputdir=kargs.get('outputdir', None), suffix=model_id) if kargs.get('load_model', True) else None 
    if model_dm is None: 
        model_dm = Doc2Vec(dm=1, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=D2V.negative, 
                            hs=0, min_count=D2V.min_count, workers=D2V.n_workers, iter=D2V.n_iter)
        tNewModel = True
    model_dbow = D2V.load(d2v_method='dbow', outputdir=kargs.get('outputdir', None), suffix=model_id) if kargs.get('load_model', True) else None
    if model_dbow is None: 
        model_dbow = Doc2Vec(dm=0, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=D2V.negative, 
                                hs=0, min_count=D2V.min_count, workers=D2V.n_workers, iter=D2V.n_iter)
        tNewModel = True
      
    # docs_train, docs_test = np.array(D_train), np.array(D_test)  # need to use list indexing
    X = None
    n_epoch = kargs.get('n_epoch', 1)
    if tNewModel: 
        print('  + Computing d2v model ((pv-dm followed by pv-dbow)) on training corpus (n=%d)' % nDoc) 
        print('    + Building vocabulary for all coding sequences ...')     

        # docs = np.concatenate((D_train, D_test)) # array(list, list) => [error] 'numpy.ndarray' object has no attribute 'words'
        # docs = D_train + D_test
        assert isinstance(labeled_docs, list)
        
        # [note] if model has been trained would lead to [error] cannot sort vocabulary after model weights already initialized.
        build_vocabulary(labeled_docs)

        # vector.D2V.n_iter = 50
        X = train_model(labeled_docs, n_epoch=n_epoch)  # [params] model_dm, model_dbow
        assert X.shape[0] == nDoc
        assert X.shape[1] == 2 * n_features, "X_train.shape[1]: %d != 2 * n_features: %d" % (X.shape[1], 2*n_features)  
    
        # gc.collect()

        # save model? 
        if tSaveModel:  # after training on both train and test splits
            # e.g. tpheno/seqmaker/data/CKD/Mpv-dm-Pf50w5.d2v
            methods = [ ('dm', model_dm), ('dbow', model_dbow), ]
            for name, model in methods: 
                # suffix: combined (to distinguish from the models that separate train and test as those in getDocVecPV2())
                D2V.save(model, d2v_method=name, outputdir=kargs.get('outputdir', None), suffix=model_id) 
    else: 
        # use pre-computed models to compute vectors
        X = get_vector(labeled_docs)       

    if tTestModel: 
        assert model_dm is not None and model_dbow is not None
        assert model_dm.corpus_count == nDoc, "corpus_count: %d while nDoc: %d" % (model_dm.corpus_count, nDoc)
        # D = D_train + D_test
        D = labeled_docs
        
        # [note] this only evalutes the model individually but not on the level of joint vectors
        for model in [model_dm, model_dbow, ]: 
            print('getDocVecPV> testing model: %s' % model)
            assess(model=model, docs=D, cohort=kargs.get('cohort', '?'), labels=kargs.get('labels', []))

        # [todo] how to assess the joint vector? 
    return X 

# prefix evalDocVec + method: paragraph vector (PV): pv-dm (+ pv-dbow)
def getDocVecPV2(D_train, D_test, **kargs):
    """
    Similar to getDocVecPV() but used in classification with train-test split validation setting. 

    Input: traing corpus and test corpus 
    Output: a 2-tupe of document vectors, one for training split and one for test split

    """
    def show_params(): 
        msg = "Parameter Setting (d2v):\n"
        msg += '1.  number of features: %d\n' % n_features
        msg += '2.  window length: %d > total: 2 * window: %d\n' % (window, 2*window)
        msg += '3.  min count: %d\n' % min_count

        d2v_method_eff = 'pv-dm+pv-dbow' if d2v_method == 'pv-dm' else d2v_method

        msg += '4.  PV method: %s\n' % d2v_method_eff # "PV-DM" if dm == 1 else "PV-DBOW"
        msg += '4a. concat? %s\n' % str(dm_concat == 1)
        div(message=msg, symbol='%')
        return 
    def model_type(): 
        # [condition] model is not None and has been well tested

        # PS: can also do isinstance(model, gensim.models.Doc2Vec) but then this won't generalize into non-gensim implementations
        return 'wv' if D2V.isWordVectorBased(d2v_method) else 'dv'
    def build_vocabulary(D):  # D_train, D_test
        
        print("   + (build_vocab) example doc: %s ~ type: %s" % (D[0], type(D[0])))  # np.array(list(), list())
        model_dm.build_vocab(D) 
        print('     + pv-dm>   n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(docs), model_dm.corpus_count, model_dm.iter, D2V.n_iter))
        model_dbow.build_vocab(D) 
        print('    + pv-dbow> n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(docs), model_dbow.corpus_count, model_dbow.iter, D2V.n_iter)) 
        return    
    def train_model(D, n_epoch=10): # given model_dm, model_dbow
        # [note] pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
        # [note] not sure if it's necessary to go through multiple passes of the data

        Xs = []
        # condition: models have been initialized

        # train over n epoch (necessary?)
        for epoch in range(n_epoch):

            # train DM
            model_dm.train(D, total_examples=model_dm.corpus_count, epochs=model_dm.iter) 
            model_dm.alpha -= 0.002  # decrease the learning rate
            model_dm.min_alpha = model_dm.alpha  # fix the learning rate, no decay   
            

            # train DBOW
            model_dbow.train(D, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)
            model_dbow.alpha -= 0.002 
            model_dbow.min_alpha = model_dbow.alpha
            
        # merge vectors
        Xs.append(toDocVec(D, model_dm))  # vectors from DM model 
        Xs.append(toDocVec(D, model_dbow)) # vectors from DBOW model 
        X = np.hstack(Xs) # joint vectors
         
        return X
    def get_vector(D): # get vectors given models
        Xs = [] 
        Xs.append(toDocVec(D, model_dm))  # vectors from DM model 
        Xs.append(toDocVec(D, model_dbow)) # vectors from DBOW model 
        return np.hstack(Xs) # joint vectors
    def get_model_identifier(): 
        # all the caller (e.g. getDocVec) to take care of this
        # note that if no model ID is given, the only model distinguising parameters would only be model parameters (e.g. n_features)
        # example model ID consitituents: ctype? cohort or group? 
        return kargs.get('meta', 'generic')
         
    import random
    tTestModel = kargs.get('test_', False) or kargs.get('test_model', False)  # moved to getDocVec()
    tSaveModel = True
    
    tl_train, tl_test = isLabeled(D_train), isLabeled(D_test)
    print('getDocVecPV2> prior to labelDocuments, already labeled? %s, example: %s, type: %s' % (tl_train, D_train[0], type(D_train[0])))
    D_train = labelDocuments(D_train, label_type='train') # noop if inputs already labeled
    D_test = labelDocuments(D_test, label_type='test')
    # print('getDocVecPV2> after | example: %s' % str(D_train[0])) # D_train[0] is a TaggedDocument()

    # [params] d2v
    d2v_method = kargs.get('d2v_method', D2V.d2v_method) # 'pv-dm2' # 'pv-dm' + 'pv-dbow'
    modelType = model_type()  # 'dv', 'wv', ...

    n_features, window = kargs.get('n_features', D2V.n_features), kargs.get('window', D2V.window)
    D2V.show_params()

    model_id = get_model_identifier() # a string that serves as model identifier (embedded in the model file name)
    ### combine all input data 
    # [note] for training, can combine both docs and augmented_docs (e.g. docs have class labels but augmented are the unlabeled)
    
    # [note] D_train should be a list of TaggedDocument's
    
    nTrain, nTest = len(D_train), len(D_test)
    nDoc = nTrain + nTest
    print('getDocVecPV2> nTrain: %d + nTest: %d = total: %d' % (nTrain, nTest, nDoc))

    ### PV-DM + PV-DBOW
    # [note] hs=0 => negative sampling; dm_mean=1 => use mean of context word vectors
    # model_dm = kargs.get('model_dm', None)  # may want to continue on pretrained model (e.g. training set followed by test set)

    # attempt to load first
    # [note] do not simply use D2V.d2v_method because we have two algorithms corresponding to two different models here
    print('  + Initializing d2v models ...')
    tNewModel = False  # compute a new model? 
    model_dm = D2V.load(d2v_method='dm', outputdir=kargs.get('outputdir', None), suffix=model_id) if kargs.get('load_model', True) else None 
    if model_dm is None: 
        model_dm = Doc2Vec(dm=1, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=D2V.negative, 
                            hs=0, min_count=D2V.min_count, workers=D2V.n_workers, iter=D2V.n_iter)
        tNewModel = True
    model_dbow = D2V.load(d2v_method='dbow', outputdir=kargs.get('outputdir', None), suffix=model_id) if kargs.get('load_model', True) else None
    if model_dbow is None: 
        model_dbow = Doc2Vec(dm=0, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=D2V.negative, 
                                hs=0, min_count=D2V.min_count, workers=D2V.n_workers, iter=D2V.n_iter)
        tNewModel = True
      
    # docs_train, docs_test = np.array(D_train), np.array(D_test)  # need to use list indexing
    X_train = X_test = None  
    n_epoch = kargs.get('n_epoch', 1)
    if tNewModel: 
        print('  + Computing d2v model ((pv-dm followed by pv-dbow)) on training corpus (n=%d)' % nTrain) 
        print('    + Building vocabulary for all coding sequences ...')     

        # docs = np.concatenate((D_train, D_test)) # array(list, list) => [error] 'numpy.ndarray' object has no attribute 'words'
        docs = D_train + D_test
        assert isinstance(D_train, list) and isinstance(D_test, list) and isinstance(docs, list)
        
        # [note] if model has been trained would lead to [error] cannot sort vocabulary after model weights already initialized.
        build_vocabulary(docs)

        # [note] this will cause trouble with model saving later on 
        # for model in [model_dm, model_dbow, ]: 
        #     for epoch in range(10):
        #         # perm = np.random.permutation(nTrain)
        #         # model.train(docs_train[perm], total_examples=model.corpus_count, epochs=model.iter)  # [memo] 2. epochs: number of passes to the training data
                
        #         # random.shuffle(D_train)
        #         model.train(D_train, total_examples=model.corpus_count, epochs=model.iter)

        #         model.alpha -= 0.002  # decrease the learning rate
        #         model.min_alpha = model.alpha  # fix the learning rate, no decay   
        #     Xs.append(toDocVec(D_train, model))

        X_train = train_model(D_train, n_epoch=n_epoch)  # [params] model_dm, model_dbow
        assert X_train.shape[0] == nTrain
        assert X_train.shape[1] == 2 * n_features, "X_train.shape[1]: %d != 2 * n_features: %d" % (X_train.shape[1], 2*n_features)  
    
        gc.collect()

        print('    + computing d2v model on test corpus (n=%d)' % nTest) 
        X_test = train_model(D_test, n_epoch=n_epoch) # [params] model_dm, model_dbow
        assert X_test.shape[0] == nTest
        assert X_test.shape[1] == 2 * n_features, "X_test.shape[1]: %d != 2 * n_features: %d" % (X_test.shape[1], 2*n_features) 

        # save model? 
        if tSaveModel:  # after training on both train and test splits
            # e.g. tpheno/seqmaker/data/CKD/Mpv-dm-Pf50w5.d2v
            methods = [ ('dm', model_dm), ('dbow', model_dbow), ]
            for name, model in methods: 
                D2V.save(model, d2v_method=name, outputdir=kargs.get('outputdir', None), suffix=model_id)
    else: 
        # use pre-computed models to compute vectors
        X_train = get_vector(D_train)   
        X_test = get_vector(D_test)     

    if tTestModel: 
        assert model_dm is not None and model_dbow is not None
        assert model_dm.corpus_count == (nTrain + nTest), "corpus_count: %d while nDoc: %d" % (model_dm.corpus_count, nDoc)
        D = D_train + D_test
        
        # [note] this only evalutes the model individually but not on the level of joint vectors
        for model in [model_dm, model_dbow, ]: 
            print('getDocVecPV2> testing model: %s' % model)
            assess(model=model, docs=D, cohort=kargs.get('cohort', '?'))

        # [todo] how to assess the joint vector? 

    return (X_train, X_test)
    
def getDocVecPV0(docs, **kargs): # evalDocVec<method>
    """
    Compute document vector by combing PVDM and PV-DBOW. 

    Input: (labeled) documents 
    Output: document vectors (not the d2v model)

    Params
    ------
    dm: training algorithm 
        By default (dm=1), ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.

    sample: threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, values of 1e-5 (or lower) may also be useful, value 0. disable downsampling.

    hs: if 1, hierarchical softmax will be used for model training. 
        If set to 0 (default), and negative is non-zero, negative sampling will be used.

    dm_mean = if 0 (default), use the sum of the context word vectors. 
              If 1, use the mean. Only applies when dm is used in non-concatenative mode.

    dm_concat = if 1, use concatenation of context vectors rather than sum/average; default is 0 (off). 
                Note concatenation results in a much-larger model, as the input is no longer the size of one 
                (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words in the context strung together.


    Memo
    ----
    1. train test split should be called prior to this routine. 
       x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    """
    def show_params(): 
        msg = "Parameter Setting (d2v):\n"
        msg += '1.  number of features: %d\n' % n_features
        msg += '2.  window length: %d > total: 2 * window: %d\n' % (window, 2*window)
        msg += '3.  min count: %d\n' % min_count

        d2v_method_eff = 'pv-dm+pv-dbow' if d2v_method == 'pv-dm' else d2v_method

        msg += '4.  PV method: %s\n' % d2v_method_eff # "PV-DM" if dm == 1 else "PV-DBOW"
        msg += '4a. concat? %s\n' % str(dm_concat == 1)
        div(message=msg, symbol='%')
        return 

    # [chain] -> labelize() in absence user-defined labels 
    #      noop if labeled
    docs = labelDocuments(docs, labels=kargs.get('labels', []), label_type=kargs.get('label_type', 'doc')) # labels? check_labeling? 
    # assert isLabeled(docs, n_test=10), "input documents have not been labeled."

    # [params] d2v
    d2v_method = 'pv-dm2' # 'pv-dm' + 'pv-dbow'
    n_features, window = kargs.get('n_features', D2V.n_features), kargs.get('window', D2V.window)
    D2V.show_params()

    # combine all input data 
    # [note] for training, can combine both docs and augmented_docs (e.g. docs have class labels but augmented are the unlabeled)
    docs = docs; 
    augmented_docs = kargs.get('augmented_docs', [])
    nDoc0, nAug = len(docs), len(augmented_docs)
    if len(augmented_docs) > 0: 
        # docs = np.concatenate((docs, augmented_docs)) 
        assert isinstance(augmented_docs, list)
        docs = docs + augmented_docs
    nDoc = len(docs)
    print('getDocVecPV> nDoc(original): %d + nDoc(augmented): %d = total: %d' % (nDoc0, nAug, nDoc))
    # [reference]
    # model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    # model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # PV-DM + PV-DBOW
    # [note] hs=0 => negative sampling; dm_mean=1 => use mean of context word vectors
    # model_dm = kargs.get('model_dm', None)  # may want to continue on pretrained model (e.g. training set followed by test set)
    model_dm = Doc2Vec(dm=1, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=D2V.negative, 
                            hs=0, min_count=D2V.min_count, workers=D2V.n_workers, iter=D2V.n_iter)
    model_dbow = Doc2Vec(dm=0, dm_mean=1, dm_concat=0, size=n_features, window=window, negative=D2V.negative, 
                            hs=0, min_count=D2V.min_count, workers=D2V.n_workers, iter=D2V.n_iter)
      
    print('getDocVecPV> 1. Building vocabulary for all coding sequences ...')      
    model_dm.build_vocab(docs) 
    model_dbow.build_vocab(docs) 
    print('  + pv-dm>   n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(docs), model_dm.corpus_count, model_dm.iter, D2V.n_iter))
    print('  + pv-dbow> n_doc=%d, total_examples=%d, epochs=%d=?=%d' % (len(docs), model_dbow.corpus_count, model_dbow.iter, D2V.n_iter))
    # word count: e.g. model.wv.vocab['250.00'].count 

    # [note] pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    print('getDocVecPV> 2. Training D2V model (pv-dm followed by pv-dbow) ...') 
    Xs = []
    for model in [model_dm, model_dbow, ]: 
        for epoch in range(10):
            # perm = np.random.permutation(nDoc)
            # model.train(docs[perm], total_examples=model.corpus_count, epochs=model.iter)  # [memo] 2. epochs: number of passes to the training data
            
            model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay   
        Xs.append(toDocVec(docs, model))

    # concatenate dm-vector (Dim: n) and dbow-vector (n) => Dim: 2n
    X = np.hstack(Xs)
    assert X.shape[0] == nDoc
    assert X.shape[1] == 2 * n_features 

    # need to slice X if only the vectors of the non-augmented documents are needed
    # X = X[:nDoc0]

    return X 

def evalWordVec(docs, **kargs): # [output] model
    """
    Given a set of documents, compute w2v model where indexed tokens are mapped to vectors

    Related
    -------
    1. getWordVec: load, if not availabe then, evalWordVec 

    """
    def show_params(): 
        msg = "Parameter Setting (w2v):\n"
        msg += '1.  number of features: %d\n' % n_features
        msg += '2.  window length: %d > total: 2 * window: %d\n' % (window, 2*window)
        msg += '3.  min count: %d\n' % min_count
        msg += '4.  w2v method: %s\n' % w2v_method # "PV-DM" if dm == 1 else "PV-DBOW"
        msg += '4a. concat? %s\n' % str(dm_concat == 1)
        div(message=msg, symbol='%')
        return 
    def is_labeled():
        # is the document labeled?
        tval = False
        r = random.randint(0, len(docs)-1)
        try: 
            docs[r].tags  # assuming that the attribute name for labels is 'tags'
            tval = True
        except: 
            tval = False
        return tval
    def strip(): 
        unlabeledDocs = []
        for i, doc in enumerate(docs): 
            unlabeledDocs.append(doc.words)
        return unlabeldDocs

    # [params] d2v
    w2v_method = kargs.get('w2v_method', W2V.w2v_method)  # 'sg', 'cbow'
    sg = kargs.get('sg', None)

    n_features, window = W2V.n_features, W2V.window
    n_workers = W2V.n_workers
    min_count = W2V.min_count
    show_params()

    model = None
    if sg == 1 or w2v_method == 'sg': 
        # [todo] choose which negative sampls to use
        model = Word2Vec(docs, sg=1, size=n_features, negative=5, window=window, min_count=min_count, workers=n_workers)  # [1]
    elif sg == 0 or w2v_method == 'cbow': 
        model = Word2Vec(docs, sg=0, size=n_features, negative=5, window=window, min_count=min_count, workers=n_workers) 
    else: 
        raise NotImplementedError, "Unknown word2vec method: %s" % w2v_method

    return model

def evalVec(docs, d2v_method='pv-dm', w2v_method='sg'): # [output] model
    """
    Convent document into vector representations (return a model wherein indexed tokens, 
    including documents themselves are mapped to vectors)

    Params
    ------
    w2v_method or d2v_method 

    """
    method = d2v_method # kargs.get('d2v_mehthod', None)
    if method is not None: 
        # try d2v first, if not roll back to w2v
        if D2V.isSupported(method): 
            return evalDocVec(docs, d2v_method=method)
        else: 
            raise ValueError, "d2v method %s is not supported." % d2v_method
    else: 
        # then roll back to w2v
        assert W2V.isSupported(w2v_method), "w2v method %s is not supported." % w2v_mehthod

    print('evalVec> Unsupported d2v_method: %s > try w2v_method: %s' % (d2v_method, w2v_method))
    return evalWordVec(docs, w2v_mehthod=w2v_mehthod)

def vectorize(docs, **kargs):  # this is not the same as seqAnalyzer.vectorize which returns a model
    return getWordVecModel(docs, **kargs)
def getW2VModel(docs, **kargs): 
    return getWordVecModel(docs, **kargs)
def getWordVecModel(docs, **kargs):
    # mode = 'wv' # [options] 'wv': word vector 'dv': document vector

    # [params] w2v
    w2v_method = kargs.get('w2v_method', W2V.w2v_method) # 'sg', 'cbow'

    # [params] for saving the model
    # e.g. tpheno/seqmaker/data/<cohort>
    outputdir = kargs.get('outputdir', os.getcwd())  # sys_config.read('DataExpRoot'); try to use different output dirs ~ cohort, content type
    identifier = 'M%s-Pf%dw%d' % (w2v_method, W2V.n_features, W2V.window)
    ofile = kargs.get('fname', '%s.%s' % (identifier, 'w2v'))
    fpath = os.path.join(outputdir, ofile)
    load_model = kargs.get('load_model', True) and os.path.exists(fpath)
    
    tLoadSuccess, tComputeModel = True, False
    tSaveModel = True

    if load_model:
        try: 
            model = Word2Vec.load(fpath)  # can continue training with the loaded model! 

            if len(model.index2word) == 0: 
                print('warning> Loaded empty model from %s' % fpath)
                tLoadSuccess = False 
        except: 
            print('error> Could not load model from %s' % fpath)
            tLoadSuccess = False

    if not tLoadSuccess:
        model = evalWordVec(docs, **kargs)

        # save model? 
        if tSaveModel: 
            print('io> saving model (w2v: %s) to %s' % (w2v_method, fpath))
            model.save(fpath) 
    return model

def getWordVec(docs, **kargs):
    """
    Wrapper of evalWordVec(). 

    Get vector representations for each word. 
    This returns only word vectors (ordered according to a given criterion)
    To vectorize an entire document, use getDocVec()

    Input: documents 
    Output: key-value pairs 
            where keys: words (or documents) 
                  values: vectors 

    Related
    -------
    vectorize2 | getDocVec
    """
    mode = 'wv' # [options] 'wv': word vector 'dv': document vector

    model = getWordVecModel(docs, **kargs) # compute or load model + save

    adict = {}
    if mode == 'wv': # default
        n_tokens = 0
        index2word_set = set(model.index2word)
        
        words = docsToTokens(docs, sorted_=False)
        for word in words:
            if word in index2word_set: 
                n_tokens = n_tokens + 1.
                adict[word] = model.wv[word]
        assert n_tokens > 0
    elif mode == 'dv': # documents as tokens
        # n_doc = len(docs)
        # dmatrix = np.zeros( (n_doc, n_features), dtype="float32" )  # alloc mem to speed up
        for i, lseq in enumerate(labeled_docs): 
            v = model.docvecs[lseq.tags[0]]  # model.docvecs[i]
            assert v.shape[0] == model.vector_size
            adict[lseq.tags[0]] = v
            # dmatrix[i] = model.docvecs[lseq.tags[0]]  
    else: 
        raise NotImplementedError

    return adict

def vectorize2(docs, **kargs): 
    """

    Memo
    ----
    1. Naming convention compatibility with seqAnalyzer.vectorize (word embedding) seqCluster.vectorize2 (doc embedding)
    """
    return getDocVecModel(docs, **kargs)
def getD2VModel(docs, **kargs):
    return getDocVecModel(docs, **kargs) 
def getDocVecModel(docs, **kargs):  # load, compute, test
    """
    Given input documents (D), compute its d2v model

    Input: documents 
    Output: document embedding model

    Params
    ------
    d2v_method 
    w2v_method: only used with basic d2v_method such as 'average', 'tfidf', 
    outputdir: directory that keeps the model 
    

    """
    def check_format(n_test=10): # ensure that each input doc is in list-of-tokens format  # [refactor] unit test
        idx = np.random.randint(len(docs), size=n_test)
        for i, r in enumerate(idx): 
            thisDoc = docs[r]
            if isinstance(thisDoc, gensim.models.doc2vec.TaggedDocument): 
                if i == 0: print('+ input document is tagged')
                continue
            # if not isinstance(thisDoc, list): 
            if not hasattr(thisDoc, '__iter__'): 
                raise ValueError, "Input document is not in list-of-tokens format:\n%s\n" % str(thisDoc)
            if len(thisDoc) == 0: 
                raise ValueError, "Input document (%d-th) is empty!" % r
    def model_type(): 
        # [condition] model is not None and has been well tested

        # PS: can also do isinstance(model, gensim.models.Doc2Vec) but then this won't generalize into non-gensim implementations
        return 'wv' if D2V.isWordVectorBased(d2v_method) else 'dv'

    # [test]
    assert len(docs) > 0, "No input documents!"
    check_format(n_test=10) # input documents could also have been labeled

    # ensure that 'docs' are labeled (required to use gensim.Doc2Vec) => this is delegated to evalDocVec()
    # labeled_docs = labelDocuments(docs, labels=kargs.get('labels', []), label_type=kargs.get('label_type', 'doc')) # labels? check_labeling?  

    # [params] d2v
    d2v_method = kargs.get('d2v_method', D2V.d2v_method)
    w2v_method = kargs.get('w2v_method', 'sg') # only used when d2v_method in 'average', 'tfidf'
    
    # [params] for saving the model
    # e.g. tpheno/seqmaker/data/<cohort>
    outputdir = kargs.get('outputdir', os.getcwd())  # sys_config.read('DataExpRoot'); try to use different output dirs ~ cohort, content type
    identifier = 'M%s-Pf%dw%d' % (d2v_method, D2V.n_features, D2V.window)
    ofile = kargs.get('outputfile', '%s.%s' % (identifier, 'd2v'))
    fpath = os.path.join(outputdir, ofile)
    load_model = kargs.get('load_model', True) and os.path.exists(fpath)
    
    tTestModel = kargs.get('test_', False) or kargs.get('test_model', False)  # moved to getDocVec()
    tSaveModel = True
    model = None
    modelType = model_type()  # 'dv', 'wv', ...

    tLoadSuccess = False
    if load_model:
        try: 
            model = Doc2Vec.load(fpath)  # can continue training with the loaded model! 

            if len(model.docvecs) > 0: 
                tLoadSuccess = True 
            else: 
                print('warning> Loaded empty model from %s' % fpath)
                tLoadSuccess = False
        except: 
            print('error> Could not load model from %s' % fpath)
            tLoadSuccess = False

    if not tLoadSuccess: # then compute the model
        print('info> computing a new model (d2v=%s) ...' % d2v_method)
        # these methods need individual word vectors
        # note that pv-dbow does not give individual word vectors 

        # if d2v_method in ('average', 'tfidfavg', 'tfidf', ):  
        if modelType == 'dv': 
            print('getDocVecModel> Info: Applying d2v-based complex model (d2v=%s)' % d2v_method)
            assert isLabeled(docs, n_test=10), "input documents have not been labeled."

            # [note] evalDocVecPV combines both pv-dm and pv-dbow (instead of evalDocVec with separate cases)
            #        evalDocVecPV should support augmented training data (corpus)
            if kargs.has_key('augmented_docs'): print('getDocVecModel.evalDocVecPV> found augmented training corpus ...')
            model = evalDocVec(docs, **kargs) # only compute doc vectors and nothing more            
        elif modelType == 'wv':  
            # [note] can also use 'pv-dbow' and set dbow_words to 1

            # first compute word vectors
            print('getDocVecModel> Info: Applying w2v-based simple model (d2v=%s)' % d2v_method)
            model = evalWordVec(docs, w2v_method=w2v_method) # only compute word vectors, that's it
            modelType = 'wv'
        else: 
            raise ValueError, "Unknown model type: %s" % modelType

        # save model? 
        if tSaveModel: 
            # e.g. tpheno/seqmaker/data/CKD/Mpv-dm-Pf50w5.d2v
            print('io> saving model (method: %s, model type: %s) to %s' % (d2v_method, modelType, fpath))
            model.save(fpath)
    else: 
        print("getDocVecModel> Info: Loaded precomputed model (d2v=%s) ... " % d2v_method)

    if tTestModel: 
        assert model is not None

        # insert labels? 
        if modelType == 'dv': 

            # [note] should have been labeled by the caller getDocVec 
            # docs = labeled_docs = labelDocuments(docs, labels=kargs.get('labels', []), label_type=kargs.get('label_type', 'doc')) 
            # evaluate model with input 'docs'
            assess(model=model, docs=docs, cohort=kargs.get('cohort', '?'))
        else: 
            # noop 
            pass

    return model

def getDocVec2(D_train, D_test, **kargs): 
    """
    Similar to getDocVec but used in classification with train-test splits. 

    Params
    ------
    model_id: 
       a string used to distingulish different model files with the same paramters 
       e.g. pv-dm model trained for sequences in CKD cohort
          
            tpheno/seqmaker/data/CKD/Pf100w5i50-diag.dm
                vs 
            tpheno/seqmaker/data/CKD/Pf100w5i50.dm

    """
    def model_type(): 
        # [condition] model is not None and has been well tested

        # PS: can also do isinstance(model, gensim.models.Doc2Vec) but then this won't generalize into non-gensim implementations
        return 'wv' if D2V.isWordVectorBased(d2v_method) else 'dv'
        
    d2v_method = kargs.get('d2v_method', D2V.d2v_method)  # pv-dm is now a combination of pv-dm and pv-dbow
    modelType = model_type()

    ### load or compute d2v model
    nTrain, nTest = len(D_train), len(D_test)  # step 3
    X_train = X_test = None
    if d2v_method == 'pv-dm2':  # default for d2v-based method: pv-dm + pv-dbow
        
        # condition: labelDocuments() is included in the callee
        #
        # [test]
        # X = getDocVecPV(D_train+D_test, **kargs)
        # print('getDocVec2.test> d2v using entire corpus => dim: %s' % str(X.shape))
        # sys.exit(0)
        tCombineTrainTest = kargs.get('combine_train_test', False)
        if not tCombineTrainTest: 
            X_train, X_test = getDocVecPV2(D_train, D_test, **kargs) # [design] no need to separate compute-model and vectorize operations
        else: 
            X_train = getDocVecPV(D_train+D_test, **kargs)

    elif d2v_method.startswith('glo'): # GloVe
        raise NotImplementedError, "GloVe model has not been implemented yet."

    elif d2v_method.startswith(('seq', 'lstm', )): 
        raise NotImplementedError, "Seq2seq model has not been implemented yet."

    elif modelType == 'dv': # label + d2v + doc -> vec
        # documents should be labeled here (assuming that Word2Vec also takes in labeled documents)
        X = []
        
        corpus = [(D_train, 'train'), (D_test, 'test'), ]
        for D, label_type in corpus: 
            labeled_docs = labelDocuments(D, label_type=label_type) # labels? check_labeling? 

            # D -> M i.e. (labeled) documents (Dl) to model (M)
            model = getDocVecModel(labeled_docs, **kargs)  # load or compute d2v + save (step 1 & 2)
            # condition: documents are labeled 
            X.append(toDocVec(labeled_docs, model=model)) # throws exception if docs are not labeled
        
        X_train, X_test = X[0], X[1]

    elif modelType == 'wv': # D2V.isWordVectorBased(d2v_method)

        X = []
        corpus = [(D_train, 'train'), (D_test, 'test'), ]
        for D, label_type in corpus: 
            model = getWordVecModel(D, **kargs)  # load or compute d2v + save (step 1 & 2)
            # assumption: w2v model; need to compute document representation based on word vectors
            X.append(toSimpleDocVec(docs, model=model, d2v_method=d2v_method)) # throws exception if docs are not labeled

        X_train, X_test = X[0], X[1]

    else: 
        raise NotImplementedError, "Unrecognized model type: %s" % modelType

    return (X_train, X_test) 

def getDocVecBow0(docs, **kargs): 
    """
    Get document vectors using bag-of-words model. 

    Note
    ----
    1. Input words need to be sorted to maintain consistency. 

    Memo
    ----
    1. scipy.sparse.lil_matrix: row-based linked list sparse matrix
       This is a structure for constructing sparse matrices incrementally.

    """ 
    def to_freq_vector(doc, vec): # <- sortedTokens
        # modify vec based on word counts in doc
        wc = collections.Counter()
        wc.update(doc)
        for i, var in enumerate(sortedTokens):
            if var in wc:   
                vec[i] = wc[var]
            else: 
                pass
        return vec
    def get_counts(doc, feature_set):
        wc = collections.Counter()
        wc.update(doc)

        col, data = [], []
        for i, v in enumerate(feature_set): 
            if v in wc: 
                col.append(i)
                data.append(wc[v])
        return (col, data)
    def test_vector(doc, vec):  # nF <- 
        z = np.zeros(nF, dtype="float32")    
        assert len(doc) == 0 or LA.norm(vec-z) > 0     

    from numpy import linalg as LA  # for testing
    import collections 
    from scipy.sparse import csr_matrix # coo_matrix

    tokenCounts = collections.Counter()   
    for i, doc in enumerate(docs):
        tokenCounts.update(doc)
    print('getDocVecBow> Got %d tokens' % len(tokenCounts))

    # determine feature set 
    maxNTokens = kargs.get('max_features', None)  # set Nose to use all
    tokenSet = [tok for tok, _ in tokenCounts.most_common(maxNTokens)]
    tokenCounts = None; gc.collect()
    sortedTokens = sorted(tokenSet)
    nD, nF = len(docs), len(sortedTokens)
    
    # d2v_bow = np.zeros((nD, nF), dtype="float32") # this can be too big => MemoryError
    # X = csc_matrix(np.zeros((nD, nF), dtype="float32"))

    print('getDocVecBow> X(n_docs by n_tokens: %d by %d)' % (nD, nF))  # e.g. CKD cohot has 223090 (223K) tokens

    # sample only a subset of the documents [todo]
    tsDType = kargs.get('tset_dtype', 'sparse')
    if tsDType.startswith('d'):  # dense format
        # set a maximum number of tokens to reduce memory requirement
        X = []
        for i, doc in enumerate(docs):
            vec = np.zeros(nF, dtype="float32")
            X.append(to_freq_vector(doc, vec)) # sortedTokens 
            if i % 100 == 0: test_vector(doc, vec)
        return np.array(X)

    # sparse format by default 
    row, col, data = [], [], []
    for i, doc in enumerate(docs):
        idx, cnts = get_counts(doc, sortedTokens)  # column indices, counts
        row.extend([i] * len(idx))  # ith document
        col.extend(idx)  # jth attribute
        data.extend(cnts)  # count 
    Xs = csr_matrix((data, (row, col)), shape=(nD, nF))
    print('getDocVecBow> Found %d coordinates, %d active values, ' % (len(data), Xs.nnz))

    return Xs # scipy.sparse.csc.csr_matrix

def getDocVecBow(docs, **kargs):
    """

    Memo
    ----
    1. keras.preprocessing.text.Tokenizer input arguments:

       num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words words will be kept.
       filters: a string where each element is a character that will be filtered from the texts. 
                The default is all punctuation, plus tabs and line breaks, minus the ' character.

       lower: boolean. Whether to convert the texts to lowercase.
       split: str. Separator for word splitting.
       char_level: if True, every character will be treated as a token.
       oov_token: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls

    2. small CKD cohort 
        - 2360 by 20719 (=?= 20714-1 unique tokens)
    """
    def to_str(D, sep=' ', throw_=False):
        try: 
            D = [sep.join(doc) for doc in D]  # assuming that each element is a string
        except: 
            msg = 'getDocVecBow> Warning: Some documents may not contain string-typed tokens!'
            if throw_: 
                raise ValueError, msg 
            else: 
                print msg
            D = [sep.join(str(e) for e in doc) for doc in D]
        return D
    def separate_index_counts(X):
        for x in X: 
            col, data = [], []
            for i, v in enumerate(feature_set): 
                if v in wc: 
                    col.append(i)
                    data.append(wc[v])

    # from keras.preprocessing.text import Tokenizer
    import docProc 
    from numpy import linalg as LA  # for testing
    import collections 
    from scipy.sparse import csr_matrix # coo_matrix

    tokenCounts = collections.Counter()   
    for i, doc in enumerate(docs):
        tokenCounts.update(doc)
    nF0 = len(tokenCounts)

    # determine feature set 
    maxNTokens = kargs.get('max_features', None)  # set Nose to use all
    tokenCounts = None; gc.collect()
    
    t = docProc.tokenize(to_str(docs), lower=False, num_words=kargs.get('num_words', None), oov_token=kargs.get('oov_token', None)) 
    nF = len(t.word_index)
    nD = len(docs)
    print('getDocVecBow> X: n_docs by n_tokens => %d by %d (=?= %d-1)' % (nD, nF, nF0))  # e.g. CKD cohot has 223090 (223K) tokens

    X = t.texts_to_matrix(docs, mode=kargs.get('text_maxtrix_mode', 'tfidf')) # values: 'binary', 'count', 'tfidf', 'freq'
    Xs = csr_matrix(X)
    print('getDocVecBow> Found %d coordinates, %d active values, ' % (Xs.shape[1], Xs.nnz))

    return Xs

def getDocVec(docs, **kargs):   # getDocVec -> evalDocVec
    """
    Map documents to vectors.

    Wrapper of evalDocVec; this routine takes care of other side effects other than computing vector itself (e.g. saving model)
    
        1. compute document embeddings 
        2. save model 
        3. vectorize the input documents (docs)

    Precedence: Try d2v-based method first if specified, if not roll back to w2v-based method (followed by averaging, tf-idf weighting, etc, 
                to compute document vectors)


    Input
    -----
    documents (where each document is a list of tokens)

    kargs
    ----- 
    d2v_method 
    
    model_id: 
       a string used to distingulish different model files with the same paramters 
       e.g. pv-dm model trained for sequences in CKD cohort
          
            tpheno/seqmaker/data/CKD/Pf100w5i50-diag.dm
                vs 
            tpheno/seqmaker/data/CKD/Pf100w5i50.dm
    
    Output
    ------
    document embeddings (where each document is represented by a vector)

    Related
    -------
    vectorize | getWordVec

    Memo
    ----
    1. Abstraction on d2v to allow for easy intergrations with different implementations. 

    2. PV-DM/pv-dm vs PV-DBOW/pv-dbow

       PV-DBOW doesn't involve any NN-input-vectors per-word. There's one vector for the text, which alone is used to predict each individual word. 
       So while word-vectors still get randomly initialized (simply because of the way gensim code is shared with word2vec and other modes), 
       they're NOT updated during training at all, and are still at their random initial values at the end of training. 

       The DBOW training is very analogous to the Word2Vec "skip-gram" mode, but using vector(s) for the text-as-a-whole to predict target words, 
       rather than just vector(s) for nearby-words. 

       In comparison, Mikolov's demo '-sentence-vectors' patch to word2vec.c always includes word-training, 
       because it uses an artificial pseudo-word at the start of each example, special-cased to participate in every 'window', 
       as its sentence-vector-tag.

    3. Doc2Vec 

       dm: defines the training algorithm. By default (dm=1), ‘distributed memory’ (PV-DM) is used. 
           Otherwise, distributed bag of words (PV-DBOW) is employed.
       
       dm_mean: if 0 (default), use the sum of the context word vectors. 
                If 1, use the mean. Only applies when dm is used in non-concatenative mode.

       dm_concat: if 1, use concatenation of context vectors rather than sum/average; default is 0 (off). 
                   Note concatenation results in a much-larger model, as the input is no longer the size of one 
                   (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words in the context strung together.
    """
    def test_model(): 
        assert model is not None
        # evaluate model with input 'docs'
        seqx = docs if modelType == 'wv' else labeled_docs
        assess(model=model, docs=seqx, labels=kargs.get('labels', []), cohort=kargs.get('cohort', '?'))
        return 
    def model_type(): 
        # [condition] model is not None and has been well tested

        # PS: can also do isinstance(model, gensim.models.Doc2Vec) but then this won't generalize into non-gensim implementations
        return 'wv' if D2V.isWordVectorBased(d2v_method) else 'dv'
    def verify_params(): 
        L = kargs.get('labels', []) 
        if len(L) > 0: 
            if tSegmentVisit: 
                print('getDocVec> nL: %d <? nDoc (i.e. n_visit_segments): %d' % (len(L), len(docs)))
                assert len(L) <= len(docs), "nL: %d but nD: %d" % (len(L), len(docs))
            else: 
                assert len(L) == len(docs), "nL: %d but nD: %d" % (len(L), len(docs))
            print('getDocVec> labels (L) given for assessing similarity ...')
        # T = kargs.get('timestamps', [])
        # if len(T) > 0: 
        #     assert len(T) == len(docs), "nT: %d but nD: %d" % (len(T), len(docs))
        return

    d2v_method = kargs.get('d2v_method', D2V.d2v_method)  # pv-dm is now a combination of pv-dm and pv-dbow
    modelType = model_type()

    ### load or compute d2v model

    n_doc = len(docs)   # step 3
    tSegmentVisit = kargs.get('segment_by_visit', False)
    if d2v_method == 'pv-dm2':  # default for d2v-based method: pv-dm + pv-dbow

        # [note] 1. getDocVecPV transform docs to vectors directly
        #        params: vector.D2V, meta (secondary model ID: e.g. ctype)
        #        2. can pass labels to getDocVecPV -> assess
        print('getDocVec> method: %s, model ID: %s, segment_by_visit? %s, load precomputed? %s' % \
            (d2v_method, kargs.get('meta', 'generic'), tSegmentVisit, kargs.get('load_model', False)))
        verify_params()  # L, T

        # getDocVec* is a series of functions that transform documents into vectors
        dmatrix = getDocVecPV(docs, **kargs) # [design] no need to separate compute-model and vectorize operations

    elif d2v_method == 'bow': # bag-of-words model
        print('getDocVec> method: %s, model ID: %s, tset_dtype: %s' % (d2v_method, kargs.get('meta', 'generic'), kargs.get('tset_dtype', 'sparse')))
        print('           max_features: %s, texts_to_matrix_mode: %s' % \
            (kargs.get('max_features', 'unbounded'), kargs.get('text_maxtrix_mode', 'tfidf')))
        dmatrix = getDocVecBow(docs, **kargs)

    elif d2v_method.startswith('glo'): # GloVe
        raise NotImplementedError, "GloVe model has not been implemented yet."

    elif d2v_method.startswith(('seq', 'lstm', )): 
        raise NotImplementedError, "Seq2seq model has not been implemented yet."         

    ### older cases that are distinguished via modelType
    elif modelType == 'dv': # label + d2v + doc -> vec
        # documents should be labeled here (assuming that Word2Vec also takes in labeled documents)
        labeled_docs = labelDocuments(docs, labels=kargs.get('labels', []), label_type=kargs.get('label_type', 'doc')) # labels? check_labeling? 
        model = getDocVecModel(labeled_docs, **kargs)  # load or compute d2v + save (step 1 & 2)
        # condition: documents are labeled 
        dmatrix = toDocVec(labeled_docs, model=model) # throws exception if docs are not labeled

    elif modelType == 'wv': # D2V.isWordVectorBased(d2v_method)

        model = getWordVecModel(docs, **kargs)
        # assumption: w2v model; need to compute document representation based on word vectors
        dmatrix = toSimpleDocVec(docs, model=model, d2v_method=d2v_method)

    else: 
        raise NotImplementedError, "Unrecognized model type: %s" % modelType

    assert dmatrix.shape[0] == len(docs)

    return dmatrix

def assess2(models, docs, **kargs): 
    """
    Similar to assess but with more than one input model. 
    """
    # condition: docs have to be labeled
    docIDs = TDocTag.getDocIDs(docs, pos=0)  # use the first element as the label/document ID

    for i, doc in enumerate(docs):
        # doc_id = docIDs[i]  # document label
        components= []
        for model in models: 
            components.append(model.infer_vector(doc.words)) # labeled => has words attribute
        inferred_vector = np.hstack(docvec)

    # [todo]
    return 

def assess(model, docs, **kargs):
    """
    Used precomputed model to 'reconstruct' the input documents (e.g. train corpus) and test 
    if the reconstructed document vectors are sufficiently similar to existing 
    document vectors. 

    Params
    ------
    topn_match: given a target item, where item can be a document or a word, 
                consider the list of similar items a match if topn (say n=10) most similar items are associated with 
                the same label as that of the target. 

    <obsolete> labels are reserved for class labels, which are not the same as document labels
    labels: document labels 
            a. integer indices (0, num_doc-1)
            b. meaningful labels (e.g. classification)

    Memo
    ----
    1. size of the model: model.docvecs

    Reference 
    --------- 
    1. D2V tutorial: https://markroxor.github.io/gensim/static/notebooks/doc2vec-lee.html

    """
    def select_doc(label, n=1): 
        # select n documents of the given label
        selected = []
        for i, docID in enumerate(docIDs): 
            if len(selected) >= n: break
            if docID == label: 
                # dx.append(i)
                selected.append(docs[i])
        
        assert len(selected) > 0, 'No document matches label=%s | docIDs:%s' % (label, docIDs[:10])   
        return selected
    
    def verify_docID(r, labelmap=None):
        # docID, label = docIDs[r], L[r]   # this will not work if documents are actually visit segments
        docID = docIDs[r]

        tSegmentVisit = False
        label = None
        if len(L) > 0: 
            try: 
                label = L[r]
            except: 
                tSegmentVisit = True 
                # print('access> Warning: number of labels %d < number of documents: %d (doc = visit semgent?)' % (len(L), len(docIDs)))

        if label is not None and isinstance(docID, str): 
            if labelmap is None: labelmap = seqparams.System.label_map

            sep = TDocTag.label_sep # '_'
            dlabel, index = docID.split(sep)  # e.g. CKD G1A1-control_48375

            # map (class) label to its desired/canonical value (e.g. {CKD Stage 3a, CKD Stage 3b} -> CKD Stage 3)
            dlabel = labelmap.get(dlabel, dlabel)
            clabel = labelmap.get(label, label)
            assert dlabel == clabel, "document(label): %s but class label: %s" % (dlabel, clabel)
        else: 
            pass

        return 
    def parse_label(docID, sep='_', labelmap={}):
        dlabel, index = docID.split(sep)
        return labelmap.get(dlabel, dlabel)  # e.g. {CKD Stage 3a, CKD Stage 3b} -> CKD Stage 3
    def sim_match_ratio(r, sims, topn=10, labelmap={}, sep='_'): # sims, docIDs
        # among those top similar docs, ratio of docs sharing the same label type 
        # (e.g. CKD Stage 3a_0 should be similar to itself and another CKD Stage 3a_i doc)
        
        # assuming that docIDs are available
        docID = docIDs[r] # if len(docIDs) > 0 else r
        if not labelmap: labelmap = seqparams.System.label_map
        dlabel0 = parse_label(docID, sep=sep, labelmap=labelmap)
        
        cnt = 0
        target_sims = sims[:topn] # may not be really that many in general 
        topn_eff = len(target_sims) 
        for docid, score in target_sims: 
            # tLabelMatched = False
            dlabel = parse_label(docid, sep=sep, labelmap=labelmap)  # dlabel is effectively a class label 
            if dlabel == dlabel0: 
                cnt += 1

        r = cnt/(topn_eff+0.0)
        return r
    def sim_match_ratio2(r, sims, topn=10, labelmap={}, sep='_'):   # L
        if not labelmap: labelmap = seqparams.System.label_map

        # assuming that len(L) > 0
        clabel0 = L[r]; clabel0 = labelmap.get(clabel0, clabel0)

        cnt = 0
        target_sims = sims[:topn] # topn most similar documents wrt r-th doc; may not be really that many in general 
        topn_eff = len(target_sims) 
        for docid, score in target_sims: 
            dlabel = parse_label(docid, sep=sep, labelmap=labelmap)  #L[docid]
            if dlabel == clabel0: 
                cnt += 1

        r = cnt/(topn_eff+0.0)
        return r

    def isin_topn(r, sims, topn=10, labelmap={}, sep='_'): # if a match is found within the topn similar items, then consider it a match
        if not labelmap: labelmap = seqparams.System.label_map

        # assuming that docIDs are available
        docID = docIDs[r] # if len(docIDs) > 0 else r
        if not labelmap: labelmap = seqparams.System.label_map
        dlabel0 = parse_label(docID, sep=sep, labelmap=labelmap)

        target_sims = sims[:topn] # topn most similar documents wrt r-th doc; may not be really that many in general 
        # topn_eff = len(target_sims) 
        tLabelMatched = False
        bestRank = 0
        for i, (docid, score) in enumerate(target_sims): 
            dlabel = parse_label(docid, sep=sep, labelmap=labelmap)  #L[docid]    
            if dlabel == dlabel0: 
                tLabelMatched = True
                bestRank = i 
                break
        # r = cnt/(n_target_sims+0.0)
        return (tLabelMatched, bestRank)     
    def save_plot_label_type_test(ratios, doc_ids=[], fname=None):  # prefix: save_plot_...
        # assert len(ratios) == nDoc, "Each doc should have 1 ratio (got only %d ratios)" % len(ratios)
        
        # this is messy! 
        # df = DataFrame(ratios, columns=['hit_ratio', ]) # ratio of hit
        # df['docID'] = pd.Series(list(range(len(df))))

        # discretize ratios
        bins = np.linspace(0, 1, 11)
        delta = bins[1]-bins[0]
        digitized = np.digitize(ratios, bins)
        # bin_means = [ratios[digitized == i].mean() for i in range(1, len(bins))]

        # convert ratios to counts (level vs counts in that level)
        header = ['hit_ratio', 'n_docs', ]    
        adict = {h:[] for h in header}
        assert len(ratios) == len(digitized)
        # adict['doc_id'] = doc_ids
        for level, cnt in collections.Counter(digitized).items(): 
            adict['hit_ratio'].append(level*delta+bins[0])  # approx. range of hit ratio (ie among top N similar docs, fraction of same label type)
            adict['n_docs'].append(cnt)
        df = DataFrame(adict, columns=header) # ratio of hit
        print('save_plot> created label similarity ratio dataframe ...')
        
        plt.clf()
        # fig = df.plot(x='docID', y='hit_ratio', kind='hist') 
        fig = df.plot(x='hit_ratio', y='n_docs', kind='bar') 

        if fname is None: fname = 'label_sim_ratio_distribution-N%s.tif' % len(doc_ids)
        if not os.path.exists(outputdir): 
            print('save_plot> creating new directory: %s' % outputdir)
            os.mkdir(outputdir)
        fpath = os.path.join(outputdir, fname)
        plt.savefig(fpath) 
        print('save_plot> saved label test result to %s' % fpath)
        return
    def test_doc_labels(): # <- docs (assumed labeled), docIDs
        nuniq = len(set(docIDs))  
        ntotal = len(docIDs)
        if ntotal > nuniq: 
            msg = "Document IDs are not unique (%d vs %d)" % (ntotal, nuniq)
            counter = collections.Counter(docIDs)
            for i, (di, cnt) in enumerate(counter.items()): 
                if i < 10: 
                    print('  + docID: %s, count: %d >' % (di, cnt))
            
            adict = {}
            for i, doc in enumerate(docs): 
                seq = doc.words
                dl = doc.tags[0]
                if not adict.has_key(dl): adict[dl] = []

                front, back = seq[:100], seq[150:200]
                abridged = front+['...']+back if len(back) > 0 else front
                adict[dl].append(abridged)

            adict = sutils.sample_dict(adict, n_sample=10)
            for i, (dl, seq) in enumerate(adict.items()): 
                print('  + dl: %s =>\n%s\n' % (dl, seq))

            raise ValueError, msg
        else: 
            print('Document labels are unique => ok.')
        return
    def get_outputdir(rootdir='test'): 
        # save output to local cohort directory, which is underneath ./data/<cohort>
        basedir = seqparams.getCohortDir(sysparams.cohort)  
        fpath = os.path.join(basedir, rootdir)
        if not os.path.exists(fpath): 
            print('assess> Creating new directory: %s' % fpath)
            os.mkdir(fpath)
        return fpath
    def summarize_similarity_test(ranks, doc_ids=[]): # <- topNMatchRatio, topNMatchTest, topNMatchTestComplete, nDoc, L
        """
        Similarity tests. Given a document, find its most similar document set and inspect their labels. 

        Test
        ----
        1. Given a test document (d) and its top N most similar documents {d1, d2, ... dn}: 
           a. can we expect to find at least a document of the same label? 
           b. try different N 
           c. accuracy
           d. self similarity: the rank position of d should ideally be 0 (itself) but can we at least find it within top N position? 
        """
        nD = len(doc_ids)
        if not doc_ids: doc_ids = range(0, len(ranks))

        # topNMatchTest {1/0}, topNMatchRatio [0,1]
        save_plot_label_type_test(topNMatchRatio)  # [params] fname, (outputdir), (doc_ids)

        # [temp]
        r_accuracy = sum(topNMatchTest)/(nD+0.0)
        print('assess> Similarity test accuracy %f (wrt topn=%d)\n' % (r_accuracy, topNBase))
        print('   + similarity test in comprehensive topn (e.g. n=1, 3, 5, ...)')
        for top_n in [1, 3, 5, 10, ]: 
            tvals = topNMatchTestComplete.get(top_n, [])
            r_accuracy = sum(tvals)/(nD+0.0)
            if tvals: 
                print('    + accuracy (wrt topn=%d) => %f' % (top_n, r_accuracy))

        # use ranks 
        print('\nassess> Similarity test via ranks data structure ...\n')
        rankCounts = collections.Counter(ranks)  # Results vary due to random seeding and very small corpus_count

        # [output] example: Counter({0: 289, 1: 11}) => 95%, I am ranked highest
        #          CKD with 11 labels: # {0: 1545, 1: 288, 2: 245, 3: 173, ... 10: 36}  => ranked top 1545 times
        div(message='Rank self similarity (cohort=%s):\n%s\n' % (kargs.get('cohort', '?'), rankCounts))
        for top_n in [1, 3, 5, 10, ]:
            sT = 0.0

            # how many times was an item ranked top_n? 
            for r in range(top_n): # top3 includes top 1, 2 and 3
                sT += rankCounts[r]  # how many times did an item assume rank 'r' (e.g. top 3, r=3)
            print('   + ratio of self similarity (ideally 1.0): %f vs within top %d: %f' % (rankCounts[0]/(nD+0.0), top_n, sT/(nD+0.0)))

        return

    import random, collections
    from labeling import TDocTag
    import pandas as pd
    from system import utils as sutils
    # from pandas import DataFrame
    from seqparams import System as sysparams  # system parameters (which )

    ranks = []
    second_rankscores = []

    isLabeledDocs = True if isLabeled(docs, n_test=10) else False
    if not isLabeledDocs: 
        raise ValueError, 'Input documents are not labeled yet.'

    d2v_method = kargs.get('d2v_method', '?') # this may not be always D2V.d2v_method
    # docs can be train corpus or test corpus
    nDoc = len(docs)
    docIDs = TDocTag.getDocIDs(docs, pos=0)  # use the first element as the label/document ID
    print('assess> nDoc: %d, len(docIDs): %d (example: %s)' % (nDoc, len(docIDs), docIDs[0]))
    L = kargs.get('labels', [])
    # condition: len(docIDs) == len(docs)

    outputdir = kargs.get('outputdir', get_outputdir(rootdir='test'))  # os.path.join(os.getcwd(), 'assess')

    # if len(set(docIDs)) == len(docIDs): isUniqueDocID = True
    test_doc_labels()
    # assert len(set(docIDs)) == len(docIDs), "Document IDs are not unique (%d vs %d)" % (len(set(docIDs)), len(docIDs))

    # len(model.docvecs) returns the number of unique docuemnt labels (e.g. eMerge CKD: 11 labels)
    print('assess> d2v_method=%s, labeled? %s, docID unique? must be yes' % (d2v_method, isLabeledDocs))
    print('          + n_doc: %d >=? len(model.docvecs): %d | cohort=%s' % (nDoc, len(model.docvecs), kargs.get('cohort', '?')))
    print('          + docIDs:\n%s\n' % docIDs[:20])
 
    n_modeled_docs = len(model.docvecs) # size of the d2v model
    print('assess> n_docs: %d, n_docIDs: %d, n_modeled: %d' % (nDoc, len(docIDs), n_modeled_docs))
    
    topNMatchRatio, topNMatchTest, topNBase = [], [], kargs.get('topn_match', 10)
    topNMatchTestComplete = {}  # test different possible matches within top_n most similar items where n is a list e.g. [1, 3, 5, 10]
    
    nDocTest = 1000   # <<< 
    docIDSampled = random.sample(range(nDoc), min(nDocTest, nDoc))  # pick ith documents where i = ... 
    nDocSampled = len(docIDSampled)
    sim_match_routine = sim_match_ratio2 if len(L) > 0 else sim_match_ratio
    testIDs = set(random.sample(docIDSampled, min(100, len(docIDSampled))))
    for i, r in enumerate(docIDSampled): # foreach rth doc 

        # take r-th element of docID, D, L
        docID = docIDs[r]  # document label
        doc = docs[r]
        # label = L[r] if len(L) > 0 else None
        verify_docID(r) # docIDs, L

        # prediction (as if input doc is a new training instance)
        inferred_vector = model.infer_vector(doc.words) # docs[doc_id].words

        # most similar to least similar
        sims = model.docvecs.most_similar([inferred_vector], topn=n_modeled_docs)  # len(model.docvecs): number of unique labels
            
        # condition: should be able to find doc_id (document label) in 'sims', a list of 2-tuple of (doc_id, sim score)
        rank = [docid for docid, sim_score in sims].index(docID)  # find the rank position of myself (doc_id), hopefully the first (0)
        ranks.append(rank) # rank position among all (similar) items
        second_rankscores.append(sims[1]) # (docID, score) for second most similar doc to the inferred vector

        # [test] ratio of same labels in topN similar documents  
        r_matched = sim_match_routine(r, sims, topn=topNBase, 
            labelmap=seqparams.System.label_map, sep=TDocTag.label_sep) # cf: isin_topn(docID, sims, topn=10, labelmap=None)
        topNMatchRatio.append(r_matched)
        tTopN, bestRank = isin_topn(r, sims, topn=topNBase, labelmap=seqparams.System.label_map, sep=TDocTag.label_sep) # 1 if r_matched > 0.0 else 0
        topNMatchTest.append(tTopN)

        # comprehensive topn-match test
        for top_n in [1, 3, 5, 10]: 
            if not topNMatchTestComplete.has_key(top_n): topNMatchTestComplete[top_n] = []
            tTopNp, bestRankp = isin_topn(r, sims, topn=top_n, labelmap=seqparams.System.label_map, sep=TDocTag.label_sep)
            topNMatchTestComplete[top_n].append(tTopNp)

        # [test]
        if i in testIDs: 
            tHit = 'yes' if tTopN == 1 else 'no'

            # rank: identical to self, best rank: as long as the label is identical
            n_ = 20
            print('  + docID=%s | rank=(%d, relaxed: %d), hit? %s(within topn=%d), top-%d sims:\n%s\n' % \
                (docID, rank, bestRank, tHit, topNBase, n_, sims[:n_]))
        else: 
            if i % 20 == 0: print('  + processing %dth doc / total= %d' % (i, nDocSampled))

    print('assess> completed documents similarity ranking ...')
    
    summarize_similarity_test(ranks, doc_ids=docIDSampled) # <- topNMatchRatio, topNMatchTest, topNMatchTestComplete
    
    # [test]
    isLabeledDocs = True 
    nTrials = 10 
    docPos = {docIDs[i]:i for i in range(nDoc)} # ID to document position
    for j in range(nTrials): 

        # doc positions/indices sampled from docIDs
        i = random.choice(range(len(docIDSampled))) # random.sample(range(nDoc), 1)[0] or random.randint(0, nDoc-1): end points included
        if isLabeledDocs: 
            r = docIDSampled[i] # ith randomly sampled doc ID is in rth position (of docIDs)
            docID = docIDs[r] # document label
            div(message='1. Most similar [trial #%d] ...' % j) 
            print('> Document (label={}): «{}»\n'.format(docID, ' '.join(docs[r].words)))
            print(u'> SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
            for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
                simDocID = sims[index][0] # document label of the similar document {(docID, score)}
            
                # can pick any document of label 'simDocId'
                # d = select_doc(simDocId, n=1) # among all the second-ranked documents, pick those of doc label=simDocId
                # simDoc = d[0]
                simDoc = docs[docPos[simDocID]] # doc ID -> doc position (in docIDs) -> doc

                print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(simDoc.words)))
    
            # second most similar 
            div(message='2. Second most similar ...')
            # Compare and print the most/median/least similar documents from the train corpus
            print('> Train Document (label={}): «{}»\n'.format(docID, ' '.join(docs[r].words)))
            
            # i-th randomly selected document's second-ranked most similar document: (docID, score)
            simR2DocID = second_rankscores[i][0] # (docID, score) second most similar document (doc_id, score) to the di-th document

            # can pick any document of label 'simDocId'
            # d = select_doc(simR2DocID, n=1) # among all the second-ranked documents, pick those of doc label=sim_id
            # simDoc = d[0]

            simDoc2 = docs[docPos[simR2DocID]]
            print('> Second most similar document (label={}): «{}»\n'.format(simR2DocID, ' '.join(simDoc2.words)))
            print('  + similarity score of the second ranked documents should be much lower ...')
        else: 
            raise ValueError
            # else: # not labeled, list of list of tokens/strings
            #     docId = docIDs[di] # document label

    return 

def toSimpleDocVec(docs, model, d2v_method='tfidf'): # document vectors based only w2v model
    """
    Compute document vectors given pre-computed word vectors (from the input w2v model). 

    Input: documents (unlabeld) + w2v model
    Output: document embeddings (where each document is repr. by a vector)

    Related 
    -------
    toDocVec 
    getDocVec: compute model + save model + vectorize

    """
    if len(docs) == 0: 
        print('warning> no input documents.')
        return np.array([])

    # condition: input model is a w2v model
    # d2v_method = kargs.get('d2v_method', 'tfidf')
    assert isinstance(docs[0], list), "Format: Input document should be a list of tokens: %s" % docs[0]
    assert D2V.isWordVectorBased(d2v_method), "Not a w2v-based model type: %s" % d2v_method

    # need to compute document represntation separately
    dmatrix = None
    if d2v_method == 'average': 
        dmatrix = byAvg(docs, model=model, test_=True) # precondition: model has been established
    elif d2v_method in ('tfidf', 'tfidfavg', 'tf-idf', ): 
        dmatrix = byTfidf(docs, model=model, test_=True)
    else: 
        raise NotImplementedError, "Unsupported (simple) d2v model: %s" % d2v_method

    return np.array(dmatrix)

def isLabeled(docs, n_test=10):  # use labeling.TDocTag.isLabeled()
    # ensure that documents were already labeled
    nD = len(docs)
    if not nD: 
        print('warning> empty corpus!')
        return False

    is_labeled = False
    idx = np.random.randint(nD, size=n_test)  # sampling with replacement
    for n, i in enumerate(idx): 

        # test two essential attributes of a labeled document (via gensim.models.doc2vec.TaggedDocument
        try: 
            labels = docs[i].tags  # ith document; each document is a namedtuple 
            assert len(labels) > 0, "isLabeled> empty labeling for doc: %s" % docs[i][:50]
            # can also test doc.words
            is_labeled = True
        except: 
            is_labeled = False 
            # break
        
        # condition: trust that the 'tags' attribute is consistent throughout all documents 
        if is_labeled or (n > n_test): break
        
    # assert is_labeled, "Documents are not labeled yet. Try using makeD2VLabels()"
    return is_labeled

def toDocVec(corpus, model, **kargs):
    """

    Params
    ------
    pos: position in the tags that serves as the document ID; 0 by default
    """
    # docs = corpus
    size = kargs.get('size', model.vector_size)
    pos = kargs.get('pos', 0)

    if kargs.get('check_labeling', True): assert isLabeled(corpus, n_test=10), "Input docs are not labeled!"

    vecs = [np.array(model.docvecs[z.tags[pos]]).reshape((1, size)) for z in corpus]
    dmatrix = np.concatenate(vecs) # dim: (N, size)  N=len(corpus)
    assert dmatrix.shape[0] == len(corpus) 
    assert dmatrix.shape[1] == size
    return dmatrix  # np.array 

def toDocVec0(docs, model, **kargs):
    """
    Find vector representation of the input docs given pre-computed d2v model. 

    Input: labeled documents + d2v model
    Output: document vectors
    
    Related
    -------
    evalDocVec: compute document vectors (without a precomputed d2v model)
    
    getDocVec: load, evalDocVec, toDocVec 
        try loading the model and if not found, then 
            evalDocVec
        once model is available, then do
            toDocVec

    """
    assert model is not None
    if len(docs) == 0: 
        print('warning> no input documents.')
        return np.array([])

    # ensure that documents were already labeled
    isUniqueDocID = False
    if kargs.get('check_labeling', True): 
        assert isLabeled(docs, n_test=10), "Input docs are not labeled!"
        # assert not D2V.isWordVectorBased(d2v_method), "Not a w2v-based model type: %s" % d2v_method
        docIDs = [doc.tags[0] for doc in docs]
        if len(set(docIDs)) == len(docIDs): isUniqueDocID = True

    n_doc = len(docs)
    dmatrix = np.zeros( (n_doc, model.vector_size), dtype="float32" )  # alloc mem to speed up
    if isUniqueDocID: 
        for i, lseq in enumerate(docs): 

            # condition: assuming doc IDs are integers (check labeling.makeD2VLabels)
            if i == 0: assert all(model.docvecs[i] == model.docvecs[lseq.tags[0]]) 
            dmatrix[i] = model.docvecs[i]    # lseq.tags to get labels  lseq.words to get sequence  
            # dmatrix[i] = model.docvecs[lseq.tags[0]]  
    else: 
        for i, lseq in enumerate(docs): 
            # Documents of the same tag has the same document vector
            dmatrix[i] = model.docvecs[lseq.tags[0]]    # lseq.tags to get labels  lseq.words to get sequence  
            # dmatrix[i] = model.docvecs[lseq.tags[0]]  

    return np.array(dmatrix)  # coerce into np array

def getFeatureVectors(docs, model=None, n_features=None, **kargs):
    """
    Get document vectors given model. 

    This is not the same as getDocVec, getWordVec, for which model is assumed not given and therefore 
    it needs to be either loaded or comptued. 

    """
    if n_features is None: 
        n_features = W2V.n_features
    else: 
        print('config> setting vector dimension to %s' % n_features)
        W2V.n_features = n_features

    # model is given? 
    if model is None: 
        return getDocVec(docs, **kargs)  # model is not given 

    # model is precomputed
    d2v_method = kargs.get('d2v_method', None)  # None for no assumption 
    if d2v_method is None or D2V.isWordVectorBased(d2v_method): 
        return toSimpleDocVec(docs, model=model, **kargs)

    # model is available, assuming to be d2v model
    # condition: documents are properly labeled
    return toDocVec(docs, model=model, **kargs)

def consolidateVisits(X, y, docIDs, **kargs):
    """
    Consolidate visit vectors for regular classifier

    Input
    -----
    * docIDs: Expanded document IDs e.g. [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, ...] 
                   => 0th doc has 3 sessions, 1th has 2 sessions, etc.
              
              Essentially, docIDs here contain session IDs in which repeated IDs of the same number reference
              the same document

    * last_n_visits: used to determine the cosolidated document vector by concatinating this many 
                     session vectors to form a single (consolidated) document vector. 

                     e.g. if we look at the last 10 sessions, each of which is represnted by a 100-D vector 
                          then we get a 10 * 100 = 1000-D vector for each document

                          documents with fewer sessions than 10 will be padded zeros to the front, thereby 
                          maintaining the consistency of document vector dimensionality

    kargs
    -----
    concat_last_n 
    W: (normalized) visit weight which gauges the importance of a visit


    Memo
    ----
    1. sequence.pad_sequences(vx, maxlen=100, dtype='float32')

    """
    def eval_dimensionality(X): 
        # use case: check dimensionality of concatenated visit vectors
        dims = []
        for x in X: 
            dims.append((len(x)+0.))
        print("(verify) mean dim: %f, median: %f, std: %f" % (np.mean(dims), np.median(dims), np.std(dims)))
        print('(verify) dim(X[0]): %s, dim(X): %s' % (str(X[0].shape), str(X.shape)))
        return
    def summary(): 
        print('(summary) number of total docs: %d' % nDoc0)
        print('... number of selected docs: %d > total number of session: %d ' % (nDoc, nVGrand))
        print('... Contatenated vector dim: %d | lastN=%d, indv fDim=%d' % (fDimPrime, lastN, fDim))

        return
    def subsample(docIDs, n=20000):
        uidx = np.unique(docIDs)
        ndoc = len(uidx)
        return set(random.sample(uidx, min(n, ndoc)))

    from keras.preprocessing.sequence import pad_sequences

    # policy 1: Simply concatenate the last N visits (e.g. last 10 visits)  <<< in use now
    # policy 2: take weighted average of visit vectors (but how to determine weights?)  # [todo]

    nDoc = nDoc0 = len(np.unique(docIDs))  # note that docIDs here are actually session IDs
    nDocMax = kargs.get('max_n_samples', None)
    nVGrand = len(docIDs)  # total number of visits in all documents
    lastN = kargs.get('last_n_visits', 10)  # consider only the last N visists
    weights = kargs.get('W', [])  # visit weights (e.g. normalized tf-idf score -> token, then take max)
    fDim = len(X[0])  # e.g. each visit is embedded into a 100 vector space
    fDimPrime = lastN * fDim  # e.g. 10 * 100 = 1000, assuming that we use last-N-visist vector contatenation to repr each document

    selectedIDs = []
    if nDocMax is not None: 
        idset = subsample(docIDs, n=nDocMax) # only consider n unique docIDs
        selectedIDs = sorted(idset)
        nDoc = len(selectedIDs)
    summary()
    # assuming that docIDs are sorted in ascending order 
    # e.g. docIDs <- [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, ] => 6+1=7 documents

    id0 = docIDs[0]
    idx = [0, ]  # after this look => [0, 3, 5, 6, 10, 12, 17, 19]
    for i, docid in enumerate(docIDs): 
        if docid != id0: 
            idx.append(i) # end index
            id0 = docid
    
    # add the last one  
    idx.append(nVGrand) # [0, 3, 5, 6, 10, 12, 17] + 19 => [0, 3, 5, 6, 10, 12, 17, 19]
    print('... visit idx:\n%s\n(check) scope idx:\n%s\n' % (docIDs[:100], idx[:100]))
    # scopes = []  # [test]
    
    Xp = []
    # number of scoping indices should be nDoc + 1 (counting first and last [0 and 19])
    nCuts = len(idx); assert nCuts == nDoc+1, "nCuts: %d <> nDoc+1: %d" % (nCuts, nDoc+1)
    if not selectedIDs: 
        for i in range(nCuts): 
            # if i == nCuts-1: break # i.e. i+1 out of bound
            if i+1 < nCuts: 

                # scopes.append((idx[i], idx[i+1]))  # [(0, 3), (3, 5), (5, 6), (6, 10), (10, 12), (12, 17), (17, 19)]
                n_visits = idx[i+1]-idx[i]  # i-th doc spans from idx[i] to idx[i+1] of the visit sequence (docIDs which is actually expanded docIDs)
                x = X[idx[i]:idx[i+1]].ravel()
     
                # [test]
                # print('  ... i=%d => n visits: %d > idx[i]:%d, idx[i+1]:%d => x- :%s | dim: %s' % \
                #     (i, n_visits, idx[i], idx[i+1], X[idx[i]: idx[i+1]], str(X[idx[i]:idx[i+1]].shape)))
                # print(' ... dim(x):%s | x[:100]:%s' % (str(x.shape), X[idx[i]:idx[i+1]].ravel()))

                # # if less than lastN visits in total, then we need to pad 0s to the front 
                # if n_visits < lastN: # e.g. np.pad(A, (2, 0), 'constant') => pad two 0's to the front and none in the back
                #     n_zeros = fDim * (lastN - n_visits)
                #     x = np.pad(x, (n_zeros, 0), 'constant') 
            
                Xp.append(x)
        X = np.array(Xp)
    else:  # only keep the vectors of the selected documents
        # idset = set(selectedIDs)
        # for i in range(nCuts): 
        #     if i+1 < nCuts and (i in idset): 
        #         n_visits = idx[i+1]-idx[i]  # i-th doc spans from idx[i] to idx[i+1] of the visit sequence (docIDs which is actually expanded docIDs)
        #         x = X[idx[i]:idx[i+1]].ravel()
        #         Xp.append(x)
        # y = np.array(y)[selectedIDs]
        # assert len(Xp) == len(selectedIDs), "number of vectors: %d but desired: %d" % (len(Xp), len(selectedIDs))
        # assert len(Xp) == len(y)

        # alternatively but more memory-consuming 
        for i in range(nCuts): 
            if i+1 < nCuts: 
                n_visits = idx[i+1]-idx[i]  # i-th doc spans from idx[i] to idx[i+1] of the visit sequence (docIDs which is actually expanded docIDs)
                x = X[idx[i]:idx[i+1]].ravel()
                Xp.append(x)
        X = np.array(Xp)[selectedIDs]; Xp=None; gc.collect()
        y = np.array(y)[selectedIDs]

        assert len(X) == len(selectedIDs), "number of vectors: %d but desired: %d" % (len(Xp), len(selectedIDs))
    
    eval_dimensionality(X)
    ### pad 0 to the front be default; i.e. pre-padding, if exceeding maxlen => truncate, take lastN's portion

    # this could result in memory shortage when fDimPrime is too big
    X = pad_sequences(X, maxlen=fDimPrime, dtype='float32')  # padding='post' to pad to the back
    # print('(check) after padding, dim(Xp): %s\n   + example:\n%s\n' % (str(Xp.shape), Xp[random.randint(0, len(Xp)-1)]))

    # now verify dimensions
    assert X.shape[0] == len(y)
    assert X.shape[0] == nDoc, "New X should have %d row but got %d" % (nDoc, X.shape[0])        
    assert X.shape[1] == fDimPrime, "New X should have %d features but got %d" % (fDimPrime, X.shape[1])         

    # e.g. (last) 10 visits, each repr by a 100-D vector
    # then x: 10 * 100 = 1000-D vector
    return (X, y)

def consolidateVisitsLSTM(X, y, docIDs, **kargs):
    """
    Consolidate visit vectors for LSTM (which requires inputs in 3D array)

    Input
    -----
    * docIDs: Expanded document IDs e.g. [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, ...] => 0th doc has 3 visits, 1th has 2 visits, etc.

    kargs
    -----
    concat_last_n 
    W: (normalized) visit weight which gauges the importance of a visit


    Memo
    ----
    1. sequence.pad_sequences(vx, maxlen=100, dtype='float32')

    """
    from keras.preprocessing.sequence import pad_sequences

    # policy 1: Simply concatenate the last N visits (e.g. last 10 visits)  <<< in use now
    # policy 2: take weighted average of visit vectors (but how to determine weights?)  # [todo]

    nDoc = len(np.array(docIDs).unique())
    nVGrand = len(docIDs)  # total number of visits in all documents
    lastN = kargs.get('last_n_visits', 10)  # consider only the last N visists
    weights = kargs.get('W', [])  # visit weights (e.g. normalized tf-idf score -> token, then take max)
    fDim = len(X[0])  # e.g. each visit is embedded into a 100 vector space
    fDimPrime = lastN * fDim  # e.g. 10 * 100 = 1000, assuming that we use last-N-visist vector contatenation to repr each document

    # assuming that docIDs are sorted in ascending order 
    # e.g. docIDs <- [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, ] => 6+1=7 documents

    id0 = docIDs[0]
    idx = [0, ]  # after this look => [0, 3, 5, 6, 10, 12, 17, 19]
    for i, docid in enumerate(docIDs): 
        if docid != id0: 
            idx.append(i) # end index
            id0 = docid
    # add the last one  
    idx.append(nVGrand) # [0, 3, 5, 6, 10, 12, 17] + 19 => [0, 3, 5, 6, 10, 12, 17, 19]
    
    # scopes = []  # [test]
    Xp, yp = [], []
    # number of scoping indices should be nDoc + 1 (counting first and last [0 and 19])
    nCuts = len(idx); assert nCuts == nDoc+1, "nCuts: %d <> nDoc+1: %d" % (nCuts, nDoc+1)
    for i in range(nCuts): 
        # if i == nCuts-1: break # i.e. i+1 out of bound
        if i+1 < nCuts: 
            # scopes.append((idx[i], idx[i+1]))  # [(0, 3), (3, 5), (5, 6), (6, 10), (10, 12), (12, 17), (17, 19)]
            n_visits = idx[i+1]-idx[i]  # i-th doc spans from idx[i] to idx[i+1] of the visit sequence (docIDs which is actually expanded docIDs)
            x = X[range(idx[i], idx[i+1])].ravel()

            # pad 0 to the front be default; i.e. pre-padding, if exceeding maxlen => truncate, take lastN's portion
            x = pad_sequences(x, maxlen=fDimPrime, dtype='float32')  # padding='post' to pad to the back

            Xp.append(x)
    Xp = np.array(Xp)

    # now verify dimensions
    assert Xp.shape[0] == nDoc, "New X should have %d row but got %d" % (nDoc, Xp.shape[0])

    # convert to 3D format that LSTM units expect
    Xp = Xp.reshape((nDoc, lastN, fDim))  # n_samples, n_timesteps, n_features    
    assert Xp.shape[0] == len(y)

    return Xp

def getWeightedAvgVec(tokens, model, weights): # ~ getAverageVec
    return weightedAvg(tokens, model, weights)
def weightedAvg(tokens, model, weights): 
    """
    Given a set of vectors (from a sentence), and their weights (e.g. td-idf scores), 
    find their weighted average

    Input
    -----
    * weights: a dictionary mapping from feature/token to tfidf score 

    """
    # doc = words = tokens
    index2word_set = set(model.index2word)

    # if n_features is None: 
    #     # use modeled word to infer feature dimension 
    #     for word in index2word_set: 
    #         fv = model.wv[word]
    #         n_features = fv.shape[0]
    #         break
    # featureVec = np.zeros((n_features,), dtype="float32")

    n_tokens = 0 
    wx = []  # weight vector
    vecs = [] # word vector
    for token in tokens: 
        if token in index2word_set:
            wx.append(weights[token])  # if indexed in the model, then it should have a weight
            wv = model.wv[token]
            assert len(wv) == model.vector_size
            vecs.append(wv)  # The word vectors are stored in a KeyedVectors instance in model.wv
            n_tokens = n_tokens + 1
    assert n_tokens > 0, "None of the tokens is in the model!\n  + doc: %s" % tokens[:100]
    # print('weightedAvgTfidf> w:\n%s' % wx)  # CKD: weights all identical

    wAvg = vecs
    try: 
        wAvg = np.average(np.asarray(vecs), weights=wx, axis=0)
    except Exception, e:

        # [test]
        div(message='Something went wrong in computing tf-idf-weighted average! Diagnosing ...\n')
        wx = [weights.get(token, 0) for w in index2word_set] 
        print('  + tfidf on modeled words: %s' % wx[:100])  # CKD: all has same weights
        print('  + index2word set: %s' % list(set(model.index2word))[:50])

        print('  + vec (n_dim: %d) =>\n    %s\n' % (len(vecs), vecs[:100])) # 'vecs' is empty
        print('  + wx  (n_dim: %d) =>\n    %s\n' % (len(wx), wx[:100]))

        for i, (tok, w) in enumerate(weights.items()): 
            if i < 100: 
                print('  + %s => tfidf: %f' % (tok, w))

        print('--- \n') 
        raise ValueError, e
    
    # return np.average(np.asarray(vecs), weights=wx, axis=0)
    return wAvg

def evalTfidfScores(docs, **kargs):
    """
    Compute TF-IDF scores for tokens in the document set ('docs'). 

    Input: documents (where each document is a list of tokens)
    Output: a dictionary mapping from tokens (in 'docs') to TF-IDF scores

    """
    def t_tfidf(): # [cond] adict; i.e. called after adict/weights matrix has been computed
        scored = []
        for doc in docs: 
            n_scored_per_doc = 0
            for w in doc: 
                if adict.has_key(w): 
                    n_scored_per_doc += 1
            r = n_scored_per_doc/(len(doc)+0.0)
            scored.append(r)

        mr = np.mean(scored)
        print('tfdif_test> average tfidf scored ratio: %f' % mr)
        print('            + example ratios (of tfidf-scored tokens): %s' % scored[:50])

        return mr 
    def to_lower(doc):  # not necessary if vocabulary is given to TfidfVectorizer
        for i, tok in enumerate(doc): 
            doc[i] = str(tok).lower()
        return
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from batchpheno import sampling

    docx = toStrings(docs)  # turn lists to strings so that they are compatible with TfidfVectorizer 
    tokens = kargs.get('tokens', None)
    min_doc_freq = kargs.get('min_df', 1)
    if tokens is None: tokens = docsToTokens(docs, uniq=True, sorted_=False)
    n_tokens = len(tokens) # number of unique tokens

    # [note] need to provide tokens because otherwise diag codes like 250.01 may not be considered as tokens 
    vectorizer = TfidfVectorizer(min_df=min_doc_freq, vocabulary=tokens)   
    X = vectorizer.fit_transform(docx) # X is in sparse matrix format
    invDocFreq = vectorizer.idf_
    # info
    stop_symbols = []
    try: 
        stop_symbols = vectorizer.stop_words_  # This is only available if no vocabulary was given.
    except: 
        print('info> vocabulary was given.')

    # tokens to weights
    fW = dict(zip(vectorizer.get_feature_names(), invDocFreq)) # features to weights (fW)

    # [test]
    print('getTfidfAvgFeatureVecs> n_features: %d (n_tokens: %d), n_stop_words: %d' % (len(fW), n_tokens, len(stop_symbols))) 
    print('verify> example maps:\n%s\n' % sampling.sample_dict(fW, n_sample=10))
    if kargs.get('test_', False): t_tfidf()

    return fW

def vectorizeByTfidf(docs, model, **kargs): 
    return byTfidf(docs, model, **kargs)
def byTfidf(docs, model, **kargs): 
    """

    Note
    ----
    1. input 'docs' must be converted to a list of strings for TfidfVectorizer to work

    """
    n_features = model.vector_size
    
    # [params] TfidfVectorizer()
    # min_df = kargs.get('min_df', 1)
    # max_df = kargs.get('max_df', 1.0)
    # max_features = kargs.get('max_features', None)

    # tokens to weights
    fW = evalTfidfScores(docs, **kargs)  # [params] min_df, tokens, test_, {}

    n_doc = len(docs)
    docFeatureVecs = np.zeros((len(docs), n_features), dtype="float32")  # alloc memory for speedup
    for i, doc in enumerate(docs): 
        if i % 5000 == 0:
            print "+ computing doc #%d of %d via tfidf-averaging" % (i, n_doc)

        # [note] CKD cohort: "Weights sum to zero, can't be normalized
        docFeatureVecs[i] = weightedAvg(doc, model, weights=fW) # make feature vectors by (tfidf-)weighted average     

    return docFeatureVecs

def create_bag_of_centroids(wordlist, word_centroid_map):
    """
    The function above will give us a numpy array for each review, each with a number of features 
    equal to the number of clusters.

    doc => [c1, c2, c3]: [8, 17, 26]  frequency of occurrences of cluster N:{1, 2, 3}
    """
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids

def docsToTokens(seqx, uniq=False, sorted_=False, reverse=False): 
    """
    Convert a set of documents to a set of tokens where 
    each docuemnt consists a list of tokens. 

    Input: a list of documents, each of which is repr by a list of tokens (NOT strings)

    Related
    -------
    1. tokenize(): read + docsToTokens

    """
    if uniq: 
        tokens = set() 
        for seq in seqx:  
            tokens.update(seq)  
        tokens = list(tokens)       
    else: 
        tokens = seq_to_token(seqx)
    
    if sorted_: 
        tokens = sorted(tokens, reverse=reverse) # ascending order by default

    return tokens  
def seq_to_token(seqx):
    tokens = []
    for i, seq in enumerate(seqx):  
        if i < 3: assert hasattr(seq, '__iter__'), "each document should be represented as a list of tokens:\n%s\n" % seqx[:5]
        tokens.extend(seq) 
    return tokens  

# already exist in seqAnalyzer ... 
def read(**kargs):
    """
    Input
    -----
    simply_code: 
       250.00 => 250
       749.13 => 749

    Memo
    ----
    1. already exist in seqAnalyzer but replicated here for convenience (ideally seqAnalyzer shall depends on analyzer but not in reverses)

    Log
    ---
    1. number of documents (patients): 432,000
       + number of sentences (visits): 15,755,982
       + number of (unique) tokens: 200,310  
       ... 02.20.17

    """
    # import seqReader as sr
    doctype = 'seq'
    basedir = sys_config.read('DataIn') 
    ofile = kargs.get('ofile', kargs.get('output_file', 'condition_drug.%s' % doctype))
    fpath = os.path.join(basedir, ofile)
    load_seq = kargs.get('load_', False) and os.path.exists(fpath)
    save_seq = kargs.get('save_', False) # overwrite

    sequences = [[], ]
    if load_seq: 
        print('read> loading sequences from file: %s' % fpath)
        sequences = pickle.load(open(fpath, 'rb'))
    else: 
        sequences = sr.read(**kargs)
    
    # sequences = pickle.load(open(fpath, 'rb')) if load_seq else sr.read(**kargs) # this doesn't work with w_vectorize
    # sequences = sr.read(**kargs)
    print('read> number of sentences: %d' % len(sequences))  
    # [log] Number of sentences: 15755982, unique tokens: 219917  # if considered root codes only
    # [log] Number of sentences: 15755982, unique tokens: 238357  # if considered complete codes

    if save_seq: # this is big! ~1.2G 
        div(message='data> saving sequences (size: %d) to %s' % (len(sequences), fpath), symbol='#')
        pickle.dump(sequences, open(fpath, "wb" ))
    
    return sequences
def getSequences(**kargs): 
    return read(**kargs)

def v_mini_doc2vec():
    import seqAnalyzer as sa
    # from gensim.models import Word2Vec, Doc2Vec

    read_mode = 'doc'
    n_subset = 100
    doctype = 'd2v'
    descriptor = 'test'
    doc_basename = 'mini'
    basedir = sys_config.read('DataExpRoot')

    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    documents = seqx = sa.read(load_=True, simplify_code=False, mode=read_mode, verify_=False, seq_ptype=seq_ptype)
    assert seqx is not None and len(seqx) > 0
    n_doc = len(seqx)
    print('input> got %d documents.' % n_doc)

    # labeling 
    df_ldoc = labelDoc(documents, load_=True, seqr='full', sortby='freq', seq_ptype=seq_ptype) # sortby: order labels by frequencies
    labels = list(df_ldoc['label'].values)
    # labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
    print('t_doc2vec1> labels (dtype=%s): %s' % (type(labels), labels[:10]))

    lseqx = makeD2VLabels(sequences=seqx, labels=labels)
    
    lseqx_subset = random.sample(lseqx, n_subset)
    model = Doc2Vec(dm=1, size=20, window=8, min_count=5, workers=8)

    model.build_vocab(lseqx_subset)
    model.train(lseqx_subset)

    # for epoch in range(10):
    #     model.train(lseqx_subset)
    #     model.alpha -= 0.002  # decrease the learning rate
    #     model.min_alpha = model.alpha  # fix the learning rate, no decay    

    # Finally, we save the model
    
    ofile = '%s_%s.%s' % (doc_basename, descriptor, doctype)
    fpath = os.path.join(basedir, ofile)

    print('output> saving (test) doc2vec models (on %d docs) to %s' % (len(lseqx_subset), fpath))
    model.save(fpath)

    n_doc2 = model.docvecs.count
    print('verify> number of docs (from model): %d' % n_doc2)

    lseqx = random.sample(lseqx_subset, 10)
    for i, lseq in enumerate(lseqx): 
        tag = lseq.tags[0]   # [log] [0] label: V22.1_V74.5_V65.44 => vec (dim: (20,)):
        vec = model.docvecs[tag]  # model.docvecs[doc.tags[0]]
        print('[%d] label: %s => vec (dim: %s):\n%s\n' % (i, tag, str(vec.shape), vec)) 
        sim_docs = model.docvecs.most_similar(tag, topn=5)
        print('doc: %s ~ :\n%s\n' % (tag, sim_docs))

    return

def t_lstm_many2one():
    def build(n_timesteps, n_features, n_units=10, n_classes=5):  
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])

        # estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=10, verbose=0)
        # return estimator
        return model
    def create_model(n_timesteps=3, n_features=5, n_units=10, n_classes=3): 
        # def
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])
        return model
    def baseline_model(n_classes=5):
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    import numpy
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.wrappers.scikit_learn import KerasClassifier  ###
    from keras.utils import np_utils

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV

    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline

    # 1, 2, 3 => 1 
    # 2, 1, 3 = > 2
    # 3, 2, 1 => 3 
    X = visits = np.array([ [[1, 1, 2, 1, 1], [2, 2, 1, 3, 2], [3, 1, 2, 3, 3]],
                          [[2, 2, 2, 2, 1], [1, 1, 3, 2, 1], [3, 3, 3, 2, 3]], 
                          [[3, 2, 1, 3, 3], [2, 2, 2, 3, 1], [1, 2, 1, 1, 3]], 
                          [[3, 2, 3, 3, 3], [2, 2, 2, 3, 2], [1, 2, 1, 1, 1]], 
                          [[2, 2, 1, 2, 1], [1, 1, 1, 2, 1], [3, 2, 3, 3, 3]], 
                          [[1, 2, 1, 1, 1], [2, 2, 2, 3, 2], [3, 2, 1, 3, 3]], 
                          [[1, 2, 1, 2, 1], [2, 2, 3, 1, 2], [3, 2, 1, 3, 3]], 
                          ], dtype='float32')
    # 3 t-step, 5 features
    # y = np.array(['a', 'b','c', ])
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], ])
    yw = np.array([1, 2, 3, 3, 2, 1, 1, ])

    # ytest = [1, 2, 3, 3, 2]
    Xtest = np.array([ [[1, 1, 3, 1, 2], [2, 2, 2, 3, 2], [3, 2, 2, 3, 3]],
                          [[2, 2, 2, 1, 1], [1, 1, 3, 2, 1], [3, 1, 3, 2, 3]], 
                          [[3, 3, 1, 3, 3], [2, 2, 2, 3, 2], [1, 2, 1, 1, 1]], 
                          [[3, 3, 2, 3, 3], [2, 3, 2, 2, 2], [1, 1, 1, 1, 1]], 
                          [[2, 1, 2, 2, 1], [1, 1, 2, 1, 1], [3, 3, 3, 2, 3]],
                        ], dtype='float32')    

    nt, nf = X.shape[1], X.shape[2]
    print('> n_timesteps: %d, n_features: %d' % (nt, nf))
    
    tWrapper = False
    # a. define model & train the model directly 
    if not tWrapper: 
        model = build(n_timesteps=nt, n_features=nf, n_units=15, n_classes=3)
        model.fit(X, y, epochs=100, batch_size=10)
 
        # make predictions
        ypred = model.predict(Xtest)
        yl = [np.argmax(y)+1 for y in ypred]

        print('> predictions:\n%s\n ~ \n%s\n' % (ypred, yl))
    else: 
        # b. sklearn wrapper 
        model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=10, verbose=2)
        model.fit(X, yw)

        # make predictions
        ypred = model.predict(Xtest)
        # yl = [np.argmax(y)+1 for y in ypred]

        print('> wrapper predictions:\n%s\n' % ypred)
                 
        # evaluate using 10-fold cross validation
        seed = 53
        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        results = cross_val_score(model, X, yw, cv=kfold)
        print(results.mean())
    
    return 

def t_embedding_lstm(): 
    def lstm_with_dropout(v_size, ndim, max_doc_length): 
        # embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(v_size, ndim, input_length=max_doc_length))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model
        
    # LSTM with Dropout for sequence classification in the IMDB dataset
    import numpy
    from keras.datasets import imdb  
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    
    # fix random seed for reproducibility
    np.random.seed(7)

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # create the model
    embedding_vecor_length = 32
    model = lstm_with_dropout(v_size=top_words, ndim=embedding_vecor_length, max_doc_length=max_review_length)
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    return

def test(**kargs): 

    ### example LSTM network 
    # t_embedding_lstm()

    ### LSTM on visit sequence input format 
    t_lstm_many2one()

    return

if __name__ == "__main__": 
    test()