import numpy as np 
import seqparams

import seqReader as sr

from operator import itemgetter
from collections import OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer

import random
from batchpheno import sampling

####################################################################################################
#
#  Features 
#  ---------
#  1. compute document vectors from word vectors
# 
#  Note
#  ----
#  This module is subsumed by vector (except for Tf-Idf weighting)
#
#  Reference 
#  ---------
#  1. https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors
# 
# 
#
#
####################################################################################################

def word2index(model):
    """
    Extract word2index mapping from Word2Vec model.
    """
    word2index = OrderedDict((v, k) for k, v in sorted(model.index2word, key=itemgetter(1)))
    return word2index

def makeFeatureVec(words, model, n_features=None):
    """
    Function to average all of the word vectors in a given
    paragraph. 

    """
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)

    if n_features is None: 
        # use modeled word to infer feature dimension 
        for word in index2word_set: 
            fv = model[word]
            n_features = fv.shape[0]
            break
        # print('verify> inferred n_features=%d' % n_features)

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((n_features,), dtype="float32")
    #
    nwords = 0 
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(docs, model, n_features):
    """
    Given a set of temporal docs (each one being a list of medical codes), calculate 
    the average feature vector for each one and return a 2D numpy array 
    """
    docFeatureVecs = np.zeros((len(docs), n_features), dtype="float32")
    # 
    # Loop through the (patient) docs

    n_doc = len(docs)
    for i, doc in enumerate(docs):
       if i % 2000 == 0:
           print "+ computing doc #%d of %d via averaging" % (i, n_doc)
       docFeatureVecs[i] = makeFeatureVec(doc, model, n_features)
    return docFeatureVecs

def weightedAvgTfidf(doc, model, weights, n_features=None): 
    """
    Given a set of vectors (from a sentence), and their weights (e.g. td-idf scores), 
    find their weighted average

    Input
    -----
    * weights: a dictionary mapping from feature/token to tfidf score 

    """
    def get_score(tok):
        try: 
            return weights[tok]
        except: 
            pass 
        return 0.0

    index2word_set = set(model.index2word)

    # if n_features is None: 
    #     # use modeled word to infer feature dimension 
    #     for word in index2word_set: 
    #         fv = model[word]
    #         n_features = fv.shape[0]
    #         break
    # featureVec = np.zeros((n_features,), dtype="float32")

    n_tokens = 0 
    wx = []  # weight vector
    vecs = [] # word vector
    for token in doc: 
        if token in index2word_set:
            wx.append(get_score(token))
            vecs.append(model[token])  # np.add(featureVec,model[word])
            n_tokens = n_tokens + 1
    
    return np.average(np.asarray(vecs), weights=wx, axis=0)

def toStrings(docs, sep=' '): 
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

def getPVFeatureVecs(labeled_docs, model=None, n_features=None, **kargs):
    """
    Compute sentence embeddings. 


    Input 
    ----- 
    labeled_doc: labeled sequences (i.e. makeD2VLabels() was invoked)

    Related 
    -------
    seqAnalyzer.vectorize (word2vec)
    seqCluster.vectorize2(doc2vec)

    Memo
    ----
    1. Example settings 

        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),

    """
    def show_params(): 
        msg = "Parameter Setting (d2v):\n"
        msg += '1.  number of features: %d\n' % n_features
        msg += '2.  window length: %d > total: 2 * window: %d\n' % (window, 2*window)
        msg += '3.  min count: %d\n' % min_count
        msg += '4.  PV method: %s\n' % "PV-DM" if dm == 1 else "PV-DBOW"
        msg += '4a. concat? %s\n' % str(dm_concat == 1)
        div(message=msg, symbol='%')
        return 
    from gensim.models import Doc2Vec
    import os, seqparams

    # GNFeatures = seqparams.W2V.n_features  # 100
    GWindow = seqparams.W2V.window  # 7 
    GNWorkers = seqparams.W2V.n_workers
    GMinCount = seqparams.W2V.min_count

    # simple_models = [
    #     # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    #     Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=W2V.n_cores),
    #     # PV-DBOW 
    #     Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=W2V.n_cores),
    #     # PV-DM w/average
    #     Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=W2V.n_cores),
    # ]

    # [params]
    cohort_name = kargs.get('cohort', kargs.get('cohort_name', None)) # 'diabetes', 'PTSD'

    # [params]
    seq_compo = composition = kargs.get('seq_compo', kargs.get('composition', 'condition_drug'))
    doc2vec_method = 'PVDM'  # distributed memory
    n_features = kargs.get('n_features', n_features)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)
    dm, dm_concat = 1, 1 

    n_cores = multiprocessing.cpu_count()
    print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', GNWorkers)

    # [params]
    doctype = 'd2v' 

    # [note] sequence pattern type only affects model file naming 
    seq_ptype = kargs.get('seq_ptype', 'regular')  # values: regular, random, diag, med, lab ... default: regular
    doc_basename = seq_compo if seq_ptype.startswith('reg') else '%s-%s' % (seq_compo, seq_ptype)
    # if not seq_ptype.startswith('reg'):
    if cohort_name is not None: 
        doc_basename = '%s-%s' % (doc_basename, cohort_name)

    descriptor = kargs.get('meta', doc_basename)  # algorithm type: PV-DBOW, DM
    basedir = sys_config.read('DataExpRoot')
    load_model = kargs.get('load_model', True)
    ofile = kargs.get('output_file', '%s-%s_f%dw%d.%s' % (descriptor, doc2vec_method, n_features, window, doctype))
    fpath = os.path.join(basedir, ofile)

    print('io> reading d2v model file from %s' % fpath)
    show_params()

    # Mikolov pointed out that to reach a better result, you may either want to shuffle the 
    # input sentences or to decrease the learning rate alpha. We use the latter one as pointed
    # out from the blog provided by http://rare-technologies.com/doc2vec-tutorial/

    # negative: if > 0 negative sampling will be used, the int for negative specifies how many "noise words"
    #           should be drawn (usually between 5-20).

    if model is None: 
        load_model = load_model and os.path.exists(fpath)
        # compute_model = not load_model
        if load_model:
            print('d2v> loading pre-computed model from %s' % fpath)

            try: 
                model = Doc2Vec.load(fpath)  # may also need .npy file
            except Exception, e: 
                print('error> failed to load model: %s' % e)
                load_model = False

        if not load_model: 
            # dm, dm_concat = 1, 1 
            print('d2v> computing d2v model (dm: %s, dm_concat: %d, n_features: %d, window: %d, min_count: %d, n_workers: %d)' % \
                    (dm, dm_concat, n_features, window, min_count, n_workers))
            model = doc2vec.Doc2Vec(dm=1, dm_concat=1, size=n_features, window=window, negative=5, hs=0, min_count=min_count, workers=n_workers, 
                                alpha=0.025, min_alpha=0.025)  # use fixed learning rate
            model.build_vocab(labeled_docs) # [error] must sort before initializing vectors/weights

            for epoch in range(10):
                model.train(labeled_docs)
                model.alpha -= 0.002  # decrease the learning rate
                model.min_alpha = model.alpha  # fix the learning rate, no decay    

            # [output] Finally, we save the model
            print('output> saving doc2vec models (on %d docs) to %s' % (len(labeled_docs), fpath))
            model.save(fpath) 

    # [test]
    tag_sample = [ld.tags[0] for ld in labeled_docs][:10]
    print('verify> example tags (note: each tag can be a string or a list):\n%s\n' % tag_sample)
    # [log] ['V70.0_401.9_199.1', '746.86_426.0_V72.19', '251.2_365.44_369.00', '362.01_599.0_250.51' ... ] 

    print('Given d2v model, now create matrix of document vectors ...')
    # makeD2VLabels(sequences=seqx, labels=labels, cohort=cohort_name)

    lseqx = labeled_docs

    # [params] inferred 
    n_doc = len(lseqx)
    n_features_prime = model.docvecs[lseqx[0].tags[0]].shape[0]
    assert n_features_prime == n_features
    print('verify> n_doc: %d, fdim: %d' % (n_doc, n_features_prime))

    # [test]
    # lseqx_subset = random.sample(lseqx, 10)
    # for i, lseq in enumerate(lseqx_subset): # foreach labeld sequence
    #     tag = lseq.tags[0]
    #     vec = model.docvecs[tag]  # model.docvecs[doc.tags[0]]
    #     print('[%d] label: %s => vec (dim: %s)' % (i, tag, str(vec.shape)))

    # dmatrix = [model.docvecs[lseq.tags[0]] for lseq in lseqx]
    dmatrix = np.zeros( (n_doc, n_features), dtype="float32" )  # alloc mem to speed up
    for j, lseq in enumerate(lseqx): 
        dmatrix[i] = model.docvecs[lseq.tags[0]]  

    return dmatrix


def getFeatureVectors(docs, model, n_features, **kargs):
    d2v_method = kargs.get('d2v_method', 'tfidfavg')

    print('d2v> using %s method to create doc vectors' % d2v_method)

    if d2v_method.startswith('tf'):
        return getTfidfAvgFeatureVecs(docs, model=model, n_features=n_features, **kargs)
    elif d2v_method.startswith(('ave', 'mean')): # 'average'
        return getAvgFeatureVecs(docs, model=model, n_features=n_features, **kargs)

    # PV
    print('d2v> using PVDM method with input: d2v-labeled doc')

    # need to label the sequences first
    # df_ldoc = labelDocsByFreq(docs, load_=True, seqr='full', sortby='freq')
    print('info> assuming input docs have been labeled (e.g. by labelDocsByFreq(...))')
    return getPVFeatureVecs(docs, **kargs)

def getTfidfAvgFeatureVecs(docs, model, n_features, **kargs): 
    """

    Note
    ----
    1. input 'docs' must be converted to a list of strings for TfidfVectorizer to work

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
    def tokenize(): 
        tokens = []
        for doc in docs:  
            tokens.extend(doc)
        return list(set(tokens)) 
    def prefix_codes(): 
        # for doc in docs: 
        #     for i, w in enumerate(doc):
        #         
        pass 

    docFeatureVecs = np.zeros((len(docs), n_features), dtype="float32")

    code_prefix = 'c'
    
    # [params] TfidfVectorizer()
    min_df = kargs.get('min_df', 1)
    max_df = kargs.get('max_df', 1.0)
    max_features = kargs.get('max_features', None)

    docx = toStrings(docs)
    tokens = tokenize()
    n_tokens = len(tokens) # number of unique tokens

    # [note] need to provide tokens because otherwise diag codes like 250.01 may not be considered as tokens 
    vectorizer = TfidfVectorizer(min_df=1, vocabulary=tokens)   

    X = vectorizer.fit_transform(docx) # X is in sparse matrix format
    idf = vectorizer.idf_
    # info
    stop_symbols = []
    try: 
        stop_symbols = vectorizer.stop_words_  # This is only available if no vocabulary was given.
    except: 
        print('info> vocabulary was given.')

    adict = dict(zip(vectorizer.get_feature_names(), idf)) 

    # [log] n_features: 180546, n_stop_words: 0
    print('verify> n_features: %d (n_tokens: %d), n_stop_words: %d' % (len(adict), n_tokens, len(stop_symbols))) 
    print('verify> example maps:\n%s\n' % sampling.sample_dict(adict, n_sample=10))
    
    # [test]
    if kargs.get('test_', False): t_tfidf()

    n_doc = len(docs)
    for i, doc in enumerate(docs): 
       if i % 5000 == 0:
           print "+ computing doc #%d of %d via tfidf-averaging" % (i, n_doc)
       docFeatureVecs[i] = weightedAvgTfidf(doc, model, weights=adict, n_features=n_features) # make feature vectors by (tfidf-)weighted average     

    return docFeatureVecs

def create_bag_of_centroids( wordlist, word_centroid_map ):
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


def test(**kargs): 
    pass 

if __name__ == "__main__": 
    test()