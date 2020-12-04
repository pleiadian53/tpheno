# encoding: utf-8

import seqMaker2 as smk2  # this interfaces DB and create time series documents
import seqReader as sr    # this intefaces time series documents themselves

# [todo] __init__
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, gc, sys, random 
from os import getenv 
import time, re
import timeit
try:
    import cPickle as pickle
except:
    import pickle

# local modules 
from batchpheno import icd9utils, sampling, qrymed2
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed
import seqparams  # all algorithm variables and settings
import analyzer
import vector  # w2v, d2v 

# word2vec modules
import gensim
from gensim.models import Word2Vec, Doc2Vec

# logging 
import logging

import multiprocessing

###############################################################################################
#
#  This module analyzes the data set under /data (source, raw data) and /data-exp (curated)
#  and performs exploratory data analysis; this module is intended for testing and serving as 
#  a supportive module.
# 
#  Dependency 
#  ---------- 
#    seqReader -> {analyzer, vector} -> seqAnalyzer
#    
#  Usage Note
#  ----------
#  The module vector subsumes analyzer in creating feature vectors, word & document vectors
#  
#  
###############################################################################################

##### Set module variables ##### 
GNFeatures = seqparams.W2V.n_features
GWindow = seqparams.W2V.window
GNWorkers = seqparams.W2V.n_workers
GMinCount = seqparams.W2V.min_count

# Within the collection of source valus (from odhasi tables), separate literal values from coded values
def funnel(data_file=None):
	pass

def subset(docs=None, **kargs): # [todo]
    return docs

def select(docs , idx=None, **kargs): 
    # if docs is None: 
    #     docs = read(**kargs)
    assert docs is not None and len(docs) > 0
    if idx is None: 
        return docs
    return np.asarray(docs)[idx].tolist()

def readDocToCSV(**kargs):
    """
    Save coding sequences (from sources) to .csv format

    This routine has another version: seqReader.readDocToCSV

    Use
    ---
    Use this to save coding sequence data to a csv file while preserving 
    meta data such as timestamps, labels in csv columns

    e.g. A CSV file of a PTSD-specific coding sequences (i.e. cohort='PTSD') could 
         contain the following header: 

         sequence | timestamp | label 

         where rows in 'sequence' are the coding sequences of patients
               timestamp 
               label: labels assigned to the coding sequences 

    Input 
    ----- 
    cohort, inputdir, ifiles => readDoc 
    labels: 

    #1 cohort 
    #2 ifiles

    Output 
    ------
    dataframe containing following columns 
        
        ['sequence', 'timestamp', 'label', ]

        where 'timestamp' and 'lable' are only included when available

    Params
    ------
    doctype 
       'doc'
       'labeled'
       'timed'
       'visit'

    Versions
    --------
    seqReader.readDocToCSV(): uses readTimedDocPerPatient() to read .dat sources

    """
    def seq_to_str(seqx, sep=','): 
        return [sep.join(str(s) for s in seq) for seq in seqx] 

    def str_to_seq(df, col='sequence', sep=','):
        seqx = []
        for seqstr in df[col].values: 
            s = seqstr.split(sep)
            seqx.append(s)
        return seqx

    # first, read the docs from source files (.dat)
    # [note] seqReader.readDocToCSV() uses readTimedDocPerPatient()
    if not kargs.has_key('read_mode'): kargs['read_mode'] = 'timed'  # default. or set include_timestamps to True
    sequences, timestamps = kargs.get('sequences', []), kargs.get('timestamps', [])
    if len(sequences) == 0 or len(timestamps) == 0: 
        sequences, timestamps = readDoc(**kargs) # use 'read_mode' to determine the read routine 

    doctype = kargs.get('doctype', 'doc')  # 'doc': one-doc-per-patient format | 'visit':  one-doc-per-visit format; not in use
    if len(timestamps) > 0: 
        doctype = 'timed'
        assert len(sequences) == len(timestamps)

    labels = kargs.get('labels', []) # labels are external because they are included in regular sequencing documents (.dat files)
    if len(labels) > 0: 
        doctype = 'labeled'

        # the nubmer of labels is not always identicial to number of documents (because a subset of patients may not have data)
        assert len(labels) == len(sequences), "n_labels=%d <> n_docs=%d" % (len(labels), len(sequences))
        # if len(labels) <> len(sequences): 
        #     print("warning> n_labels=%d <> n_docs=%d | use .id file to verify consistency ..." % (len(labels), len(sequences)))
        
    print('info> n_doc: %d, doctype: %s, n_labels: %d' % (len(sequences), doctype, len(labels)))

    # [params] IO 
    ret = {}  # output
    basedir = kargs.get('outputdir', sys_config.read('DataExpRoot'))  # sys_config.read('DataIn') # document in data-in shows one-visit per line
    fname = seqparams.TDoc.getName(cohort=kargs['cohort'], doctype=doctype, doc_basename=None, ext='csv') # None-s are default values 
    fpath = os.path.join(basedir, fname)

    df = DataFrame() # dummy 
    if len(sequences) > 0: # this is big! ~1.2G 
        
        # pickle.dump(sequences, open(fpath, "wb" ))
        if doctype in ['timed', 'labeled',]:
            header = ['sequence', 'timestamp', ]   # ['sequence', 'timestamp', 'label', ]
            adict = {} # {h: [] for h in header}
            adict['sequence'] = seq_to_str(sequences)
            adict['timestamp'] = seq_to_str(timestamps)
            
            if doctype == 'labeled':  # qualified earlier
                header = header + ['label', ]
                adict['label'] = labels

            df = DataFrame(adict, columns=header)
            df.to_csv(fpath, sep='|', index=False, header=True)
        else: # doctype <- 'doc', 'visit'
            header = ['sequence', ]   # seqMaker2 now also can be configued to produce this (but without seq_ptype as part of the file ID)
            # seq_to_str(sequences)
            df = DataFrame(seq_to_str(sequences), columns=header)
            df.to_csv(fpath, sep='|', index=False, header=True)

        div(message='io> saved sequences (size: %d, doctype: %s, dim: %s) to:\n%s\n' % \
            (len(sequences), doctype, str(df.shape), fpath), symbol='#')

    return df

def loadDocFromCSV(**karsg):
    """
    Load coding sequences from .csv sources 
    This is just a wrapper/alias of readFromCSV()
    """
    # import seqReader as sr
    return sr.loadDocFromCSV(**kargs) 
def readDocFromCSV(**kargs): 
    """
    Read coding sequences from .csv files 

    Input
    -----
    cohort: name of the cohort (e.g. PTSD) ... must be provided; no default
    ifiles: ifiles: sources of coding sequences 
            if 
    (o) basedir: directory from which sources are stored (used when ifiles do not include 'prefixes')
    
    Output
    ------
    A dictionary with the following key-value pairs: 

    'sequence': coding sequence (list of lists of tokens)
    'timestamp': time stamps 
    'label': 

    
    """
    # import seqReader as sr
    return sr.readDocFromCSV(**kargs)  

def readDoc(**kargs):
    """
    Input
    -----
    cohort: name of the cohort (as an identifier for cohort-specific coding sequences); e.g. 'PTSD'
    inputdir: prefix of the path to coding sequence source files 
             used when 'ifiles' are not provided and when 'ifiles' do not contain prefixes
    ifile: source file containing coding sequences
    ifiles: multiple input source files
             if provided, cohort will be ignored

    simply_code: always set to False now ... 09.17 

       250.00 => 250
       749.13 => 749
       use seqAlgo to simplify the code if necessary 

    If sources are not already available (i.e. 'ifiles' not given), then 

    Log
    ---
    1. number of documents (patients): 432,000
       + number of sentences (visits): 15,755,982
       + number of (unique) tokens: 200,310  
       ... 02.20.17

    Call Chain
    -----------
    read() -> sa.read() | sa.read_doc()

    Memo
    ----
    1. Input 
       .dat: de-identified coding sequences (as opposed to sensitive .csv files with person_ids and other meta data)

    1. Output 
       .csv: seqmaker.seqMaker2 also can be configured to produce .csv file 


    """    
    def shift_ext(ext='pkl'):
        ifiles_prime = ifiles[:]  # don't want to replace ifiles inplace (may need to passed to seqReader later)
        for i, f in enumerate(ifiles_prime): 
            base_, ext_ = os.path.splitext(f) 
            ext_ = ext_[1:]  # don't include '.'
            if ext_ != ext: 
                ifiles_prime[i] = base_ + '.' + ext  

        # 'ifiles' refers to the same ifiles in the outer function 
        return ifiles_prime
    def seq_to_str(seqx, sep=','): 
        return [sep.join(str(s) for s in seq) for seq in seqx] 
    def str_to_seq(df, col='sequence', sep=','):
        seqx = []
        for seqstr in df[col].values: 
            s = seqstr.split(sep)
            seqx.append(s)
        return seqx
    def normalize_paths(ifiles):
        if not ifiles: return [] # noop 
        for i, f in enumerate(ifiles): 
            # contain rootdir? 
            rootdir, fname0 = os.path.dirname(f), os.path.basename(f) 
            if not rootdir: rootdir = basedir
            fp = os.path.join(rootdir, fname0)
            assert os.path.exists(fp), "Invalid input source file: %s" % fp
            ifiles[i] = fp
        return ifiles 

    # import seqReader as sr

    # output mode: 1) sequences across patients 2) patient-specific document (i.e. one patient, one sequence combining all visits)
    read_mode = mode = kargs.get('read_mode', 'doc') # or 'seq'
    if kargs.get('include_timestamps', True): read_mode = mode = 'timed'  # made ompatible with seqReader.read*
    print('info> read mode: %s' % read_mode)

    # sequence pattern type: regular (ordering perserved) (use 'regular')
    #                        shuffled (removing ordering) (use 'random')
    #                        diagnostic codes only (use 'diag'), medications only (use 'med'), labs only, etc. 
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
    print('verify> requested sequence pattern type: %s' % seq_ptype)
    # doc_basename = 'condition_drug'

    # file name of the input sequences is fixed: <doc_basename>_seq-<cohort>.dat  e.g. condition_drug_seq-PTSD.dat
    # if not seq_ptype.startswith('reg'): 
    #     doc_basename = '%s-%s' % (doc_basename, seq_ptype)

    ### select read routine 
    # [note] seqReader.read() or .read_doc() does not involve disk I/O 
    read_routine = None
    if read_mode.startswith('v'): 
        print('info> making per-visit documents ... ')
        doctype = 'visit'  # one sequence per visit 
        read_routine = sr.readDocPerVisit
    elif read_mode.startswith('d'):  # one patient, one doc 
        print('info> making per-patient documents (combining all visit-specific sub-sequences) ... ')
        doctype = 'doc'  # one patient per doc (which then contains all visits)
        read_routine = sr.readDocPerPatient
    elif read_mode.startswith('t'): # timed sequences 
        doctype = 'timed'  # ditto but include timestamps (of codes) as well
        read_routine = sr.readTimedDocPerPatient  # returns 2-tuple: sequences and timestamps
    elif read_mode.startswith('l'): # labeled 
        # read sequences from .csv files
        doctype = 'labeled'  # [todo]
        read_routine = sr.readDocFromCSV 
        # this allows for reading in multiple attributes
        # header = ['sequence', 'timestamp', 'label', ] 
        
    # [params] cohort 
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')

    # [I/O] basedir (from which the input comes from)
    #       used when 'ifiles' are not provided and when 'ifiles' do not contain prefixes 
    kargs['inputdir'] = basedir = kargs.get('inputdir', sys_config.read('DataExpRoot'))  # sys_config.read('DataIn') # document in data-in shows one-visit per line
    print('sa.read> basedir: %s' % basedir)

    # [input] temporal doc file(s)
    #         diabetes: 'condition_drug_seq.dat' |  PTSD: 'condition_drug_seq-PTSD.dat'
    #         .dat is the source from seqmaker.seqMaker2, whereas 
    #         .pkl or .csv are processed input files  => readFromCSV()
    # ifile_default = seqparams.TDoc.getName()
    # '%s_seq-%s.dat' % (doc_basename, cohort_name) if cohort_name is not None else '%s_seq.dat' % doc_basename
    # if doctype == 'timed': 
    #     ifile_default = '%s_timed_seq-%s.dat' % (doc_basename, cohort_name) if cohort_name is not None else '%s_timed_seq.dat' % doc_basename
    kargs['ifiles'] = ifiles = normalize_paths(kargs.get('ifiles', [])) # ensure each input file has a full path

    # [params] but the true identity for input sequences has either extension in {.pkl, .csv} 
    #          => readFromCSV 
    # seq_ext = 'csv' # extension for the file of the processed input sequences 
    # ifiles_prime = shift_ext(ext=seq_ext)  # ensure that we don't end up reading .dat files (sources) when load_ is set to True

    sequences = []
    timestamps = [] 
    # [input] single source vs multiple sources
    #         attempt to load directly from .csv files 
    #         => readFromCSV()
    seq_loaded = False
    if kargs.get('load_from_csv', False): 
        ret = readFromCSV(**kargs)
        sequences, timestamps = ret['sequences'], ret['timestamps']
        seq_loaded = True

    if not seq_loaded or not sequences: 
        # [note] 
        # 1. use 'ifiles' to change the set of source document files (default: ['condition_drug_seq.dat', ])
        # 2. use 'inputdir' to change basedir (from which ifiles are read)
        # 3. file name depends on 'cohort_name'

        # [assumption] cohort-specific coding sequence document was generated (.dat) (via seqMaker2.py)
        div(message='Creating new temporal document (params> read_mode: %s, ptype: %s, cohort: %s)' % \
            (read_mode, seq_ptype, cohort_name))
        if not kargs.has_key('inputdir'): kargs['inputdir'] = basedir
        if doctype == 'timed': 
            sequences, timestamps = read_routine(**kargs) # read_routine is a function pointer
            assert len(sequences) == len(timestamps)
        elif doctype == 'labeled': 
            ret = read_routine(**kargs)
            sequences, timestamps, labels = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])
            assert len(sequences) == len(timestamps)
            if len(labels) > 0: assert len(sequences) == len(labels)
        else: # 'doc', 'visit'
            sequences = read_routine(**kargs) # read_routine is a function pointer
    
    # sequences = pickle.load(open(fpath, 'rb')) if load_seq else sr.read(**kargs) # this doesn't work with w_vectorize
    # sequences = sr.read(**kargs)
    print('readDoc> number of sentences (doctype=%s): %d' % (doctype, len(sequences)))  # len(seq_to_str(sequences)

    # [log] Number of sentences: 15755982, unique tokens: 219917  # if considered root codes only
    # [log] Number of sentences: 15755982, unique tokens: 238357  # if considered complete codes
    
    return (sequences, timestamps)
def getSequences(**kargs): 
    return read(**kargs)

def read(**kargs): 
    sequences, _ = readDoc(**kargs)
    return sequences

def read2(**kargs): # returns both documents and their timestamps
    return readDoc(**kargs)
def readTimedDoc(**kargs): 
    return readDoc(**kargs)

def filterSequence(seqx): 
	pass

def getFreqCodes(**kargs):
    """
    Get most common medical codes in the document set. 

    Related
    -------
    tokenize() 
    getTestcodes() 
    """ 
    import collections
    # print('Warning: unknown cohort: %s' % cohort_name)
        
    topn = kargs.get('topn', 100)
    assert kargs.has_key('cohort'), "Missing cohort info."
    kargs['seq_ptype'] = 'regular'
    codes = tokenize(**kargs)
    
    counter = collections.Counter(codes)  
    freqCodes = counter.most_common(topn)  # topn are mostly unigrams
    print('info> frequent codes: %s' % freqCodes[:10])
    codeset = [c for c, cnt in freqCodes]

    return codeset

def tokenize(**kargs): 
    """
    Similar to read() but tokenize the document into sequence of medical codes and symbols 
    without per-visit sentence structure. 

    Definition 
    ----------
    * sequence vs sentence: 
      Each sequence is effectively a sentence of consecutive medical codes and symbols ordered by timestamps

    Output
    ------
    A list of all medical codes/symbols. 

    """
    doctype = 'token'
    seq_compo = kargs.get('composition', 'condition_drug') # what does the sequence consist of? 
    cohort_name = kargs['cohort'] 

    # [params] sequence properties
    seq_ptype = seqparams.normalize_ctype(**kargs) # values: regular, random, diag, med, lab ... default: regular
    
    doc_basename = seq_compo
    # if not seq_ptype.startswith('reg'): 
    #     doc_basename = '%s-%s' % (doc_basename, seq_ptype)

    # set lod to true
    read_mode = kargs.get('read_mode', 'doc')  # documents/doc (one patient one sequence) or sequences/seq
    
    # [input] temporal sequences and word2vec model
    basedir = sys_config.read('DataExpRoot') 
    default_ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo
    ifile = kargs.get('ifile', default_ifile) 
    ifiles = kargs.get('ifiles', None) # [todo]

    simplify_code = kargs.get('simplify_code', False)

    default_ofile = '%s_simple.%s' % (doc_basename, doctype) if simplify_code else '%s.%s' % (doc_basename, doctype)
    ofile = kargs.get('output_file', default_ofile)

    fpath = os.path.join(basedir, ofile)
    load_seq = True

    tUniqOnly = False

    # load_tok = kargs.get('load_', True) and os.path.exists(fpath) and os.path.getsize(fpath) > 0
    # save_tok = kargs.get('save_', False) # overwrite

    # if load_tok: 
    # 	print('load> tokens from %s' % fpath)
    # 	return pickle.load(open(fpath, 'rb')) 

    sequences = read(load_=load_seq, simplify_code=simplify_code, mode=read_mode, verify_=False, 
                        seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name)
    
    # Alternative, one can just invoke seqReader's method directly
    # sequences = sr.readDoc(**kargs)

    # each sequence is effectively a sentence of consecutive medical codes and symbols ordered by timestamps

    # too space consuming to worth it
    # if save_tok: 
    #     div(message='data> saving sequences (size: %d) to %s' % (len(sequences), fpath), symbol='#')  
    #     pickle.dump(tokens, open(fpath, "wb" ))
    
    return docsToTokens(sequences, uniq=tUniqOnly)

def docsToTokens(seqx, uniq=False, sorted_=True, reverse=False): 
    """
    Convert a set of documents to a set of tokens where 
    each docuemnt consists a list of tokens. 

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
    for seq in seqx:  
        tokens.extend(seq) 
    return tokens  

def demo_word2vec(sequences, **kargs): 
    model = vectorize(sequences, **kargs)
    model.build_vocab(sentences)

    # model.most_similar(positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None)
    # n_similarity(ws1, ws2)
    # similar_by_vector(vector, topn=10, restrict_vocab=None)
    # similar_by_word(word, topn=10, restrict_vocab=None)

    # sort_vocab()
    # Sort the vocabulary so the most frequent words have the lowest indexes.

    return 

def classify_codes(tokens=None, save_table=False, **kargs): 
    """
    Separate diagnostic codes, medications and labs. 
    """
    if save_table: return analyze(tokens, **kargs)
    res = {'diag': set(), 'drug': set(), 'lab': set(), 'other': set()}
    if tokens is None: 
    	tokens = tokenize()
    n_tokens = len(tokens)
    assert n_tokens > 0 
    print('classify_codes> number of tokens: %d' % n_tokens)

    for i, token in enumerate(tokens): 
    	if pmed.isICD(token): 
            res['diag'].add(token)
        elif pmed.isMed(token): 

        	res['drug'].add(token)
        # elif pmed.isLab(token): 
        # 	res['lab'].add(token)
        else: 
        	if i < 100: print('   + classified as other: %s' % token)
        	res['other'].add(token)

    return res

def load_noncoded(**kargs): 
    import seqMaker2 as sm 
    return sm.load_noncoded(**kargs)

def load_noncoded_by(name='condition', cohort='diabetes'):
    import seqMaker2 as sm 
    return sm.load_noncoded_by(name=name, cohort=cohort)


def analyze_lab(tokens, **kargs): # also see seqmaker.cohort
    pass
def analyze_drug(tokens, **kargs): # isMedication(x)
    def get_code(x):
        return x.split(delimit)[-1]

    # import qrymed2, icd9utils
    # from pattern import medcode as pmed

    cohort_name = kargs.get('cohort', 'diabetes')
    identifier = kargs.get('identifier', None)
    # [I/O] global output directory
    # output_dir = sys_config.read('DataExpRoot')  # convention: only doc files are saved to DataIn 

    # [I/O] local output directory (module dependent)
    
    basedir = outputdir = kargs.get('outputdir', seqparams.get_basedir(cohort=cohort_name))  # [output] local data directory
    ofile = 'token_lookup-med.csv' if identifier is None else 'token_lookup-%s.csv' % identifier
    fpath = os.path.join(outputdir, ofile)

    # MED:12345 
    delimit = ':'

    header = ['med', 'description']    # 'med' actually consists of at least 3 standards: {MED, NDC, MULTUM}
    adict = {h: [] for h in header}
    
    codeset = set()
    n_valid_names = 0
    for code in tokens: 
        if not pmed.isMedCode(code): continue  # isICDv2 has a better regex (than isICD)
        
        # standard 1: MED 
        medcode = get_code(code) # remove prefix (e.g. MED:12345)
        name = qrymed2.getName2(medcode) # this won't work with MULTUM, NDC (e.g. NDC:00143126401, MULTUM:5324)
        
        if not name in (None, '', ): n_valid_names += 1 
        print('  + med: %s -> desc: %s' % ('?' if medcode in (None, '') else medcode, name))

        adict['med'].append(medcode)
        adict['description'].append(name)

        codeset.add(code)

    print('info> Found %d unique medication codes (%d has descriptions)' % (len(codeset), n_valid_names))

    lookuptb = DataFrame(adict, columns=header)
    print('io> saving medication code lookup table (dim=%s) to:\n%s\n' % (str(lookuptb.shape), fpath))
    lookuptb.to_csv(fpath, sep='|', index=False, header=True)

    return 

def analyze_diag(tokens, **kargs):
    # import qrymed2, icd9utils
    # from pattern import medcode as pmed

    cohort_name = kargs.get('cohort', 'diabetes')
    identifier = kargs.get('identifier', None)
    # [I/O] global output directory
    # output_dir = sys_config.read('DataExpRoot')  # convention: only doc files are saved to DataIn 

    # [I/O] local output directory (module dependent)
    
    basedir = outputdir = kargs.get('outputdir', seqparams.get_basedir(cohort=cohort_name))  # [output] local data directory
    ofile = 'token_lookup-diag.csv' if identifier is None else 'token_lookup-%s.csv' % identifier
    fpath = os.path.join(outputdir, ofile)

    header = ['code', 'med', 'description']
    adict = {h: [] for h in header}
    
    codeset = set()
    n_valid_names = 0
    for code in tokens: 
        if not pmed.isICD(code): continue  # isICD() now use the new regex (isICDv2 has a better regex (than  isICDv0))
        
        medcode = qrymed2.ICD9ToMed(code)
        if medcode is None:  
            name = icd9utils.getName(code)
        else: 
            name = qrymed2.getName2(medcode)
        
        if not name in (None, '', ): n_valid_names += 1 
        print('  + code: %s -> med: %s -> desc: %s' % (code, '?' if medcode in (None, '') else medcode, name))

        adict['code'].append(code)
        adict['med'].append(medcode)
        adict['description'].append(name)

        codeset.add(code)

    print('info> Found %d unique diagnosis codes (%d has descriptions)' % (len(codeset), n_valid_names))

    lookuptb = DataFrame(adict, columns=header)
    print('io> saving diagnosis code lookup table (dim=%s) to:\n%s\n' % (str(lookuptb.shape), fpath))
    lookuptb.to_csv(fpath, sep='|', index=False, header=True)
    
    return
    
def analyze(tokens, **kargs): 
    def is_modeled(c): 
        if w2v_model is None: 
            return True 
        else: 
            try: 
                w2v_model[c] # w2v_model does not have has_key 
            except: 
                return False
        return True
	
    # [params]
    res_header = ['diag', 'drug', 'lab', 'other', 'modeled']
    res = {h:set() for h in res_header}  # [output]

    cohort_name = kargs.get('cohort', 'diabetes')
    identifier = cohort_name 

    output_dir = sys_config.read('DataExpRoot')  # convention: only doc files are saved to DataIn 
    ofile = 'token_lookup.csv' if identifier is None else 'token_lookup-%s.csv' % identifier
    fpath = os.path.join(output_dir, ofile)
    load_data = kargs.get('load_', True) and os.path.exists(fpath)
    sep = '|'
    w2v_model = seqparams.arg(['model', 'w2v_model'], None, **kargs) # kargs.get('model', None)

    lookup = {}  # [output]
    lookuptb = None
    if load_data: # offer to load data because it takes time to build the table
    	lookuptb = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
    	print('data> loaded token lookup table (dim: %s) from:\n%s\n' % (str(lookuptb.shape), fpath))
    else: 
        pass 

    # if tokens is None: tokens = tokenize()
    assert tokens is not None and len(tokens) > 0, "No input tokens!"
        
    tokens = list(set(tokens)) # unique tokens
    n_tokens = len(tokens)
    assert n_tokens > 0 
    print('analyze> number of (unique) tokens: %d' % n_tokens) # umber of (unique) tokens: 245976 >? 238357

    ### analyze tokens 
    # 1. build lookup table 

    # noncoded lookup 
    noncoded_condition = load_noncoded(name='condition') # e.g. condition_noncoded-PTSD.cs
    noncoded_drug = load_noncoded(name='prescription') # e.g. drug_noncoded-PTSD.csv

    bypass_lookup = kargs.get('bypass_lookup', False)
    build_lookup = True if lookuptb is None or lookuptb.empty else False

    lookuptb = None  # [expensive]
    if not bypass_lookup and build_lookup: 
        print('analyze> building lookup table ... ') # takes long

        noncoded = load_noncoded()

        header = ['token', 'description']
        adict = {}
        for h in header: 
            adict[h] = []

        for i, token in enumerate(tokens): 
            val = ''
    	    if pmed.isICD9(token): 
                val = icd9utils.lookup2(token)

            elif pmed.isICD10(token):
            	# http://icd10api.com/
                pass 
            elif pmed.isMedCode(token): # coded (either officially in ontology or internally coded with prefix 'drug')
                med_code = token.split(':')[0]
            	val = qrymed2.getName2(med_code)

            else: 
            	val = noncoded.get(token, '')
            	
            if not val in ('', None, 'n/a', ): 
            	adict['token'].append(token)
                adict['description'].append(val)

        lookuptb = DataFrame(adict, columns=header)
        print('output> saving token lookup table (dim=%s) to:\n%s\n' % (str(lookuptb.shape), fpath))
        lookuptb.to_csv(fpath, sep='|', index=False, header=True)
        
    # convert to dictionary
    # lookup = {}
    if lookuptb is not None and not lookuptb.empty: 
        lookup = dict(zip(lookuptb['token'].values, lookuptb['description'].values))

    n_cond = n_drug = n_other = 0 # overwrite previous
    n_modeled = 0
    for i, token in enumerate(tokens):
        if is_modeled(token): 
            res['modeled'].add(token)
        else: 
            continue

        # isCondition > isICD
    	if pmed.isCondition(token) or noncoded_condition.has_key(token): 
            if n_cond < 10: 
                print('   + classified as condition: %s' % token)
            res['diag'].add(token)
            n_cond += 1 
                       
        elif pmed.isMed(token) or noncoded_drug.has_key(token): 
            if n_drug < 10: 
                print('   + classified as prescription: %s' % token)
            res['drug'].add(token)
            n_drug += 1 
            # elif pmed.isLab(token): 
            # 	res['lab'].add(token)
        else: 
            if n_other < 10: print('   + classified as other: %s' % token)
            res['other'].add(token)
            n_other += 1 

    return (res, lookup)

def vectorize_tf(sequences, **kargs): 
    """
    Vectorize using Tensorflow 
	"""
    pass

def vectorize_keras(sequences, **kargs): 
    """

    Reference
    ---------
    1. https://keras.io/preprocessing/text/
    """

    # keras.preprocessing.text.one_hot(text, n,
    #      filters=base_filter(), lower=True, split=" ")

    return

def vectorize_w2v(sequences, **kargs): 
    return vectorize(sequences, **kargs)
def vectorize_d2v(sequences, **kargs):
    return vectorize2(sequences, **kargs)

def getDocVec(sequences, **kargs):  # model will be stored ~ outputdir
    """
    Params
    ------
    d2v_method
    outputdir: the directory in which d2v model is saved. 

    """
    # import vector
    return vector.getDocVec(sequences, **kargs)  # [output] docuemnt vectors

def getDocVectorModel(sequences, **kargs):
    return vector.getDocVectorModel(sequences, **kargs) # [output] model (d2v)

def vectorize2(sequences, **kargs):
    # return vector.getDocVectorModel(sequences, **kargs) # [output] model (d2v)
    raise NotImplementedError, "Use vector.vectorize2(), or seqCluster.vectorize2() with cohort-specific outputdir"

def getWordVecModel(sequences, **kargs):   
    """

    Related
    -------
    vector.getDocVec
    """
    return vector.vectorize(sequences, **kargs)

def vectorize(sequences, **kargs):  # use vector.getWordVec(); wrapper on top of vector.vectorize
    """
    Given sequences (list of lists of symbols, where symbols refer to medical codes in general), 
    compute their vector representation via word2vec methods. 

    Experiments
    -----------
    1. isolate diagnostic codes from medications (and other noises)
       => given a sequence of (temporally-ordered) d-codes, can we predict: 
          a. a set of medications
          b. a set of temporally ordered medications => treatment pathways 
    2. conversely, given a sequence of medications, predict diagnoses 
       causal link? correlations? bidrectional? 

    3. similarity of codes observed from the data 
       3a. similarity of medications observed from the data 

    4. predict diagnoses given history

    *5. common treatment pathways? medication sequences given diagnostic sequences
        common treatment patterns


    Note
    -----
    1. example temp. doc file: 
         condition_drug_seq.dat
         condition_drug_seq-PTSD.dat

    Memo
    ----
    1. Word2Vec parameters
        sg: defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.

        size: the dimensionality of the feature vectors.
        window: the maximum distance between the current and predicted word within a sentence.
        hs: if 1, hierarchical softmax will be used for model training.
            If set to 0 (default), and 'negative' is non-zero, negative sampling will be used.

        negative: if > 0, negative sampling will be used, the int for negative
                  specifies how many "noise words" should be drawn (usually between 5-20).
                  Default is 5. If set to 0, no negative samping is used.

        cbow_mean: if 0, use the sum of the context word vectors. If 1 (default), use the mean.
                   Only applies when cbow is used.

    2. Doc2Vec 

       dm_mean = if 0 (default), use the sum of the context word vectors. 
                 If 1, use the mean. Only applies when dm is used in non-concatenative mode.

       dm_concat = if 1, use concatenation of context vectors rather than sum/average; default is 0 (off). 
                   Note concatenation results in a much-larger model, as the input is no longer the size of one 
                   (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words in the context strung together.

    """
    # [params]
    # seq_compo = composition = kargs.get('composition', 'condition_drug')
    cohort_name = kargs.get('cohort', 'diabetes') # 'diabetes', 'PTSD'

    # w2v_method = seqparams.normalize_w2v_method(kargs.get('w2v_method', 'sg')) # kargs.get('w2v_method', 'sg')   # skip-gram (sg) or CBOW (cbow)
    # d2v_method = kargs.get('d2v_method', None)  # this may subsume w2v_method setting

    # sg = 1 if str(w2v_method).lower().startswith(('s', 'skip', '1')) else 0 
    # for the moment when sg = 0, it could be cbow or LSTM type of word embedding methods

    # logdir = kargs.get('log_dir', sys_config.read('LogDir'))
    # logging.basicConfig(filename=os.path.join(logdir, 'makeWordVec.log'), level=logging.DEBUG)

    # sys_config.read('DataExpRoot')
    # use cohort-dependent output directory, tentatively under data/<cohort>
    outputdir = basedir = kargs.get('outputdir', seqparams.getCohortDir(cohort=cohort_name)) # sys_config.read('DataIn')

    # sequence pattern type: regular (ordering perserved) (use 'regular')
    #                        shuffled (removing ordering) (use 'random')
    #                        diagnostic codes only (use 'diag'), medications only (use 'med'), labs only, etc. 
    # seq_ptype = kargs.get('seq_ptype', 'regular')

    if d2v_method is not None: 
        assert vector.D2V.isSupported(d2v_method), "d2v method: %s is not supported yet" % d2v_method
        

    # doctype = 'model'
    # descriptor = kargs.get('meta', doc_basename)
    # if descriptor is None: descriptor = 'test'

    # [params] training 
    # these are configured via seqparams or vector 
    # n_features = kargs.get('n_features', GNFeatures)
    # window = kargs.get('window', GWindow)
    # min_count = kargs.get('min_count', GMinCount)
    # n_cores = multiprocessing.cpu_count()
    # print('word2vec> number of cores: %d' % n_cores)
    # n_workers = kargs.get('n_workers', max(n_cores-10, 15))

    # n_docs = len(sequences)

    # compute or load w2v model
    model = getWordVecModel(sequences, outputdir=outputdir) 

    if kargs.get('test_model', False): 
        test_similarity(model, sequences=sequences, seq_ptype=seq_ptype, n_features=n_features, cohort=cohort_name)

    # if kargs.get('test_accuracy', True)
    #     model.accuracy(args.accuracy)

    return model

def makeD2VLabels(sequences, **kargs): 
    """
    Label sequences/sentences for the purpose of using Doc2Vec. 

    Adapted from seqCluster.makeD2VLabels()

    Related 
    -------
    * labelDoc()

    """ 
    import labeling
    return labeling.makeD2VLabels(sequences, **kargs)

# [deprecated]
def labelDocByFreqDiag(seqx, **kargs): 
    import labeling
    return labeling.labelDocByFreqDiag(seqx, **kargs) 

def labelDoc0(sequences, **kargs): 
    """
    Label (patient) documents via heuristics (e.g. highest code frequencies) 

    Adapted from seqCluster.labelDoc

    Output
    ------
    1. df(label, sequence)
       where label is in diagnostic code-based multilabel format and 
             sequence constains only a subset of the orignial sequence 
             with length determined by 'topn_repr'

    """
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)

    seq_compo = composition = kargs.get('composition', 'condition_drug')
    read_mode = kargs.get('read_mode', 'doc')  # doc: per-patient documents; seq: per-visit documents/sentences
    seq_ptype = seqparams.normalize_ctype(**kargs) # values: regular, random, diag, med, lab ... default: regular
    cohort_name = kargs.get('cohort', 'diabetes')

    simplify_code = kargs.get('simplify_code', False)
    load_label = kargs.get('load_', False)
    doctype = 'csv' 
    
    basedir = sys_config.read('DataExpRoot')  # [I/O] global data directory (cf: local data directory /data)
    fsep = '|'
    lsep = '_' # label separator (alternatively, '+')

    seqr_type = kargs.get('seqr', 'full') # sequence string type: full vs diag 
    sortby = kargs.get('sortby', 'freq') # sort labels by their frequencies or alphabetic order? only applicable to multilabeling

    # [input]
    assert sequences is not None and len(sequences) > 0

    n_doc = len(sequences)
    print('info> read %d doc' % n_doc)
    
    # [policy] label by diagnostic codes
    repr_seqx = [None] * n_doc  # representative sequence (formerly diag_seqx)
    n_anomaly = 0
    if seq_ptype in ('regular', 'random', 'diag', ):  # only use diagnostic codes for labeling in these cases
        for i, sequence in enumerate(sequences): 
            # [c for c in sequence if pmed.isICD(e)] 
            repr_seqx[i] = filter(pmed.isICD, sequence)  # use diagnostic codes to label the sequence
            if len(repr_seqx[i]) == 0: 
                print('warning> No diagnostic code found in %d-th sequence/doc:\n%s\n' % (i, to_str(sequence)))
                n_anomaly += 1 
        div(message='A total of %d out of %d documents have valid diag codes.' % (n_doc-n_anomaly, n_doc), symbol='%')
    elif seq_ptype in ('med', ): 
        for i, sequence in enumerate(sequences): 
            # [c for c in sequence if pmed.isICD(e)] 
            repr_seqx[i] = filter(pmed.isMed, sequence)
            if len(repr_seqx[i]) == 0: 
                print('warning> No med code found in %d-th sequence/doc:\n%s\n' % (i, to_str(sequence)))
                n_anomaly += 1 
        div(message='A total of %d out of %d documents have valid med codes.' % (n_doc-n_anomaly, n_doc), symbol='%')
        # div(message='Only examining non-diagnostic codes > seq_ptype: %s' % seq_ptype, symbol='%')
    else: 
        # label the sequence by a customized criterion
        raise NotImplementedError, "Unsupported sequence type: %s" % seq_ptype

    ### I. single lable 
    sldict = {}  # single-label dictionary

    # save two labeling formats: single label & multilabel format
    # 1a. label + top 10 in whole sequence (.csv) 
    # 1b. label + top 10 in diag sequence (.csv) 
    # 2?. label + whole doc (.txt)

    topn, topn_repr = 1, 10
    header = ['label', 'sequence']
    freq_cseq_map = {h: [] for h in header}  # most frequent from complete sequences
    freq_diag_map = {h: [] for h in header}  # most frequent from diag sequences

    for i, dseq in enumerate(repr_seqx): 
        counter_diag = collections.Counter(dseq)
        counter_full = collections.Counter(sequences[i])

        # labeling 
        if dseq: 
            label = counter_diag.most_common(1)[0][0]
        else: 
            label = 'unknown' # no diagnostic codes
        # labels.append(label)

        # complete sequence
        seqr = to_str([pair[0] for pair in counter_full.most_common(topn_repr)])
        freq_cseq_map['label'].append(label)
        freq_cseq_map['sequence'].append(seqr)

        # diag sequence
        seqr = to_str([pair[0] for pair in counter_diag.most_common(topn_repr)])
        freq_diag_map['label'].append(label)
        freq_diag_map['sequence'].append(seqr)

    ### II. multi-label 
    mldict = {}  # multi-label dictionary
    topn, topn_repr = 3, 10  # use 'topn' most frequent labels
    # header = ['label', 'sequence']
    freq_cseq_map = {h: [] for h in header}  # most frequent from complete sequences
    freq_diag_map = {h: [] for h in header}  # most frequent from diag sequences

    for i, dseq in enumerate(repr_seqx): 
        counter_diag = collections.Counter(dseq)
        counter_full = collections.Counter(sequences[i])

        # labeling 
        if dseq: 
            # use frequent diag codes as label 
            ltuples = counter_diag.most_common(topn)  # if topn > # unique token, will only show topn (symbol, count)-tuples

            # sort according to frequencies? 
            if sortby.startswith('freq'): 
                sl = sorted(ltuples, key=lambda x: x[1], reverse=True) # sort ~ count from high to low
                label = to_str([l[0] for l in sl], sep=lsep)
            else: 
                label = to_str(sorted([l[0] for l in ltuples]), sep=lsep)
        else: 
            label = 'unknown'

        # mlabels.append(label)

        # complete sequence
        seqr = to_str([pair[0] for pair in counter_full.most_common(topn_repr)])
        freq_cseq_map['label'].append(label)
        # freq_cseq_map['sequence'].append(seqr)

        # diag sequence
        seqr = to_str([pair[0] for pair in counter_diag.most_common(topn_repr)])
        freq_diag_map['label'].append(label)
        # freq_diag_map['sequence'].append(seqr)

    if seqr_type.startswith('full'): 
        return freq_cseq_map['label']
    return freq_diag_map['label']

def polarize(target, codeset=None, simplify_code=False):
    """
    Separate positive sample from negative sample, where 
    positive sample matches with the target code according to 
    a pre-defined policy (e.g. ICD-9 codes sharing the same 
    first three digits are considered as "similar")

    target: 250.0 
    positive: 250.0, 250.53 ... 
    negative: 249.1, 249.10, 648.00 ... 

	Memo
	----
	1. Policy 
	   a. different base codes 

    """
    if codeset is None: 
    	codeset = t_diabetes_codes(simplify_code=simplify_code)
    
    # print('codeset:\n%s\n' % codeset)
    tc = icd9utils.getRootCode2(target)
    pos, neg = set(), set()
    for code in codeset: 
    	c = icd9utils.getRootCode2(code)
        if c == tc: 
        	pos.add(code)
        else: 
        	neg.add(code)

    return (list(pos), list(neg))

def hsample(codeset, n_sample=5): 
    """
    Build key-value pairs first and for each key, sample a subset of its corresponding values.
    """
    # random.seed(10)

    adict = {}
    for code in codeset: 
        c = icd9utils.getRootCode2(code)
        if not adict.has_key(c): adict[c] = []
        adict[c].append(code)

    return sampling.sample_hashtable(adict, n_sample=n_sample)

def loadW2VModel(**kargs): 
    return loadModel(**kargs)
def loadModel(**kargs): 
    """
    A wrapper that accomplishes the following 3 operations: 
    
    1. read sequences 
    2. analyze sequences (symbol lookup)
    3. compute/load w2v vectors
         i.e. Load pre-computed w2v model (if not existed, compute it). 


    Related
    -------
    tokenize()
    docsToTokens()
    t_word2vec()


    """
    def show_params():
        msg = 'params> n_features:%d, window:%d, min_count:%d | w2v:%s | seq_ptype:%s, read_mode:%s | test_model:%s, lookup? %s' % \
                    (n_features, window, min_count, w2v_method, seq_ptype, read_mode, test_model, load_lookuptb)

        # msg = 'Parameter settings:\n1. load w2v-model: %s\n2. load_seq: %s\n2a. simplify_code: %s\n3. load_lookuptb: %s\n' % \
        #       (load_model, load_seq, simplify_code, load_lookuptb)
        div(message=msg)
        return

    result = {}

    # [I/O] read, analyze, vectorize

    # [params]
    seq_compo = kargs.get('composition', 'condition_drug') # what does the sequence consist of? 
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')
    seq_ptype = seqparams.normalize_ctype(**kargs) # values: regular, random, diag, med, lab ... default: regular
    # doc_basename = 'condition_drug'
    # if not seq_ptype.startswith('reg'): 
    #     doc_basename = '%s-%s' % (doc_basename, seq_ptype)

    # set lod to true
    read_mode = kargs.get('read_mode', 'doc')  # documents/doc (one patient one sequence) or sequences/seq
    
    # [input] temporal sequences and word2vec model
    default_ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo
    ifile = kargs.get('ifile', default_ifile) 
    ifiles = kargs.get('ifiles', None) # [todo]

    load_model = kargs.get('load_model', True) # load the pre-computed word2vec model
    load_seq = kargs.get('load_seq', False)  # load the processed sequences
    load_lookuptb = kargs.get('load_lookuptb', True) # symbol lookup table (takes long to compute due to querying via REST)
    bypass_lookup = kargs.get('bypass_lookup', False)
    test_model = kargs.get('test_model', False)

    simplify_code = kargs.get('simplify_code', False)
    verify_seq = False

    # [params] training 
    w2v_method = kargs.get('w2v_method', 'sg')  # sg: skip-gram
    d2v_method = kargs.get('d2v_method', None) # if d2v_method is specified, then it may subsume w2v_method setting

    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)
    n_cores = multiprocessing.cpu_count()
    print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-10, 15))

    show_params()

    print('input> temporal doc file: %s' % ifile)
    seqx = read(load_=load_seq, simplify_code=simplify_code, mode=read_mode, verify_=verify_seq, 
                    seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name)
    div(message='io> read in %d sequences' % (len(seqx)))

    assert seqx is not None and len(seqx) > 0
    if read_mode.startswith('doc'): 
        # find average sequence length 
        seqlen = sum([len(seq) for seq in seqx])/len(seqx)
        # print('params> (tentative) Re-adjusting window length from %d to %d' % (window, seqlen))
        # window = seqlen

    result['sequences'] = seqx 

    # map_tokens: token_type => tokens, token_type: {'diag', 'drug', 'lab', 'other', }
    map_tokens, lookuptb = analyze(tokens=docsToTokens(seqx, uniq=False), load_=load_lookuptb, bypass_lookup=bypass_lookup, cohort=cohort_name)
    div(message='lookup> Completed token lookup (bypass_lookup? %s, cohort:%s)' % (bypass_lookup, cohort_name))

    assert len(map_tokens) > 0
    result['lookup'] = result['symbol_chart'] = lookuptb
    result['token_type'] = map_tokens  # type: diag, drug, ... -> tokens

    # train model  params: meta=read_mode
    # [note] seq_ptype affects only in the naming of the model file for vectorize()
    
    # [policy][todo]
    # simplify the code prior to training
    # use seqparams to configure W2V, D2V parameters
    model = vectorize(seqx, test_model=test_model, cohort=cohort_name)
    div(message='w2v> Computed word vectors (params> nf:%d, w:%d, mcnt:%d, cohort:%s, test_model?%s)' % \
        (n_features, window, min_count, cohort_name, test_model))
    print('test> word vector dim:\n%s\n' % str(model.syn0.shape))

    # no more updates, only querying =>  trim unneeded model memory = use (much) less RAM.
    # model.init_sims(replace=True)
    result['model'] = model

    n_diag, n_drug, n_other = len(map_tokens['diag']), len(map_tokens['drug']), len(map_tokens['other'])
    msg = 'number of condition tokens: %d\n' % n_diag
    msg += 'number of prescription tokens: %d\n' % n_drug 
    msg += 'number of other tokens: %d' % n_other
    div(message=msg, symbol='%') 

    return result # (sequences, lookup, model)

def t_diabetes_codes(simplify_code=False): 
    # Diabetes mellitus without complication
    code_str = '24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546'

    # Diabetes mellitus with complications
    code_str += ' ' + """24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
            25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
            25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093"""

    # Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium
    code_str += ' ' + "64800 64801 64802 64803 64804 64880 64881 64882 64883 64884" 	

    codes = icd9utils.preproc_code(code_str, base_only=simplify_code)
    print('> n_codes: %d\n> codes:\n%s\n' % (len(codes), codes))
    print('> unique codes:\n%s\n' % list(set(codes)))
    # [log] unique: ['791', '790', 'V65', 'V45', 'V53', '648', '250', '249']
    #       all: ['250.71', '249.01', '249.00', 'V65.46', '249.81', '249.80', 'V45.85', '250.60', '250.21', '791.6', '250.23', '250.22', 
    #             '250.43', '250.42', '250.41', '249.40', '250.03', '250.02', '250.01', '250.00', '250.81', '648.83', '250.83', '250.82', 
    #             '249.50', '249.51', '250.53', '249.70', '249.71', '250.63', '250.62', 'V53.91', '250.92', '250.20', '249.30', '249.31', 
    #             '250.93', '648.04', '249.10', '249.11', '250.31', '249.90', '249.91', '648.84', '648.82', '250.52', '648.80', '648.81', 
    #             '250.32', '250.33', '250.30', '790.29', '648.02', '648.03', '648.00', '648.01', '790.22', '250.61', '250.80', '790.21', 
    #             '250.10', '250.11', '250.12', '250.13', '249.41', '790.2', '791.5', '250.90', '250.91', '249.21', '249.20', '250.50', '250.51', 
    #             '249.61', '249.60', '250.72', '250.73', '250.70', '250.40']

    return codes

def t_negative_sample(**kargs):
    simplify_code = kargs.get('simplify_code', False) 
    codeset = t_diabetes_codes(simplify_code=simplify_code)
    
    tc = '250.01'
    pos, neg = polarize(target=tc, codeset=codeset, simplify_code=simplify_code)

    print('target: %s' % tc)
    print('pos:\n%s\n' % pos)
    print('neg:\n%s\n' % neg)

    return

def w_vectorize(t_vec, **kargs): # wrap it in order to time it 
    def wrapped(): 
        return t_vec(**kargs)
    return wrapped
def t_vectorize(**kargs): 
	# [params]
	load_model = kargs.get('load_model', False)

	seqx = read(load_=True, simplify_code=False)
	assert seqx is not None and len(seqx) > 0
	model = vectorize(seqx, load_model=load_model)

	return model

def getTestcodes(**kargs):
    """
    
    Use
    ---
    test_similarity()

    Memo
    ----
    1. PTSD code set: 

    [('309.81', 5360), ('311', 2985), ('300.00', 2064), ('401.9', 1726), ('296.30', 1654), 
     ('789.00', 1508), ('296.20', 1506), ('784.0', 1436), ('493.90', 1258), ('465.9', 1164), 
     ('V70.0', 1161), ('786.50', 1126), ('V62.89', 1123), ('599.0', 1107), ('724.2', 1065), 
     ('E849.8', 1030), ('729.5', 1010), ('V67.59', 946), ('V65.44', 933), ('300.4', 920)]

    2. CKD code set: 

    """
    import collections
    cohort_name = kargs['cohort']
    print('info> getting diag codes for cohort=%s' % cohort_name)
    # configure code set

    codeset_root, codeset = [], []
    if cohort_name.lower().startswith('diab'): 
        # [params] input source (diag) codes
        codeset = t_diabetes_codes(simplify_code=False)
        codeset_root = t_diabetes_codes(simplify_code=True)
    elif cohort_name.lower().startswith('pt'): 
        codeset = ['309.81', 'F43.1', '311', '300', '401.9', '296.30', 
                   '789.00', '296.20', '784.0', '493.90', ] # only '309.81', 'F43.1' are truly PTSD
        # codeset_root = ['309', 'F43']
    else: 
        # print('Warning: unknown cohort: %s' % cohort_name)
        topn = 100
        simplify_code = kargs.get('simplify_code', False)
        ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')

        # following segement can also be replaced by tokenize()
        seqx = read(load_=False, simplify_code=simplify_code, mode='doc', verify_=False, 
                    seq_ptype='regular', ifile=ifile, cohort=cohort_name)
        div(message='io> read in %d sequences' % (len(seqx)))
        codes = docsToTokens(seqx, uniq=False)
        counter = collections.Counter(codes)  
        freqCodes = counter.most_common(topn)  # topn are mostly unigrams
        print('info> frequent codes: %s' % freqCodes[:10])
        codeset = [c for c, cnt in freqCodes]
    
    if not codeset_root: 
        for c in codeset: 
            if c.find('.') > 0: 
                ce = c.split('.')[0]
            else: 
                ce = c 
            codeset_root.append(ce)
    return (codeset, codeset_root)

def test_similarity(model, **kargs):  # derivative of template code: t_similarity()
    """
    Test if the learned word vectors are useful. 
    Need to 'configure'  getTestcodes()
    
    test_similarity(model, sequences, seq_ptype, n_features)

    Use
    ---
    1. In loadModel() when test mode is activated. 

    Related 
    -------
    seqCluster.test_similarity(): 
       test similarity on the document level (i.e. based on document vectors)


    """
    def name_ofile(ext='csv'): 
        if n_features is None: 
            ofile = 'similarity_diag_%s' % seq_ptype
        else: 
            ofile = 'similarity_diag_%s_f%d' % (seq_ptype, n_features)

        if cohort_name is not None: 
            ofile = ofile + '-' + cohort_name
        
        ofile += '.csv'
        return ofile
    def rsubset(population, n): # random subset
        return random.sample(population, min(len(population), n))
    def show_params():
        msg = 'Parameter settings:\n1. load_model: %s\n2. load_seq: %s\n2a. simplify_code: %s\n3. load_lookuptb: %s\n' % \
              (load_model, load_seq, simplify_code, load_lookuptb)
        div(message=msg)
        return
    def get_modeled_set(candidates): 
        csubset = []
        # model.syn0[c]  # a word is a medical code here
        for c in candidates: 
            is_modeled = True
            try: 
                model[c]  
            except: 
                print('test> code %s is not being modeled (min_count > 1?)' % c)
                is_modeled = False 
            if is_modeled: 
                # assert model.has_key(c) # Word2vec object has no attribute 'has_key'
                csubset.append(c)
        return csubset
    def remove_from(x, alist):
        try: 
            alist.remove(x)
        except:
            pass
        return alist
        
    from batchpheno.utils import indent
    # no more updates, only querying =>  trim unneeded model memory = use (much) less RAM.
    # model.init_sims(replace=True)

    # [params]
    cohort_name = kargs.get('cohort', 'diabetes')

    # configure code set
    codeset, codeset_root = getTestcodes(cohort=cohort_name)
    if len(codeset) == 0: 
        print('Warning: test_similarity > no input code set available for cohort=%s > exiting ...' % cohort_name)
        return

    basedir = outputdir = seqparams.get_basedir(cohort=cohort_name) # kargs.get('output_dir', os.path.join(os.getcwd(), 'data/%s' % cohort_name))

    bypass_lookup = kargs.get('bypass_lookup', True)
    simplify_code = kargs.get('simplify_code', False)

    n_features = kargs.get('n_features', GNFeatures)
    sequences = docs = seqparams.arg(['sequences', 'docs'], None, **kargs)  #  kargs.get('sequences', kargs.get('docs', None))
    seq_ptype = kargs.get('seq_ptype', 'regular')
    n_root = len(codeset_root)
    topn = 3

    div('0. Activating test on similarity ... analyzing docuemnts ...')
    tokens = docsToTokens(sequences, uniq=False) if sequences is not None else None
    n_tokens = len(tokens)
    res, lookuptb = analyze(tokens=tokens, load_=True, model=model, bypass_lookup=bypass_lookup, cohort=cohort_name) # analyze should only return modeled codes
    n_modeled = len(res['modeled'])
    if tokens is not None: 
        n_tokens = len(tokens)
        print('+ out of %d tokens, %d are modeled > ratio: %f' % (n_tokens, n_modeled, n_modeled/(n_tokens+0.0)))

    div('1. Key diagnostic codes in the model?')
    modeled_diag_codes = get_modeled_set(codeset) # pick the codes that actually appear in the model
    n_modeled_diag =  len(modeled_diag_codes)
    print('+ Found %d diag codes in the model.' % n_modeled_diag)
   
    div('2. Can reasonably separate negative from positive?')
    # hsample(codeset, n_sample=n_root)
    for target in hsample(codeset, n_sample=n_root+5):

        # given target code, find its positive and negative example codes
        pos, neg = polarize(target=target, codeset=codeset, simplify_code=simplify_code)
        try: 
            div(message='A. Synonym for %s' % target, symbol='*')
            for c in rsubset(pos, 3): 
                s = model.similarity(target, c)
                print('   %s ~ %s: %f' % (target, c, s))
            div(message='B. Antonym for %s' % target, symbol='*')
            for c in rsubset(neg, 3): 
                s = model.similarity(target, c)
                print('   %s != %s: %f' % (target, c, s))
            print('\n')
        except Exception, e: 
            print('warning> %s' % e)
            print('         Could not evaluate %s > skipping target %s' % (target, target))
            continue

    # [output][save]
    fpath = os.path.join(basedir, name_ofile())
    # pickle.load(open(fpath, 'rb')) 

    # find the most similar codes given a target
    header = ['code', 'similar', 'description', 'similar_description']
    # adict = dict.fromkeys(header, None)
    adict = {}
    for h in header: 
        if not adict.has_key(h): adict[h] = []
    
    n_diag, n_drug, n_other = len(res['diag']), len(res['drug']), len(res['other'])
    print('stats> size(diag): %d, size(drug): %d, size(other): %d => total: %d' % \
        (n_diag, n_drug, n_other, (n_diag+n_drug+n_other)))

    # modeled_codes = get_modeled_set(res['diag'])
    n_topn_match = n_first_match = 0  # compared to diagnostic codes only 
    n_topn_match2 = n_first_match2 = 0 # compared to all codes

    for code in modeled_diag_codes: 
        # if not code in targets: continue
        othercodes = set(modeled_diag_codes) - set([code])
        code_scores = [(othercode, model.similarity(code, othercode)) for othercode in othercodes]  
        scores = sorted(code_scores, key=lambda x:x[1], reverse=True) # desending order
        cutoff = min(len(scores), topn)  # top n similar codes

        # simcodes: comparison with only the other diagnostic codes
        simcodes = [str(s[0]) for s in scores[:cutoff]]

        # out of n_modeled (diag) codes, find out the fraction of true positives within the candidate set (among top N similar codes)
        iset = icd9utils.match(target=code, candidates=simcodes) # base_only=True for loose match
        if iset: # [(pos1, matched_code1), ...]
            n_topn_match += 1 
            if iset[0][0] == 0: 
                n_first_match += 1

        # simcodes2: this would include symbols in the "drug" category (in addition to the other diag codes)
        simcodes2 = model.similar_by_word(code, topn=topn+1, restrict_vocab=None) # restrict_vocab is used to filter infrequent words
        iset2 = icd9utils.match(target=code, candidates=simcodes2) # base_only=True for loose match
        if iset2: # [(pos1, matched_code1), ...]
            n_topn_match2 += 1 
            if iset2[0][0] == 0: 
                n_first_match2 += 1

        # [log] result not including myself e.g. code: 250.01 | simcodes: ['250.03', 'V45.85', '250.91']
        print('info> code: %s | simcodes: %s =?= %s' % (code, str(simcodes), str(simcodes2)))

        simcodes_str = ' '.join(simcodes)
        print('summary> code %s ~ %s' % (code, simcodes_str))
        
        # detailed descriptions 
        print('lookup> Finding code interpretation ...')
        message = indent('%s: %s\n' % (code, lookuptb.get(code, '?')), nfill=6, mode='r')
        # "{:>8}".format('%s: %s\n' % (c, lookuptb.get(c, '?')))
        for c in simcodes:
            message += indent('%s: %s\n' % (c, lookuptb.get(c, '?')), nfill=10, mode='r') 
        div(message=message, symbol='#')

        # print('vector> code: %s:\n%s' % (code, model[code]))
        
        # save 
        adict['code'].append(code) 
        adict['similar'].append(simcodes_str)
        adict['description'].append(lookuptb.get(code, '?'))
        adict['similar_description'].append(lookuptb.get(simcodes[0], '?')) # icd9utils.lookup2(simcodes[0]

    ### end foreach (modeled) code

    # [stats]
    r_topn, r_first = n_topn_match/(n_modeled_diag+0.0), n_first_match/(n_modeled_diag+0.0)
    msg = 'Result> Seq type: %s > Among %d modeled diag codes, n_topn_match=%d(ratio: %f), n_first_match=%d (%f)' % \
            (seq_ptype, n_modeled_diag, n_topn_match, r_topn, n_first_match, r_first)
    div(message=msg, symbol='*')

    r_topn2, r_first2 = n_topn_match2/(n_modeled+0.0), n_first_match2/(n_modeled+0.0)
    msg = 'Result> Seq type: %s > Among %d modeled codes in GENERAL, n_topn_match2=%d(ratio: %f), n_first_match2=%d (%f)' % \
            (seq_ptype, n_modeled, n_topn_match2, r_topn2, n_first_match2, r_first2)
    div(message=msg, symbol='*')

    df = DataFrame(adict, columns=header)
    df.to_csv(fpath, sep='|', index=False, header=True) # [output]

    # e.g. similarity_diag_regular_f100-PTSD.csv
    print('io> saved test_similarity result dataframe (dim=%s) to:\n%s\n' % (str(df.shape), fpath))

    # save similarity matrices

    return

def t_similarity(**kargs): 
    def rsubset(population, n): # random subset
        return random.sample(population, min(len(population), n))

    def show_params():
        msg = 'Parameter settings:\n1. load_model: %s\n2. load_seq: %s\n2a. simplify_code: %s\n3. load_lookuptb: %s\n' % \
              (load_model, load_seq, simplify_code, load_lookuptb)
        div(message=msg)
        return
    
    from batchpheno.utils import indent
    # [output]
    # odict = {'sequences': None, 'lookuptb': None, 'model': None}

    # [params]
    output_dir = kargs.get('output_dir', sys_config.read('DataExpRoot'))
    cohort_name = kargs.get('cohort', 'diabetes')
    
    ifile = kargs.get('ifile', None) # input temporal doc
    load_model = kargs.get('load_model', False) # load the pre-computed word2vec model
    load_seq = kargs.get('load_seq', False)  # load the processed sequences
    load_lookuptb = kargs.get('load_lookuptb', False) # symbol lookup table (takes long to compute due to querying via REST)
    bypass_lookup = kargs.get('bypass_lookup', False)
    simplify_code = kargs.get('simplify_code', False)

    codeset = t_diabetes_codes(simplify_code=False)
    codeset_root = t_diabetes_codes(simplify_code=True)
    n_root = len(codeset_root)
    topn = 3

    # # [params] training 
    # n_features = kargs.get('n_features', GNFeatures)
    # window = kargs.get('window', GWindow)
    # min_count = kargs.get('min_count', GMinCount)

    print('info> %d codes > %d roots' % (len(codeset), n_root))
    # target = '250.01'
    show_params()

    div(message='1. train model')
    # model = t_vectorize(load_model=load_model)

    seqx = read(load_=load_seq, simplify_code=simplify_code, read_mode='doc', ifile=ifile)
    assert seqx is not None and len(seqx) > 0
    # res = classify_codes(tokens=seq_to_token(seqx), save_table=True)
    res, lookuptb = analyze(tokens=docsToTokens(seqx, uniq=False), load_=load_lookuptb, bypass_lookup=bypass_lookup)
    assert len(res['diag']) > 0
    model = vectorize(seqx, load_model=load_model)
    
    # no more updates, only querying =>  trim unneeded model memory = use (much) less RAM.
    # model.init_sims(replace=True)

    # save output for later use e.g. clustering 
    # odict

    div(message='2. test similarity')

    targets = set()
    for c in codeset:  # [log] 249.31 is not in the model
    	is_in = False
        try: 
        	model[c]
        	is_in = True
        except: 
            print('warning> code %s is not in the model' % c)
        if is_in: 
        	targets.add(c)
    div(message='Found %d codes in the model.' % len(targets), symbol='%')

    # hsample(codeset, n_sample=n_root)
    for target in hsample(codeset, n_sample=n_root):
        pos, neg = polarize(target=target, codeset=codeset, simplify_code=simplify_code)

        try: 
            div(message='A. Synonym for %s' % target, symbol='*')
            for c in rsubset(pos, 3): 
                s = model.similarity(target, c)
                print('   %s ~ %s: %f' % (target, c, s))
            div(message='B. Antonym for %s' % target, symbol='*')
            for c in rsubset(neg, 3): 
                s = model.similarity(target, c)
                print('   %s != %s: %f' % (target, c, s))
            print('\n')
        except Exception, e: 
            print('warning> %s' % e)
            print('         skipping target %s' % target)
            continue
    
    # [save]
    fpath = os.path.join(output_dir, 'similarity_diag-%s.csv' % cohort_name)
    # pickle.load(open(fpath, 'rb')) 
    # pickle.dump(sequences, open(fpath, "wb" ))

    # find the most similar codes given a target
    header = ['code', 'similar', 'description', 'similar_description']
    # adict = dict.fromkeys(header, None)
    adict = {}
    for h in header: 
    	if not adict.has_key(h): adict[h] = []
    
    n_diag, n_drug, n_other = len(res['diag']), len(res['drug']), len(res['other'])
    print('stats> size(diag): %d, size(drug): %d, size(other): %d => total: %d' % \
    	(n_diag, n_drug, n_other, (n_diag+n_drug+n_other)))

    for code in res['diag']: 
    	if not code in targets: continue
    	othercodes = set(res['diag']) - set([code])
    	scores = [(othercode, model.similarity(code, othercode)) for othercode in othercodes]  
        scores = sorted(scores, key=lambda x:x[1], reverse=True) # desending order
        cutoff = min(len(scores), topn)
        simcodes = [str(s[0]) for s in scores[:cutoff]]

        # this would include symbols in the "drug" category
        simcodes2 = model.similar_by_word(code, topn=topn+1, restrict_vocab=None) # restrict_vocab is used to filter infrequent words
        print('info> code: %s | simcodes: %s =?= %s' % (code, str(simcodes), str(simcodes2)))

        simcodes_str = ' '.join(simcodes)
        print('summary> code %s ~ %s' % (code, simcodes_str))
        
        # detailed descriptions 
        message = indent('%s: %s\n' % (code, lookuptb.get(code, '?')), nfill=6, mode='r')
        # "{:>8}".format('%s: %s\n' % (c, lookuptb.get(c, '?')))
        for c in simcodes:
            message += indent('%s: %s\n' % (c, lookuptb.get(c, '?')), nfill=10, mode='r') 
        div(message=message, symbol='#')

        # print('vector> code: %s:\n%s' % (code, model[code]))
        
        # save 
        adict['code'].append(code) 
        adict['similar'].append(simcodes_str)
        adict['description'].append(lookuptb.get(code, '?'))
        adict['similar_description'].append(lookuptb.get(simcodes[0], '?')) # icd9utils.lookup2(simcodes[0]

    df = DataFrame(adict, columns=header)
    df.to_csv(fpath, sep='|', index=False, header=True)
    print('output> (1) saving diagnoses df (dim=%s) to:\n%s\n' % (str(df.shape), fpath))

    adict = {}; gc.collect()
    fpath = os.path.join(output_dir, 'similarity_drug.csv')

    # find most similar drugs 
    header = ['drug', 'similar', 'description', 'similar_description']
    for h in header: 
    	if not adict.has_key(h): adict[h] = []

    cutoff = min(100, len(res['drug']))  # inspect at most N drugs
    for drug in list(res['drug'])[:cutoff]: 
        otherdrugs = set(res['drug']) - set([drug])
        scores = [(otherdrug, model.similarity(drug, otherdrug)) for otherdrug in otherdrugs]  
        scores = sorted(scores, key=lambda x:x[1], reverse=True)  # desending order
        # cutoff = min(len(scores), 3)
        simdrugs = [str(s[0]) for s in scores[:3]]
        
        simdrugs2 = model.similar_by_word(drug, topn=topn, restrict_vocab=None)
        print('info> drug: %s | simcodes: %s =?= %s' % (drug, str(simdrugs), str(simdrugs2)))

        simdrugs_str = ' '.join(simdrugs)
        print('summary> drug %s ~ %s' % (drug, simdrugs_str))

        # detailed descriptions 
        message = indent('%s: %s\n' % (drug, lookuptb.get(drug, '?')), nfill=6, mode='r')
        # "{:>8}".format('%s: %s\n' % (c, lookuptb.get(c, '?')))
        for d in simdrugs:
            message += indent('%s: %s\n' % (d, lookuptb.get(d, '?')), nfill=10, mode='r') 
        div(message=message, symbol='#')

        # save 
        adict['drug'].append(drug) 
        adict['similar'].append(simdrugs_str)
        adict['description'].append(lookuptb.get(drug, '?'))
        adict['similar_description'].append(lookuptb.get(simdrugs[0], '?')) 

        # qmed2? 
        # adict['description'].apppend(icd9utils.lookup2(drug))
        # adict['similar_description'].append(icd9utils.lookup2(simdrugs[0]))
    
    df = DataFrame(adict, columns=header)
    df.to_csv(fpath, sep='|', index=False, header=True)
    print('output> (2) saving medications df (dim=%s) to:\n%s\n' % (str(df.shape), fpath))

    print('verify> size of other values: %d' % len(res['other']))

    div(message='Mission completed ... :)', symbol='*')

    # [todo]
    adict = {}; gc.collect()
    fpath = os.path.join(output_dir, 'similarity_lab.csv')
    
    return (model, seqx, lookuptb)

def print_cluster(word_centroid_map):
    # For the first 10 clusters
    for cluster in range(0,10):
        # Print the cluster number  
        print "\nCluster #%d" % cluster

        # Find all of the words for that cluster number, and print them out
        # word_centroid_map: word2index => cluster id
        n_map = len(word_centroid_map)
        words = []
        for i in range(n_map):  # foreach entry
            if ( word_centroid_map.values()[i] == cluster ): 
                words.append(word_centroid_map.keys()[i])
        print words 

    return

def t_visualize(**kargs):
    pass

def t_word2vec0(**kargs):

    # [params]
    read_mode = 'doc'  # use doc to form per-patient sequences

    # [params] training 
    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)
    n_workers = kargs.get('n_workers', GNWorkers)

    # [params] load pre-computed data
    load_cluster = kargs.get('load_cluster', True)
    load_label = kargs.get('load_label', True)

    # [params] cluster document
    doctype = 'txt' 
    doc_basename = 'condition_drug'

    cohort_name = kargs.get('cohort', 'PTSD')
    bypass_lookup = kargs.get('bypass_lookup', True)

    # [input]
    # read_mode: {'seq', 'doc', 'csv'}  # 'seq' by default
    # cohort vs temp doc file: diabetes -> condition_drug_seq.dat
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')   
    result = loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
                ifile=ifile, cohort=cohort_name, bypass_lookup=True, 
                read_mode=read_mode, load_seq=False, load_model=True, load_lookuptb=True) # attributes: sequences, lookup, model
    div(message='Successfully loaded learned model ... :)', symbol='%')

    return   

def t_word2vec(**kargs):
    """

    Reference
    ---------
    
    Example
    -------
    python word2vec.py -train <fpath> -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
    
    Note that this word2vec.py comes from gemsim, not the word2vec.py in this module (which is used to demo TensorFlow)

    where fpath: <prefix>/tpheno/data-exp/condition_drug_seq.dat

    """
    import argparse, tp_parser, logging
    from numpy import seterr

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    logging.info("running %s", " ".join(sys.argv))
    logging.info("using optimization %s", FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    from gensim.models.word2vec import Word2Vec  # avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", help="Use text data from file TRAIN to train the model") # required=True
    parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors")
    parser.add_argument("-window", help="Set max skip length WINDOW between words; default is 5", type=int, default=5)
    parser.add_argument("-size", help="Set size of word vectors; default is 100", type=int, default=100)
    parser.add_argument("-sample", help="Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)", type=float, default=1e-3)
    parser.add_argument("-hs", help="Use Hierarchical Softmax; default is 0 (not used)", type=int, default=0, choices=[0, 1])
    parser.add_argument("-negative", help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)", type=int, default=5)
    parser.add_argument("-threads", help="Use THREADS threads (default 12)", type=int, default=12)
    parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=5)
    parser.add_argument("-min_count", help="This will discard words that appear less than MIN_COUNT times; default is 5", type=int, default=5)
    parser.add_argument("-cbow", help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)", type=int, default=1, choices=[0, 1])
    parser.add_argument("-binary", help="Save the resulting vectors in binary mode; default is 0 (off)", type=int, default=0, choices=[0, 1])
    parser.add_argument("-accuracy", help="Use questions from file ACCURACY to evaluate the model")

    args = parser.parse_args()

    if args.cbow == 0:
        skipgram = 1
    else:
        skipgram = 0

    # corpus = tp_parser.LineSentence(args.train)
    corpus = read(load_=True, simplify_code=False, mode='doc', verify_=False, seq_ptype='regular')
    n_doc = len(corpus)
    n_tokens = len(docsToTokens(corpus, uniq=False))
    print('info> n_doc: %d, n_tokens: %d' % (n_doc, n_tokens)) 

    model = Word2Vec(
        corpus, size=args.size, min_count=args.min_count, workers=args.threads,
        window=args.window, sample=args.sample, sg=skipgram, hs=args.hs,
        negative=args.negative, cbow_mean=1, iter=args.iter)

    if args.output:
        outfile = args.output
        model.wv.save_word2vec_format(outfile, binary=args.binary)
    else:
        outfile = args.train
        model.save(outfile + '.model')
    if args.binary == 1:
        model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
    else:
        model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

    if args.accuracy:
        model.accuracy(args.accuracy)

    logger.info("finished running %s", program) 

def t_lookup(**kargs): 
    """


    Memo
    ----
    1. Ran this on PTSD cohort with the following result: 

       info> found 27636 unique tokens (n_total: 1331636 | cohort=PTSD, ifile=condition_drug_seq-PTSD.dat)
      
       where, 

       n_total: number of tokens in the entire sequence set (from all patients)
                on the order of 1M+ 

       unique tokens: 27K+ 

    """

    seq_compo = kargs.get('composition', 'condition_drug') # what does the sequence consist of? 
    cohort_name =kargs.get('cohort', 'PTSD') 
    seq_ptype = 'regular' #seqparams.normalize_ctype(**kargs) # values: regular, random, diag, med, lab ... default: regular

    # set lod to true
    read_mode = kargs.get('read_mode', 'doc')  # documents/doc (one patient one sequence) or sequences/seq
    
    # [input] temporal sequences and word2vec model
    default_ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo
    ifile = kargs.get('ifile', default_ifile) 

    print('input> temporal doc file: %s' % ifile)
    seqx = read(load_=False, simplify_code=False, mode=read_mode, verify_=False, 
                    seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name)
    div(message='io> read in %d sequences' % (len(seqx)))

    assert seqx is not None and len(seqx) > 0
    if read_mode.startswith('doc'): 
        # find average sequence length 
        seqlen = sum([len(seq) for seq in seqx])/len(seqx)
        # print('params> (tentative) Re-adjusting window length from %d to %d' % (window, seqlen))
        # window = seqlen

    # result['sequences'] = seqx 

    # map_tokens: token_type => tokens, token_type: {'diag', 'drug', 'lab', 'other', }
    tokens = seq_to_token(seqx) 
    n_tokens_total = len(tokens)
    utokens = list(set(tokens))
    n_tokens_uniq = len(utokens)
    print('info> found %d unique tokens (n_total: %d | cohort=%s, ifile=%s)' % (n_tokens_uniq, n_tokens_total, cohort_name, ifile))

    analyze_diag(utokens, cohort=cohort_name)
    print('status> completed diagnosis code lookup.')
    analyze_drug(utokens, cohort=cohort_name)
    print('status> completed med code lookup.')


    return

def t_cluster(**kargs): 
    msg = "this is to be done in module seqmaker.seqCluster()"
    raise NotImplementedError, msg  

def t_read_doc(**kargs): 
    
    seq_ptype = 'regular'
    ifile = None # use default 
    read_mode = 'timed'  # 'doc', 'seq'
    verify_seq = True
    cohort_name = 'PTSD'
    
    seqx, tseqx = read2(load_=False, save_=True, simplify_code=False, read_mode=read_mode, verify_=verify_seq, 
                         seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name)

    return 

def t_deidentify(**kargs):
    def transform_(seqx, pol='seq'):
        if pol == 'seq':
            return seqx # do nothing 
        elif pol == 'set': 
            seqx2 = []
            for i, seq in enumerate(seqx): 
                seqx2.append(sorted(set(seq)))
            return seqx2 
        else: 
            raise NotImplementedError, 'Unknown policy: %s' % pol 

        return seqx         
            
    import collections 
    import algorithms  # local library

    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'diag'))
    read_mode = 'timed'  # 'doc', 'seq'

    # data source from sequencing (entire population)
    n_parts = 10
    basedir = sys_config.read('DataExpRoot')
    basedir_sequencing = os.path.join(basedir, 'sequencing')  #

    # partition of the patient sets in sequencing data is identified by specialized cohort names 
    # e.g. condition_drug_timed_seq-group-10.dat  
    fbase = 'condition_drug_timed_seq'  # regular: condition_drug_seq
    policies = ['seq', 'set', ]

    cache = {}  # set to None if caching is not desired
    tCaching = True if isinstance(cache, dict) else False
    
    for policy in policies: 
        polling = {} 
        n_persons = 0
        seqx, tseqx = [], []
        for i in range(n_parts):
            cohort_name = 'group-%s' % (i+1)  
            ifile = os.path.join(basedir_sequencing, '%s-%s.dat' % (fbase, cohort_name))
            assert os.path.exists(ifile), "invalid ifile: %s" % ifile

            # if cache is None or not cache.has_key(cohort_name): 
            #     # no_throw: if True, then don't throw exception when time|code cannot be parsed successfully but simply ignore that pair
            #     seqx, tseqx = read2(load_=False, save_=False, simplify_code=False, read_mode='timed', verify_=False, 
            #                                     seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name, no_throw=True) 
            #     if isinstance(cache, dict): 
            #         cache[cohort_name] = seqx, tseqx
            # else: 
            #     seqx, tseqx = cache[cohort_name]
                
            seqx, tseqx = read2(load_=False, save_=False, simplify_code=False, read_mode='timed', verify_=False, 
                                seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name, no_throw=True) 
        
            # [todo] verify their ids 
            n_persons += len(seqx) 
            
            # [policy]
            n0 = len(seqx)
            seqx = transform_(seqx, pol=policy)
            n1 = len(seqx); assert n0 == n1

            # convert seqx to string for indexing 
            for seq in seqx: 
                seqstr = '_'.join(str(e) for e in seq)     
                if not polling.has_key(seqstr): polling[seqstr] = 0 
                polling[seqstr] += 1
        ### end foreach dataset 

        # [params] control 
        n_comm, n_lcomm = (100, 100)  # most common, least common 

        n_uniq_signatures = len(polling)
        r = n_uniq_signatures/(n_persons+0.0)

        countSig = collections.Counter(polling)
        comm_sig = countSig.most_common(100) 
        div(message='Most common sigatures (n=%d, policy=%s, ctype=%s) ...' % (n_comm, policy, seq_ptype))

        # [note] n_persons: 2302000 (2.3M), n_uniq_signatures: 1476414 (1.4M), ratio: 0.641361
        print('  + n_persons: %d, n_uniq_signatures: %d, ratio: %f' % (n_persons, n_uniq_signatures, r))

        for cs, cnt in comm_sig: 
            print('  + [cnt=%d] %s' % (cnt, cs))

        div(message='Least common sigatures (n=%d, policy=%s, ctype=%s) ...' % (policy, seq_ptype))
        print('  + n_persons: %d, n_uniq_signatures: %d, ratio: %f' % (n_persons, n_uniq_signatures, r))
        l_comm_sig = algorithms.least_common(countSig, 100)
        l_comm_sig = sorted(l_comm_sig, key=lambda x: x[1], reverse=True)

        for cs, cnt in l_comm_sig:  # reverse ordering for clarity? 
            print('  + [cnt=%d] %s' % (cnt, cs))

        # [todo] use panda plotting utility 

        print('status> analysis (policy=%s, ptype=%s) complete ------* '  % (policy, seq_ptype))

    return

def t_deidentify_batch(**kargs): 

    ptypes = ['regular', 'diag', 'med', ] # 'diag', 'med', 
    for ptype in ptypes: 
        t_deidentify(seq_ptype=ptype)

    return

def t_lookup2(**kargs): 
    t_lookup_restful(**kargs)
    return
def t_lookup_restful(**kargs): # look up codes via RESTful API 
    """

    Reference 
    ---------
    1. HIPPASpace: https://www.hipaaspace.com/Medical_Web_Services/Test.Drive.RESTful.Web.Services?Type=NDC#rt
    2. openFDA 

    Memo
    ----
    (*) HIPAASpace 

        https://www.HIPAASpace.com/api/{domain}/{operation}?q={query}&rt={result type}&token={token}
        
        Three query parameters are required with each “search” request:

        Use the domain parameter to specify required data domain.
        Use the operation parameter to specify “format_check” operation.
        Use the q (query) parameter to specify your query.
        Use the token (API key) query parameter to identify your application.

        Use the rt (result type) query parameter to specify required result type (json/xml/min.json/min.xml).

        Examples 

        https://www.hipaaspace.com/api/npi/getcode?q=1285636522&rt=xml&token=3932f3b0-cfab-11dc-95ff-0800200c9a663932f3b0-cfab-11dc-95ff-0800200c9a66

    (*) Coding format 
        National Drug Code, NDC  
           4-4-2 
           5-3-2
           5-4-1

    """
    import requests, json
    from urllib2 import Request, urlopen, URLError

    operations = {'getcode', 'getcodes', 'search', 'search_and_keywords', }
    rts = {'json', 'minjson' 'xml', 'minxml', }
    
    my_token = '2DECE6D8DEFE4158AAF4F936A3CEA5557DBD99D6EE3849D589745897EA74841B'
    demo_token = '3932f3b0-cfab-11dc-95ff-0800200c9a663932f3b0-cfab-11dc-95ff-0800200c9a66'

    params = {'domain': 'ndc', 'operation': 'getcode', 'query': '0093-0832-01', 'rt': 'json', 
               'token': my_token}
    

    # e.g. https://www.hipaaspace.com/api/npi/getcode?q=1285636522&rt=xml&token=3932f3b0-cfab-11dc-95ff-0800200c9a663932f3b0-cfab-11dc-95ff-0800200c9a66
    #      https://www.HIPAASpace.com/api/npi/getcode?q=1285636522&rt=xml&token=2DECE6D8DEFE4158AAF4F936A3CEA5557DBD99D6EE3849D589745897EA74841B
    uri = 'https://www.hipaaspace.com/api/{domain}/{operation}?q={query}&rt={rt}&token={token}'
    rquery = uri.format(**params)
    print("info> URI: %s" % rquery) 

    # [todo] this doesn't work, returns an empty content
    # div(message='\nTry requests.get() ...')
    # resp = requests.get(rquery)
    # if resp.status_code != 200:
    #     # This means something went wrong.
    #     raise ApiError('GET /tasks/ {}'.format(resp.status_code))
    # print resp

    div(message='\nTry using urlopen ...')
    request = Request(rquery)
    try:
        response = urlopen(request)
        ret = response.read()
        print ret
    except URLError, e:
        print 'Something went wrong: %s', e
 
    doc = json.loads(ret)
    print('info> dtype: %s' % type(doc))
    print('info> doc:\n%s\n' % doc['NDC'][0]['ProprietaryName'])
    # print('info> %s > %s' (doc['NDC'], doc['NDC'][0]['ProprietaryName']))
    # for todo_item in resp.json():
    #     print('{} {}'.format(todo_item['id'], todo_item['summary']))

    return
def t_test_restful(): 
    import requests

    # Set up the parameters we want to pass to the API.
    # This is the latitude and longitude of New York City.
    parameters = {"lat": 40.71, "lon": -74}

    # Make a get request with the parameters.
    response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)

    # Print the content of the response (the data the server returned)
    print(response.content)

    # This gets the same data as the command above
    response = requests.get("http://api.open-notify.org/iss-pass.json?lat=40.71&lon=-74")
    print(response.content)

    return

def t_labeling(**kargs):
    """

    Memo
    ----
    1. cohort=CKD 
        + sources
          condition_drug_seq-CKD.dat
          condition_drug_seq-CKD.id     // header: ['person_id', ]

        + labels: 
          available in the input file: eMerge_NKF_Stage_20170818.csv

          sometimes labels need to be computed

    """ 
    def read_ids(fname): 
        assert fname.find('.id') > 0
        fp = os.path.join(basedir, fname)
        assert os.path.exists(fp), 'Invalid input: %s' % fp
        df_id = pd.read_csv(fp, sep=sep, header=0, index_col=False, error_bad_lines=True)
        return df_id['person_id'].values
    def seq_to_str(seqx, sep=','): 
        return [sep.join(str(s) for s in seq) for seq in seqx] 
    def str_to_seq(df, col='sequence', sep=','):
        seqx = []
        for seqstr in df[col].values: 
            s = seqstr.split(sep)
            seqx.append(s)
        return seqx
    def to_str(tokens, sep='+'): 
        return sep.join([str(tok) for tok in tokens])

    import labeling

    ### CKD cohort 
    # basedir = sys_config.read('DataIn')  # data-in simlink to data ... 10.17 
    # 'data-in' is reserved for input data not generated from within the system 
    basedir = sys_config.read('DataExpRoot')
    
    # cohort attributes
    cohort_name = 'CKD'
    fname = 'eMerge_NKF_Stage_20170818.csv'    
    header = ['patientId', 'Case_Control_Unknown_Status', 'NKF_Stage', ]
    sep = ','

    fpath = os.path.join(basedir, fname)
    df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)

    # use stages as labels 
    labelset = list(df['NKF_Stage'].unique())
    print('info> cohort: %s | labels (n=%d):\n%s\n' % (cohort_name, len(labelset), labelset))
    # [log] 7 labels

    labels = df['NKF_Stage']

    # only read documents with data 
    idx = person_ids = labeling.getPersonIDs(cohort=cohort_name, inputdir=basedir, sep=sep)  # cohort, inputdir, sep, sep_compo
    seqparams.TDoc.verifyIDs(idx)  # ascending order? duplicates? 

    n_persons = len(idx)
    print('info> n_persons: %d' % n_persons)

    ### find labels
    # don't use the data source's ordering of IDs, which when made via seqMaker2.py was sorted
    # ERROR: labels = df.loc[df['patientId'].isin(idx)]['NKF_Stage'].values
    
    sort_keys = ['patientId', ]
    # df_test1 = df.sort_values(sort_keys, ascending=True)
    # l = df['patientId'].values
    # assert all(l[i] <= l[i+1] for i in xrange(len(l)-1))   # passed
    # assert all(l == idx) # passed

    # filter > sort > extract (good!)
    # output np.array
    # labels = labels_ref = labeling.labelDocByDataFrame(, person_ids=idx, id_field='patientId', label_field='NKF_Stage')
    labels = labels_ref = labeling.labelDocByFile(fpath, person_ids=idx, id_field='patientId', label_field='NKF_Stage')

    # n_labels = len(labels)
    # print('info> Got %d labels' % n_labels)

    # [test] verify the ID and labels
    # print('status> verifying the match between IDs and labels')
    # for i, (r, row) in enumerate(df_test1.iterrows()): # sorted entries
    #     pid, label = row['patientId'], row['NKF_Stage']
    #     if pid in idx: 
    #         assert label == labels[i], "%d-th label: %s <> %s" % (i, label, labels[i])
    ## [conclusion] the label ordering via df_test1 does not agree!!! 

    # extract labels according to the ID ordering
    # sampleIDs = random.sample(range(n_persons), 50)
    # labels = []
    # for pid in idx: 
    #     row = df.loc[df['patientId']==pid]  # row is a dataframe
    #     assert row.shape[0] == 1, 'Found dups: id=%s => %s' % (pid, row)
    #     l = list(row['NKF_Stage'].values)
    #     labels.extend(l)
    # assert len(labels) == len(labels_ref) == len(idx)    # passed
    # assert all(labels_ref == labels), "ordering inconsistency:\n%s\n VS \n%s\n" % (labels_ref[:50], labels[:50])  # passed

    n_labels = len(labels)
    print('info> verified %d labels' % n_labels)

    # double check with structured version of the sequences produced by seqMaker2 (header: person_id, sequence, timestamp)
    # tfile = 'condition_drug_timed_seq-%s.csv' % cohort_name # test file
    # fpath2 = os.path.join(basedir, tfile)
    # # if os.path.exists(fpath2): 
    # dft = pd.read_csv(fpath2, sep='|', header=0, index_col=False, error_bad_lines=True)
    # print('info> from timed_seq .csv | n_persons: %d =?= n_labels: %d' % (dft.shape[0], n_labels)) # n_persons: 2833 =?= n_labels: 2833

    ### Read Sequences

    print('info> 1. CSeq from .csv')
    ret = readDocFromCSV(cohort=cohort_name, inputdir=basedir)
    print('info> making structured format of the coding sequences (cohort:%s, n_labels:%d)' % (cohort_name, n_labels))
    # df = readToCSV(cohort=cohort_name, labels=labels)
    
    seqx = ret['sequence'] # list(dft['sequence'].values)
    tseqx = ret.get('timestamp', []) # list(dft['timestamp'].values)
    if tseqx: 
        assert len(seqx) == len(tseqx), "len(seqx)=%d, len(times)=%d" % (len(seqx), len(tseqx))

    print('info> 2. CSeq from .dat')
    seqx2, tseqx2 = readDoc(cohort=cohort_name, inputdir=basedir, include_timestamps=True) # ifiles

    # can then create .csv via readDocToCSV()  # [params]  cohort, basedir, ifiles, labels

    if tseqx2: 
        assert len(seqx2) == len(tseqx2), "len(seqx)=%d, len(times)=%d" % (len(seqx2), len(tseqx2))
    n_docs, n_docs2 = len(seqx), len(seqx2)
    
    # print('info> read %d from .dat =?= %d from .csv' % (n_docs2, n_docs))
    assert n_docs == n_docs2, ".dat and .csv formats are not consistent n_doc: %d (csv) <> %d (dat)" % (n_docs, n_docs2)

    # when did they diverge? 
    # n_matched = 0
    # for i, seq in enumerate(seqx): 
    #     s1 = seq # list of tokens

    #     try: 
    #         s2 = seqx2[i] # list of tokens
    #     except: 
    #         s2 = []

    #     if s1 == s2: 
    #         n_matched += 1 
    #     else: 
    #         msg = ".csv not consistent with .dat (n_matched=%d)=>\n%s\nVS\n%s\n" % (n_matched, s1, s2)
    #         raise ValueError, msg 

    n_docs_src = df.shape[0]
    assert n_docs == n_labels, "n_labels: %d <> n_docs: %d ..." % (n_labels, n_docs)

    print('input> n_doc_src (cohort source document): %d, n_doc (parsed, has data in DB): %d' % (n_docs_src, n_docs))

    print('info> writing labels to .csv')
    df2 = readDocToCSV(cohort=cohort_name, labels=labels)
    print('info> created .csv format with columns:\n%s\n' % df2.columns.values)
    n_docs3 = df2.shape[0]
    assert n_docs3 == n_docs

    return

def t_labeling2(**kargs):

    return 

def test(**kargs): 
    """

    Note
    ----
    1. pickle.load() had trouble within timeit()

    """
    
    ### Reading temporal documents 
    # t_read_doc(**kargs)

    ### W2V Model Training 

    # w_vec = w_vectorize(t_vectorize, **kargs) # apply **kargs to t_vectorize 
    # t = timeit.timeit(w_vec, number=1)
    # div(message='Elapsed time: %f' % t, symbol='%')  # [log] Elapsed time: 275.162895

    # t_negative_sample(simplify_code=False)
    
    ### Similarity 

    # t_similarity(load_seq=False, load_lookuptb=False, load_model=False)  # [memo] seq is large, so load_seq may not work locally! 

    ### clustering 
    # t_cluster()  # [log] this is subsumed by seqCluster module


    ### cohort dependent tests 
    # t_word2vec0()

    ### lookup sympbols 
    # t_lookup()

    ### RESTful medical coding lookups 
    # t_test_restful()
    # t_lookup2()

    ### Applications 

    # deidentification 
    # t_deidentify(**kargs)
    # t_deidentify_batch(**kargs)

    # read+save coding sequences 
    t_labeling(**kargs)

	
    return

if __name__ == "__main__": 
    test()


