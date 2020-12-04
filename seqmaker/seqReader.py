import pandas as pd
from pandas import DataFrame
import numpy as np
import os, gc, sys, random, collections
from os import getenv 
import time, re

# local modules 
from batchpheno import icd9utils
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed
import seqparams # global control parameters for sequence creation and analysis
import seqTransform

# word2vec modules
import gensim

########################################################################################
#
#  Dependency 
#  ---------- 
#    seqReader -> {analyzer, vector} -> seqAnalyzer (e.g. seqAnalzyer depends on vector module)
# 
#
#  Usage Note
#  ----------
#  The module vector subsumes analyzer in creating feature vectors, word & document vectors
#
#   
#  Related Modules 
#  ---------------
#  a. tdoc <--- seqReader
#  b. seqAnalyzer <--- seqReader
#         
#
########################################################################################

class CodeSeq(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.doctype = '.dat'  # document extension
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            # [filter]
            if fname.find(self.doctype) <= 0: continue
            
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 
def make_human_readable(**kargs):
    return preprocess(**kargs)  # [I/O] DataIn instead of DataExpRoot 
def preprocess(**kargs):  
    """
    Preprocess input medical time series into a canonical form 
    recogized by a target libray (e.g. gensim): e.g. one-visit-one-line 
    format as mentioned in Memo [1] below. 
    

    Memo
    ----
    1. format
       72702 ; 58980010817 55513053010 ; 62934 68115015090 55289062730 62439 ;
       => 72702 
          58980010817 55513053010
          62934 68115015090 55289062730 62439

    2. noise 
       1. non-coded segment, extra white spaces

          67253014315 to whom it may concern bla  454

    """
    def iterdocs(): 
        for fpath in docfiles: 
            assert os.path.exists(fpath) and os.path.getsize(fpath) > 0, "Invalid input: %s" % fpath
            lines = []
            with open(fpath, 'r') as fp: # read .dat file content 
                for line in fp:   # each line ends with '\n'
                	yield line
    def get_sources():
        docfiles = seqparams.TDoc.getPaths(cohort=cohort_name, basedir=basedir, doctype='timed', 
            ifiles=kargs.get('ifiles', []), verify_=True, ext='dat')  # if ifiles is given, then cohort is ignored
        # existence test 
        docfiles = [docfile for docfile in docfiles if os.path.exists(docfile)]
        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, basedir)

        # condition: each docfile is guaranteed to be a full path to the source file (not just a file name)
        return docfiles  

    # [params]
    seq_ptype = kargs.get('seq_ptype', 'regular')

    doctype = '.dat'
    docs_default = ['condition_drug_seq.dat', ]  # if multiple condition_drug_seq-1.dat, condition_drug_seq-2.dat
    basedir = kargs.get('prefix', sys_config.read('DataExpRoot')) # doc data dir
	
    # [params] input
    docfiles = get_sources()

    token_sep = ','
    visit_sep = ';'
    token_end_history = '$' # end token of a patient record
    doc_sep = patient_sep = '\n'

    # [params] transformation 
    simplify_code = kargs.get('simplify_code', False) or kargs.get('base_only', False)

    # output
    ofile = kargs.get('ofile', 'condition_drug_seq.dat') # 'condition_drug.seq'
    output_dir = kargs.get('output_dir', sys_config.read('DataIn'))
    assert os.path.exists(output_dir), "Output dir does not exist: %s" % output_dir
    opath = os.path.join(output_dir, ofile)
    
    # todo: use coroutine
    try: 
        fp = open(opath, 'w')

        pdoc = ''
        for j, pseq in enumerate(iterdocs()):  # pseq: patient sequence (combining all visits)
            pseq = pseq.strip() # strip off '\n'
            vseqx = pseq.split(visit_sep) 
            n_visits = len(vseqx)

            vseq_front, vseq_last = vseqx[:-1], vseqx[-1]
            for i, vseq in enumerate(vseq_front):  # vseq: visit sequence (i.e. a sequence of codes within a single visit)
                # if i == n_visits-1: print('+ last vseq => %s' % vseq)
                
                vseq = transform(vseq.strip(), simplify_code=simplify_code, split=False, token_sep=token_sep)
                vseq = transform_by_ptype(vseq, seq_ptype=seq_ptype)
                
                pdoc += vseq + '\n'
            # add the last visit 
            # assert vseq_last.find(token_end_history) > 0
            assert vseq_last[-1] == token_end_history, "diagnosis> last: %s ~? %s" % (vseqx[-1], vseq_last)
            
            vseq_last0 = vseq_last = vseq_last.strip()
            vseq_last = vseq_last[:-1]  # last token == token_end_history 

            vseq_last = transform(vseq_last, simplify_code=simplify_code, split=False, token_sep=token_sep)
            vseq_last = transform_by_ptype(vseq_last, seq_ptype=seq_ptype)
            # if j % 100 == 0: print(' + last visit (n_visits=%d): %s => %s' % (n_visits, vseq_last0, vseq_last))
            
            pdoc += vseq_last + '\n' 

            # pad an extra newline at EODoc
            pdoc += '\n'

        div(message='Writing medical time series doc to:\n%s\n' % opath, symbol='#')
        fp.write(pdoc)
    finally: 
    	fp.close()      

    return 

def simplify(seq, sep=' '): 
    """
    Simplify diagnostic codes. 

    No-op for non-diagnostic codes. 

    Input
    -----
    seq: a string of space-separated diagnostic codes (icd9, icd10, etc.)


    """
    return icd9utils.getRootSequence(seq, sep=sep)  # sep is only relevant when seq is a string

def separate_time(seq, delimit='|', verify_=False, no_throw=False): 
    """

    Memo
    ----
    1. when the 'if e' conditional statement weren't added, an error showed up (upon execusting seqAnalyzer)
       that indicated the possibility of e being empty but e has ever been detected as being empty ever since
       the condtiional statement is added in. 
    """
    times, codes = [], []
    N = len(seq)
    n_empty = 0  
    for i, e in enumerate(seq): 
        if e:    # memo [1]
            try: 
                t, c = e.split(delimit)  # assumeing that the format is strictly followed (timestamp|code, )
            except: 
                msg = "time> Error: could not separate time for e='%s' ..." % e
                if no_throw: 
                    print msg
                    continue # ignore this pair 
                else: 
                    raise ValueError, msg 
            times.append(t)
            codes.append(c)
        else: 
            n_empty += 1 
            before = seq[i-1] if i >= 1 else 'n/a'
            after = seq[i+1] if i < N-1 else 'n/a'
            print('warning> found %d emtpy string element within seq: (%s) -> ((%s)) -> (%s)' % (n_empty, before, seq[i], after))
            

    return (codes, times)

def transform_by_ptype2(seq, tseq, **kargs):  
    seq_ptype = kargs.get('seq_ptype', 'regular')

    if seq_ptype == 'regular': 
        return (seq, tseq) # no op

    op_split = kargs.get('split', False)
    token_sep = kargs.get('token_sep', ',')
    
    if isinstance(seq, str): 
        seq = seq.split(token_sep)
    if isinstance(tseq, str): 
        tseq = tseq.split(token_sep)

    assert len(seq) == len(tseq)

    if seq_ptype.startswith('overlap'):
        # return transform_by_ngram_overlap(seq, **kargs)
        raise NotImplementedError

    if seq_ptype == 'random': 
        indices = range(len(seq))
        random.shuffle(indices)
        # seq = [seq[i] for i in indices]
        # tseq = [tseq[i] for i in indices]
        return ([seq[i] for i in indices], [tseq[i] for i in indices])

    elif seq_ptype == 'diag': 
        # seq = [token for token in seq if pmed.isCondition(token)]

        seq2, tseq2 = [], []
        for i, tok in enumerate(seq): 
            if pmed.isCondition(tok): 
                seq2.append(tok)
                tseq2.append(tseq[i])
        return (seq2, tseq2)

    elif seq_ptype == 'med': 
        seq = [token for token in seq if not pmed.isCondition(token) and pmed.isMed(token)]
        
        seq2, tseq2 = [], []
        for i, tok in enumerate(seq): 
            if not pmed.isCondition(token) and pmed.isMed(token): 
                seq2.append(tok)
                tseq2.append(tseq[i])
        return (seq2, tseq2)   
    elif seq_ptype == 'lab': 
        # return transform_by_ptype_diabetes(seq, **kargs)
        raise NotImplementedError, "lab sequence mode is coming soon!"
        
    raise NotImplementedError, "unrecognized seq_ptype: %s" % seq_ptype
  
def transformBySeqContentType(seq, **kargs):
    return transform_by_ptype(seq, **kargs)
def transform_by_ptype(seq, **kargs): # ptype: (sequence) pattern type
    return seqTransform.transform_by_ptype(seq, **kargs)

def transform_by_ngram_overlap(seq, **kargs):
    """
    
    Params
    ------
    flat: False by default, default

    Example
    --------
    S'pose seq = ['a', 'b', 'c', 'd']
        n=2 => [('a', 'b'), ('b', 'c'), ('c', 'd')]
        n=3 => [('a', 'b', 'c'), ('b', 'c', 'd')]
    
    """
    return seqTransform.transform_by_ngram_overlap(seq, **kargs)

def transform(seq, **kargs):
    """
    Transform the input string (a sequence of medical codes [and possibly other tokens])
    to another representation (e.g. simplified by taking base part of diagnostic codes) easier 
    to derive distributed vector representation. 

    Input
    -----
    seq: a string of medical code sequence

    """
    op_simplify = kargs.get('simplify_code', False) or kargs.get('base_only', False)
    op_split = kargs.get('split', False)
    token_sep = kargs.get('token_sep', ',')

    if op_simplify: 
        seq = simplify(seq, sep=token_sep)

    if op_split: 
        if isinstance(seq, str): 
            seq = seq.split(token_sep)

    return seq

def readAugmentedDocFromCSV(**kargs):
    def get_sources(): # [params] (cohort_name) (inputdir) where to get the document from? 
        ifiles = kargs.get('ifiles', [])
        if not ifiles: assert cohort_name is not None

        # [note] basedir <- sys_config.read('DataExpRoot') by default
        # [note] doctype has to be 'timed' because timestamps are expected to be in the .csv file
        # meta = kargs.get('meta', 'augmented') # e.g. augmented source, condition_drug_timed_seq-augmented-CKD.csv
        inputdir = docSrcDir
        if not ifiles: # cohort-specific
            for doctype in ['timed',  ]:  # the more complete (more meta data), the higher the precedence
                docfiles = seqparams.TDoc.getAugmentedDocPaths(cohort=cohort_name, prefix=docSrcDir, doctype=doctype, 
                    ifiles=[], 
                    meta=kargs.get('meta', None), 
                    ext='csv', verify_=True)  # if ifiles is given, then cohort is ignored
                
                # existence test 
                docfiles = [docfile for docfile in docfiles if os.path.exists(docfile)]
                if len(docfiles) > 0: break
        else: # doctpye specific (or ifiles are given)
            # [use] 
            # 1. ifiles <- a list of file names (of the source documents) then use basedir to figure out full paths
            # 2. ifiles <- full paths to the document sources (including prefixes)

            # note that this is not seq_ptype-dependent, as this is the whole docuemnt
            # if ifiles is given, then {cohort, doctype} are ignored
            docfiles = seqparams.TDoc.getPaths(cohort=cohort_name, basedir=docSrcDir, ifiles=ifiles, 
                meta=kargs.get('meta', None), 
                ext='csv', verify_=True) 
            # existence test 
            docfiles = [docfile for docfile in docfiles if os.path.exists(docfile)]

        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, docSrcDir)
        return docfiles   
    def parse_row(df, col='sequence', sep=','): # [note] assuming that visit separators ';' were removed
        seqx = [] # a list of (lists of tokens)
        
        isStr = False
        for i, row in enumerate(df[col].values):
            if isinstance(row, str): 
                isStr = True
            if i >= 3: break
            
        if isStr:  # sequence, timestamp
            for row in df[col].values:   # [note] label is not string
                tokens = row.split(sep)
                seqx.append(tokens)
        else: 
            # integer, etc. 
            seqx = list(df[col].values)

        return seqx
    def read_stats(): 
        if tStratify: 
            for l, entry in sdict.items(): 
                print('  + label: %s' % l)
                for h, count in entry.items(): 
                    print('     + header: %s => size: %d' % (h, count))
        else: 
            for h, count in sdict.items(): 
                print('  + header: %s => size: %d' % (h, count))
        return
    from seqparams import TDoc  # or use derived class defined in tdoc

    header = kargs.get('header', ['person_id', 'sequence', 'timestamp',  ])  # may also have 'person_id'
    listOfTokensFormat = seqparams.TDoc.fListOfTokens # ['sequence', 'timestamp', ]

    cohort_name = kargs.get('cohort', None)  # needed if ifiles is not provided
    docSrcDir = kargs.get('inputdir', seqparams.getCohortGlobalDir(cohort_name))  # sys_config.read('DataExpRoot') # document in data-in shows one-visit per line

    # if 'complete' is True => try the most information-rich format first i.e. in the order of doctype='labeled', 'timed', 'doc', 'visit'
    fpaths = get_sources() # [params] cohort, basedir
    print('read> reading from %d (augmented) source files:\n%s\n' % (len(fpaths), fpaths))

    tStratify = False # separate each data subset according to their labels
    # L = [1, ]  # unique labels
    if not tStratify:
        # [output]
        ret = {h: [] for h in header}
        
        # [policy] if there are multiple sources, their contents will be consolidated
        sdict = {h: 0 for h in header}
        for fpath in fpaths: 
            df_seq = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            if not df_seq.empty: 
                for h in header: 
                    if h in df_seq.columns: 
                        # or use seqparams.TDoc.strToSeq(df_seq, col='sequence')
                        # multilabel format e.g. '250.00,253.01' => ['250.00', '253.01']
                        print('read> processing header %s ...' % h)
                        seqx = parse_row(df_seq, col=h, sep=',')
                        ret[h].extend(seqx)  # assuming that ';', '$' are all being replaced
                        sdict[h] += len(seqx)
    else: 
        raise ValueError, 'readAugmentedDocFromCSV> Augmented data by default have no labels.'

    # read_stats()
    assert len(ret) > 0, "No data found using the given attributes: %s (columns expected: %s)" % (header, df_seq.columns.values)
    return ret # keys <- header 

def loadDocFromCSV(**karsg):
    return readDocFromCSV(**kargs) 
def readDocFromCSV(**kargs): 
    """
    Read coding sequences from .csv files 

    Input
    -----
    cohort: name of the cohort (e.g. PTSD) ... must be provided; no default
    ifiles: sources of coding sequences 
            if ifiles are given, the cohort is igonored 
    (o) basedir: directory from which sources are stored (used when ifiles do not include 'prefixes')

    meta: user-defined file ID

    Memo
    ----
    1. relabeling 
         - may want to merge certain labels together to form a single label 
           e.g. In CKD cohort, ESRD-related labels can be considered as part of stage 5 CKD
                merging different types of control data

            lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
            lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]

    """
    def get_sources(complete=True): # [params] (cohort_name) (inputdir) where to get the document from? 
        ifiles = kargs.get('ifiles', [])
        if not ifiles: assert cohort_name is not None, "Cannot resolve source document(s) when ifiles and cohort are both set to None."
        print('readDocFromCSV> input files: %s' % ifiles)
        # [note] basedir <- sys_config.read('DataExpRoot') by default
        # [note] doctype has to be 'timed' because timestamps are expected to be in the .csv file
        meta = kargs.get('meta', None) # e.g. user-defined file ID encoding info on augmented source, etc. e.g. condition_drug_timed_seq-augmented-CKD.csv
    
        hasFullPath = all([os.path.exists(ifile) for ifile in ifiles]) if len(ifiles) > 0 else False

        if hasFullPath: 
            docfiles = ifiles 
        else: 
            if complete and not ifiles: # cohort-specific
                for doctype in ['labeled', 'timed', 'doc', 'visit', ]:  # the more complete (more meta data), the higher the precedence
                    docfiles = seqparams.TDoc.getPaths(cohort=cohort_name, basedir=docSrcDir, doctype=doctype, 
                        meta=meta, ifiles=[], ext='csv', verify_=True)  # if ifiles is given, then cohort is ignored
                
                    # existence test 
                    docfiles = [docfile for docfile in docfiles if os.path.exists(docfile)]
                    if len(docfiles) > 0: break
            else: # doctpye specific (or ifiles are given)
                # [use] 
                # 1. ifiles <- a list of file names (of the source documents) then use basedir to figure out full paths
                # 2. ifiles <- full paths to the document sources (including prefixes)

                # note that this is not seq_ptype-dependent, as this is the whole docuemnt
                docfiles = seqparams.TDoc.getPaths(cohort=cohort_name, basedir=docSrcDir, doctype=kargs.get('doctype', 'timed'), 
                    meta=meta, ifiles=ifiles, ext='csv', verify_=True)  # if ifiles is given, then cohort is ignored
                # existence test 
                docfiles = [docfile for docfile in docfiles if os.path.exists(docfile)]

        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, docSrcDir)
        return docfiles   
    def parse_row(df, col='sequence', sep=','): # [note] assuming that visit separators ';' were removed
        seqx = [] # a list of (lists of tokens)
        
        isStr = False
        for i, row in enumerate(df[col].values):
            if isinstance(row, str): 
                isStr = True
            if i >= 3: break
            
        if isStr:  # sequence, timestamp
            for row in df[col].values:   # [note] label is not string
                tokens = row.split(sep)
                seqx.append(tokens)
        else: 
            # integer such as person_ids
            seqx = list(df[col].values)

        return seqx
    def read_stats(): 
        if tStratify: 
            for l, entry in sdict.items(): 
                print('  + label: %s' % l)
                for h, count in entry.items(): 
                    print('     + header: %s => size: %d' % (h, count))
        else: 
            for h, count in sdict.items(): 
                print('  + header: %s => size: %d' % (h, count))
        return
    def relabel(df, col_target): # L
        lmap = kargs.get('label_map', {})
        if not lmap: return # noop 
        assert col_target in df.columns, "label attribute %s is not in the header:\n%s\n" % (col_target, df.columns.values) 
        labelSet = df[col_target].unique()
        print('  + (before) unique labels (n=%d):\n%s\n' % (len(labelSet), labelSet))
        for label, eq_labels in lmap.items(): 
            print('  + %s <- %s' % (label, eq_labels))
            cond_pos = df[col_target].isin(eq_labels)
            df.loc[cond_pos, col_target] = label  # rename to designated label 
        labelSet = df[col_target].unique()
        print('  + (after) unique labels (n=%d):\n%s\n' % (len(labelSet), labelSet))
        return # input dataframe (df) relabeled

    from seqparams import TDoc  # or use derived class defined in tdoc

    header = kargs.get('header', ['person_id', 'sequence', 'timestamp', 'label', ])
    listOfTokensFormat = seqparams.TDoc.fListOfTokens # ['sequence', 'timestamp', ]

    cohort_name = kargs.get('cohort', None)  # needed if ifiles is not provided
    basedir = docSrcDir = kargs.get('inputdir', TDoc.prefix)  # sys_config.read('DataExpRoot') # document in data-in shows one-visit per line

    # if 'complete' is True => try the most information-rich format first i.e. in the order of doctype='labeled', 'timed', 'doc', 'visit'
    fpaths = get_sources(complete=kargs.get('complete', True)) # [params] cohort, basedir
    print('read> reading from %d source files:\n%s\n' % (len(fpaths), fpaths))

    tStratify = kargs.get('stratify', False) # separate each data subset according to their labels
    L = [1, ]  # unique labels
    if tStratify: # stratify the data by labels
        f_label = kargs.get('label_name', 'label') # name of the target label attribute e.g. label, label_lcs
        
        # [output]
        ret = {}
        # [policy] if there are multiple sources, their contents will be consolidated
        sdict = {}        
        for fpath in fpaths: 
            df_seq = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)

            if df_seq.empty: continue
            assert f_label in df_seq.columns, "label attribute %s is NOT found in:\n%s\n" % (f_label, fpath)  # [note] label_lcs for LCS labels
            relabel(df_seq, f_label)  # operational only if label_map is given

            L = df_seq[f_label].unique()  # f_label: which label? {label, label_lcs}
            print('  + found %d unique labels (example: %s)' % (len(L), L[:10]))

            for l in L:  # foreach label, create a return data unit 
                ret[l] = {h: [] for h in header}
                sdict[l] = {h: 0 for h in header}
                dfi = df_seq.loc[df_seq[f_label]==l] # ts[f_label].isin([l, ])
                for h in header: 
                    if h in df_seq.columns: 
                        # or use seqparams.TDoc.strToSeq(df_seq, col='sequence')
                        # multilabel format e.g. '250.00,253.01' => ['250.00', '253.01']
                        # print('read> processing header %s ...' % h)
                        seqx = parse_row(dfi, col=h, sep=',')
                        ret[l][h].extend(seqx)  # assuming that ';', '$' are all being replaced
                        sdict[l][h] += len(seqx)
    else: 
        # [output]
        ret = {h: [] for h in header}
        
        # [policy] if there are multiple sources, their contents will be consolidated
        sdict = {h: 0 for h in header}
        for fpath in fpaths: 
            df_seq = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            if not df_seq.empty: 
                for h in header: 
                    if h in df_seq.columns: 
                        # or use seqparams.TDoc.strToSeq(df_seq, col='sequence')
                        # multilabel format e.g. '250.00,253.01' => ['250.00', '253.01']
                        print('read> processing header %s ...' % h)
                        seqx = parse_row(df_seq, col=h, sep=',')
                        ret[h].extend(seqx)  # assuming that ';', '$' are all being replaced
                        sdict[h] += len(seqx)

    # read_stats()
    assert len(ret) > 0, "No data found using the given attributes: %s (columns expected: %s)" % (header, df_seq.columns.values)
    return ret # keys <- header

def reconstructDoc(**kargs): 
    """
    Read the coding sequences (and their respective timestamps) from .csv source files and 
    reconstruct the coding sequences by recovering appripriate delimiters such as separators for visits (';')
    and end-document token ('$')

    .csv -> .dat format

    """
    return 

# [obsolete] subsummed by readDocToCSV()
def readTimedDocToCSV(**kargs): 
    """
    Read medical coding sequences from the source (e.g. condition_drug_timed_seq-PTSD.dat) and 
    save the resulting set of sequences to a .csv file.
    
    Each patient corresponds to a single document. 

    header: sequence, timestamp 

    Note: Use the new version readDocToCSV()

    """
    def shift_ext(ext='.pkl'):
        ifiles_prime = ifiles[:]  # don't want to replace ifiles inplace (may need to passed to seqReader later)
        for i, f in enumerate(ifiles_prime): 
            base_, ext_ = os.path.splitext(f) 
            if ext_ != ext: 
                ifiles_prime[i] = base_ + ext  

        # 'ifiles' refers to the same ifiles in the outer function 
        return ifiles_prime
    def seq_to_str(seqx, sep=','): 
        return [sep.join(str(s) for s in seq) for seq in seqx] 

    print("info> Reading from source (.dat file e.g. condition_drug_timed_seq-PTSD.dat) ...")
    documents, timestamps = readTimedDocPerPatient(**kargs)

    basedir = kargs.get('data_dir', sys_config.read('DataExpRoot')) 
    cohort_name = kargs.get('cohort', 'diabetes') # 'diabetes'

    # [params] file, sequence type
    seq_compo = seq_composition = kargs.get('seq_compo', 'condition_drug')
    tdoc_prefix = seq_compo
    seq_ptype = seqparams.normalize_ctype(**kargs)
    # time_sep = '|'  # time separator (from its corresponding codes)

    # [params] time series document settings
    save_seq = True  # [I/O]

    # [params] default document file
    #          default cohort if None is diabetes; other cohorts e.g. PDSD cohort: condition_drug_timed_seq-PTSD.dat
    ftype = 'csv'
    f_tdoc = kargs.get('outputfile', None)
    identifier = seq_ptype
    if f_tdoc is None: 
        # f_tdoc = '%s_timed_seq.%s' % (tdoc_prefix, ftype) if cohort_name is None else '%s_timed_seq-%s.%s' % (tdoc_prefix, cohort_name, ftype) 
        if cohort_name is None: 
            f_tdoc = '%s_timed_seq-%s.%s' % (tdoc_prefix, identifier, ftype) 
        else: 
            f_tdoc = '%s_timed_seq-%s-%s.%s' % (tdoc_prefix, cohort_name, identifier, ftype) 
    else: 
        assert f_tdoc.find(ftype) > 0, "Invalid output file name (file type: %s): %s" % (ftype, f_tdoc) 
    fpath = os.path.join(basedir, f_tdoc)  # [I/O]
    
    if save_seq: # this is big for diabetes cohort! ~1.2G 
        # double check 
        assert len(documents) > 0 and len(documents) == len(timestamps)
        assert isinstance(documents[0], list) and isinstance(timestamps[0], list)
        n_docs = len(documents)
        
        if not os.path.exists(fpath):
            print('info> creating new (timed) coding sequence data of size: %d' % n_docs) 

        # pickle.dump(sequences, open(fpath, "wb" ))
        header = ['sequence', 'timestamp', ]
        adict = {} # {h: [] for h in header}
        adict['sequence'] = seq_to_str(documents)
        adict['timestamp'] = seq_to_str(timestamps)
        
        print('status> number of sequences: %d =?= %d' % (len(adict['sequence']), len(adict['timestamp'])))

        # seq_to_str(sequences)
        df = DataFrame(adict, columns=header)
        df.to_csv(fpath, sep='|', index=False, header=True)
        div(message='io> saved timed coding sequences (size: %d) to %s' % (n_docs, fpath), symbol='#')

    return df
def readDocToCSV(**kargs): 
    """
    Read medical coding sequences from the source (e.g. condition_drug_seq-PTSD.dat) and 
    save the resulting set of sequences to csv file. 

    Each patient corresponds to a single document. 


    Input 
    ----- 
    cohort
    seq_ptype

    sequences (D)
    timestamps (T)
    labels (L)
    
    optional: 
        outputdir
        ifiles
    

    Output 
    ------
    dataframe containing following columns 
        
        ['sequence', 'timestamp', 'label', ]

        where 'timestamp' and 'lable' are only included when available


    Memo
    ----
    1. headers of source files used to make sequence documents
            header_condition = ['person_id', 'condition_start_date', 'condition_source_value', ]
            header_drug = ['person_id', 'drug_exposure_start_date', 'drug_source_value', ]
            header_lab = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']

    """
    def seq_to_str(seqx, sep=','): 
        return [sep.join(str(s) for s in seq) for seq in seqx] 

    def str_to_seq(df, col='sequence', sep=','):
        seqx = []
        for seqstr in df[col].values: 
            s = seqstr.split(sep)
            seqx.append(s)
        return seqx
    def get_cohort(): 
        try: 
            return kargs['cohort']
        except: 
            pass 
        raise ValueError, "cohort info is mandatory."

    from tdoc import TDoc  # or from seqparams import TDoc

    # first, read the docs from source files (.dat)
    sequences, timestamps = kargs.get('sequences', []), kargs.get('timestamps', [])
    if len(sequences) == 0 or len(timestamps) == 0: 
        sequences, timestamps = readTimedDocPerPatient(**kargs) # reads documents from appropriate sources

    doctype = kargs.get('doctype', 'doc')  # 'doc': one-doc-per-patient format | 'visit':  one-doc-per-visit format; not in use
    if len(timestamps) > 0: 
        doctype = 'timed'
        assert len(sequences) == len(timestamps)

    labels = kargs.get('labels', []) 
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
    # fname = seqparams.TDoc.getName(cohort=kargs['cohort'], doctype=doctype, ext='csv') 
    seq_ptype = kargs.get('seq_ptype', 'regular') # if regular, then doc_basename is set to TDoc's default: condition_drug

    # A. fpath convention for source MCS file
    user_file_descriptor = kargs.get('meta', None)
    fpath = TDoc.getPath(cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir,
                meta=user_file_descriptor)  # get the path to the new document

    # B. fpath convention for derived MCS file || training set (see TSet.saveDoc(), TDoc.saveDoc())

    df = DataFrame() # dummy 
    if len(sequences) > 0: # this is big! ~1.2G 
        
        # pickle.dump(sequences, open(fpath, "wb" ))
        if doctype in ['timed', 'labeled',]:
            header = ['sequence', 'timestamp', ]   # ['sequence', 'timestamp', 'label', ]
            adict = {} # {h: [] for h in header}
            adict['sequence'] = seq_to_str(sequences)   # assuming visit-seperator ';' is lost 
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

        div(message='readDocToCSV> saved sequences (size: %d, doctype: %s, dim: %s) to:\n%s\n' % \
            (len(sequences), doctype, str(df.shape), fpath), symbol='#')

    return df

def readDocToDF(D, T=[], L=[], **kargs): 
    """
    Similar to readDocToCSV() but with simplier design, assuming that (D, T) is known 
    and outputs only the dataframe (df) without saving it. 
    
    """
    def seq_to_str(seqx, sep=','): 
        return [sep.join(str(s) for s in seq) for seq in seqx] 

    def str_to_seq(df, col='sequence', sep=','):
        seqx = []
        for seqstr in df[col].values: 
            s = seqstr.split(sep)
            seqx.append(s)
        return seqx

    from tdoc import TDoc  # or from seqparams import TDoc

    # first, read the docs from source files (.dat)
    sequences, timestamps, labels = D, T, L
    doctype = 'doc'
    if len(timestamps) > 0: 
        doctype = 'timed'
        assert len(sequences) == len(timestamps)

    labels = kargs.get('labels', []) 
    if len(labels) > 0: 
        doctype = 'labeled'
        # the nubmer of labels is not always identicial to number of documents (because a subset of patients may not have data)
        assert len(labels) == len(sequences), "n_labels=%d <> n_docs=%d" % (len(labels), len(sequences))
        
    print('info> n_doc: %d, doctype: %s, n_labels: %d' % (len(sequences), doctype, len(labels)))

    df = DataFrame() # dummy 
    if len(sequences) > 0: # this is big! ~1.2G 
        
        # pickle.dump(sequences, open(fpath, "wb" ))
        if doctype in ['timed', 'labeled',]:
            header = ['sequence', 'timestamp', ]   # ['sequence', 'timestamp', 'label', ]
            adict = {} # {h: [] for h in header}
            adict['sequence'] = seq_to_str(sequences)   # assuming visit-seperator ';' is lost 
            adict['timestamp'] = seq_to_str(timestamps)
            
            if doctype == 'labeled':  # qualified earlier
                header = header + ['label', ]
                adict['label'] = labels

            df = DataFrame(adict, columns=header)
            # df.to_csv(fpath, sep='|', index=False, header=True)
        else: # doctype <- 'doc', 'visit'
            header = ['sequence', ]   # seqMaker2 now also can be configued to produce this (but without seq_ptype as part of the file ID)
            # seq_to_str(sequences)
            df = DataFrame(seq_to_str(sequences), columns=header)
            # df.to_csv(fpath, sep='|', index=False, header=True)

        div(message='readDocToDF> Summary: sequences (size: %d, doctype: %s, dim: %s)' % \
            (len(sequences), doctype, str(df.shape)), symbol='#')

    return df

def verifyTimedSeqFile(cohort, seq_ptype='regular', ext='csv', basedir=None):
    """

    Memo
    ----
    1. example cohort=CKD 
       condition_drug_timed_seq-CKD.csv

    2. time_seq file should have been created in seqMaker2 

    """
    pass 
def verifyLabeledSeqFile(corhot, seq_ptype='regular', ext='csv', **kargs): # [params] cohort, doctype, seq_ptype, doc_basename, ext + basedir
    """

    Memo
    ----
    1. differences in timed_seq and labeled_seq 
       timed_seq is not deidentified because we need person_id to later match label data 
       labeled_seq is deidentified
    
    2. running this routine is to ensure the existence of labeled_seq file even if 
       there exist no annotated data
    """
    from tdoc import TDoc

    # keys: 'sequence', 'timestamp', 'label' 
    ret = readDocFromCSV(cohort=cohort, ifiles=kargs.get('ifiles', []), complete=True) # [params] doctype (timed), inputdir
    D, T, L = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])

    fpath = TDoc.getPath(cohort=cohort, seq_ptype=seq_ptype, doctype='labeled', ext=ext, basedir=kargs.get('basedir', None))  # usually there is one file per cohort  
    if not os.path.exists(fpath): 
        # path params: cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir
        if not L: 
            # create dummy labels [todo]
            L = [1] * len(D)
        assert len(T) > 0, "timestamps were not found in this source (cohort=%s)" % cohort
        assert len(D) == len(T), "inconsistent sizes between sequences and timestamps"

        header = ['sequence', 'timestamp', 'label', ]
        adict = {h: [] for h in header}
        adict['sequence'], adict['timestamp'], adict['label'] = D, T, L
        df.to_csv(fpath, sep='|', index=False, header=True)
        print('transformDocuments2> Saving the transformed document (seq_ptype=%s) to:\n%s\n' % (seq_ptype, fpath))
        # sr.readDocToCSV(cohort=get_cohort(), sequences=D, timestamps=T, labels=L, seq_ptype=seq_ptype)
    return (D, T, L)  # condition: all have data

def readTimedDocPerVisit(**kargs): 
    msg = "Probably not useful because people usually do not have sufficiently long sequences in their medical records :)"
    raise NotImplementedError, msg
    
def readTimedDocPerPatient(**kargs): 
    """
    Same as readDoc() but takes in the doc with timestamps as the input

    Input
    -----
    cohort: cohort of interests 
    ifile: single inputs
    ifiles: multiple inputs

    inputdir: directory from which documents are read

    #1 supply cohort => find default cohort-specific source(s) of document type (doctype): 'timed'
    #2 supply ifiles => 

    """
    def iterdocs(): 
        for fpath in docfiles: 
            assert os.path.exists(fpath) and os.path.getsize(fpath) > 0, "Invalid input: %s" % fpath
            # lines = []
            print('input> opening source document file at %s' % fpath)
            with open(fpath, 'r') as fp: # read .dat file content 
                for line in fp:   # assuming that each line ends with '\n' and each line corresponds to a single patient's doc
                    yield line  # line includes '\n'

    def summary(): 
        print('> reading from %s input files under:\n%s\n' % (len(docfiles), basedir))
        print('> number of patients/documents: %d' % n_doc)

        return
    def get_sources():
        
        # check if a full path is given
        # kargs.get('input_path', None)

        docfiles = seqparams.TDoc.getPaths(cohort=cohort_name, basedir=basedir, doctype='timed', 
            ifiles=kargs.get('ifiles', []), 
            meta=kargs.get('meta', None),  # user-defined file ID 
            verify_=True, ext='dat')  # if ifiles is given, then cohort is ignored
        # existence test 
        docfiles = [docfile for docfile in docfiles if os.path.exists(docfile)]
        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, basedir)

        # condition: each docfile is guaranteed to be a full path to the source file (not just a file name)
        return docfiles   
        
    # [params]
    test_, verify_ = False, True
    use_preprocessed = False 

    seq_compo = sequence_composition = kargs.get('seq_compo', 'condition_drug') # condition_drug would consist of diagnoses and medications
    cohort_name = kargs.get('cohort', None)  # default diabetes if None

    seq_ptype = kargs.get('seq_ptype', 'regular')  # [options] 'med', 'diag', 'mixed'/'regular'

    # default basedir
    default_basedir = sys_config.read('DataIn') if use_preprocessed else sys_config.read('DataExpRoot') # doc data dir
    basedir = kargs.pop('inputdir', default_basedir)

    verify_doc = kargs.get('verify_', True)

    # [params] input
    docfiles = get_sources() # this specifies (coding sequence) source files 
    print('io> reading sequences from %d documents' % len(docfiles))

    # [params] control 
    no_throw = kargs.get('no_throw', True)
    
    # [params] test
    iter_max_doc, iter_max_visit = kargs.get('max_doc', np.inf), kargs.get('max_visit', np.inf)

    # [params] doc (=> made global)
    # [note] ':' is reserved for prefixed strings like MED:1234, I9:309.81, etc. 
    token_sep = ','
    visit_sep = ';'
    token_end_history =  '$'
    doc_sep = patient_sep = '\n'
    time_sep = '|'

    simplify_code = kargs.get('simplify_code', False) or kargs.get('base_only', False)
    # div(message='Simplify code? %s' % simplify_code, symbol='%')

    documents = []  # map from patient to their invidiual documents
    timestamps = []  # per-patient timestamps (i.e. the prescription times of their corresponding codes)
    tokens = set() # unique tokens

    n_doc = len(list(iterdocs()))
    print('readTimedDocPerPatient> found %d documents' % n_doc)
    vseq_last = None
    n_error_visit = n_error_separate_time = 0
    for j, pseq in enumerate(iterdocs()):  # pseq: patient sequence (combining all visits)
        if j >= iter_max_doc: break 

        # [note] pseq[-1] is newline
        sequences, time_sequences = [], [] # collect sequences (made out of visits)
        pseq = pseq.strip() # strip off '\n' 
        vseqx = pseq.split(visit_sep) 
        n_visits = len(vseqx)

        vseq_front, vseq_last = vseqx[:-1], vseqx[-1]
        for i, vseq in enumerate(vseq_front):  # vseq: visit sequence (i.e. a sequence of codes within a single visit)
            # if i >= iter_max_visit: break 

            # [key] operation
            # string to list of codes
            vseq = transform(vseq.strip(), simplify_code=simplify_code, split=True, token_sep=token_sep) # split via token_sep

            # set exception to True to ensure the input carries time info
            # e.g. 2906-09-19|365.11,2906-11-12|250.00, ...
            #      [note] still found empty string element in vseq
            #      [note] time info may not exist; if so, skip that subsequence in the visit by default
            vseq, tseq = separate_time(vseq, delimit=time_sep, no_throw=no_throw) # for now assuming that the format is correct; o.w. .split() will crash
            if not vseq: 
                assert not tseq 
                n_error_separate_time += 1 
                print('  + warning: no valid visit sequence identified in %s, skipping ...' % str(vseq))
                continue # skip this visit            

            # preserve only desired contents (e.g. diagnostic codes only)
            vseq, tseq = transform_by_ptype2(vseq, tseq, seq_ptype=seq_ptype)  # [todo] speed, factor out
            # tseq = transform_timestamp(tseq)

            sequences.extend(vseq)  # combining visit-specific subsequence
            time_sequences.extend(tseq)
            
        # add the last visit 
        vseq_last = vseq_last.strip()  # strip off white spaces   e.g. 125.0,137.0,11155.
        assert vseq_last[-1] == token_end_history, "diagnosis> last: %s ~? %s" % (vseqx[-1], vseq_last)
        vseq_last = vseq_last[:-1]

        has_last = False
        # assert len(vseq_last) > 0 
        if len(vseq_last) == 0: 
            msg = 'error> last element is empty ((%s))' % vseq_last
            if no_throw: 
                print(msg)
            else: 
                raise ValueError, msg
        else: 
            vseq_last = transform(vseq_last, simplify_code=simplify_code, split=True, token_sep=token_sep)
            vseq_last, tseq_last = separate_time(vseq_last, delimit=time_sep, no_throw=no_throw) 
            vseq_last, tseq_last  = transform_by_ptype2(vseq_last, tseq_last, seq_ptype=seq_ptype)
        
            # tseq_last = transform_timestamp(tseq_last)

            if verify_doc and (j <= 2 or (j % 500 == 0) or (j > n_doc-3)): print('verify> vseq_last # %d => %s, %s' % (j, tseq_last, vseq_last))

            sequences.extend(vseq_last)  # last segment/visit of the per-patient sequence
            time_sequences.extend(tseq_last)

        # unique tokens
        tokens.update(sequences) 

        # per-patient doc
        assert len(sequences) == len(time_sequences)
        documents.append(sequences)
        timestamps.append(time_sequences)

    if verify_: 
        n_parsed, n_timestamp = len(documents), len(timestamps)
        assert n_doc == n_parsed == n_timestamp, "n_doc: %d <> n_parsed: %d, n_timestamps: %d" % \
            (n_doc, n_parsed, n_timestamp) # n_doc <- len(list(iterdocs()))
        summary()  # [log] number of patients/documents: 432000

        div(message='Number of documents: %d\nNumber of unique tokens: %d' % (n_doc, len(tokens)), symbol='%')

        # if documents were dictionary: utils.sample_dict(documents, n_sample=10)
        
        print('  + First document:\n%s\n' % documents[0])
        print('  + Last document:\n%s\n' % documents[-1])

        # randomly sampled 
        n_sample = 5
        for i, doc_seq in enumerate(random.sample(documents, n_sample)): 
            print('   >>> Random Doc #%d:\n%s\n' % (i, doc_seq))

        # error report
        print('  + Found %d parsing errors in visits (unable to separate codes from times)' % n_error_separate_time)

    # model = gensim.models.Word2Vec(sequences)
    return (documents, timestamps)

def readTimedDoc(fpath, **kargs):
    """
    Read sequenced EHR (MCSs) with timestamps attached. 
    
    A simplier design for readTimedDocPerPatient()

    """ 
    def iterdocs(): 
        for fp in [fpath, ]: 
            assert os.path.exists(fp) and os.path.getsize(fp) > 0, "Invalid input: %s" % fp
            # lines = []
            print('input> opening source document file at %s' % fp)
            with open(fp, 'r') as D: # read .dat file content 
                for line in D:   # assuming that each line ends with '\n' and each line corresponds to a single patient's doc
                    yield line  # line includes '\n'

    def get_sources():
        
        # check if a full path is given
        assert os.path.exists(fpath), "Invalid input path: %s" % fpath
        inputdir, fname = os.path.dirname(fpath), os.path.basename(fpath)
        fstem, fext = os.path.splitext(fname)

        return (inputdir, fstem, fext)   
        
    # [params]
    test_, verify_ = False, True
    use_preprocessed = False 
    verify_doc = kargs.get('verify_', True)
    seq_ptype = kargs.get('seq_ptype', 'regular')  # [options] 'med', 'diag', 'mixed'/'regular'
    simplify_code = kargs.get('simplify_code', False) or kargs.get('base_only', False)

    # [params] input
    inputdir, file_stem, file_ext = get_sources() # this specifies (coding sequence) source files 

    # [params] control 
    no_throw = kargs.get('no_throw', True)  # what if timestamp wasn't being separated?  
    iter_max_doc, iter_max_visit = kargs.get('max_doc', np.inf), kargs.get('max_visit', np.inf)

    # [params] doc (=> made global)
    # [note] ':' is reserved for prefixed strings like MED:1234, I9:309.81, etc. 
    token_sep = ','
    visit_sep = ';'
    token_end_history =  '$'
    doc_sep = patient_sep = '\n'
    time_sep = '|'

    # div(message='Simplify code? %s' % simplify_code, symbol='%')

    documents = []  # map from patient to their invidiual documents
    timestamps = []  # per-patient timestamps (i.e. the prescription times of their corresponding codes)
    tokens = set() # unique tokens

    n_doc = len(list(iterdocs()))
    # assert n_doc == 1, "n_doc = %d" % n_doc
    vseq_last = None
    n_error_visit = n_error_separate_time = 0
    for j, pseq in enumerate(iterdocs()):  # pseq: patient sequence (combining all visits)
        if j >= iter_max_doc: break 

        # [note] pseq[-1] is newline
        sequences, time_sequences = [], [] # collect sequences (made out of visits)
        pseq = pseq.strip() # strip off '\n' 
        vseqx = pseq.split(visit_sep) 
        n_visits = len(vseqx)

        vseq_front, vseq_last = vseqx[:-1], vseqx[-1]
        for i, vseq in enumerate(vseq_front):  # vseq: visit sequence (i.e. a sequence of codes within a single visit)
            # if i >= iter_max_visit: break 

            # [key] operation
            # string to list of codes
            vseq = transform(vseq.strip(), simplify_code=simplify_code, split=True, token_sep=token_sep) # split via token_sep

            # set exception to True to ensure the input carries time info
            # e.g. 2906-09-19|365.11,2906-11-12|250.00, ...
            #      [note] still found empty string element in vseq
            #      [note] time info may not exist; if so, skip that subsequence in the visit by default
            vseq, tseq = separate_time(vseq, delimit=time_sep, no_throw=no_throw) # for now assuming that the format is correct; o.w. .split() will crash
            if not vseq: 
                assert not tseq 
                n_error_separate_time += 1 
                print('  + warning: no valid visit sequence identified in %s, skipping ...' % str(vseq))
                continue # skip this visit            

            # preserve only desired contents (e.g. diagnostic codes only)
            vseq, tseq = transform_by_ptype2(vseq, tseq, seq_ptype=seq_ptype)  # [todo] speed, factor out
            # tseq = transform_timestamp(tseq)

            sequences.extend(vseq)  # combining visit-specific subsequence
            time_sequences.extend(tseq)
            
        # add the last visit 
        vseq_last = vseq_last.strip()  # strip off white spaces   e.g. 125.0,137.0,11155.
        assert vseq_last[-1] == token_end_history, "diagnosis> last: %s ~? %s" % (vseqx[-1], vseq_last)
        vseq_last = vseq_last[:-1]

        has_last = False
        # assert len(vseq_last) > 0 
        if len(vseq_last) == 0: 
            msg = 'error> last element is empty ((%s))' % vseq_last
            if no_throw: 
                print(msg)
            else: 
                raise ValueError, msg
        else: 
            vseq_last = transform(vseq_last, simplify_code=simplify_code, split=True, token_sep=token_sep)
            vseq_last, tseq_last = separate_time(vseq_last, delimit=time_sep, no_throw=no_throw) 
            vseq_last, tseq_last  = transform_by_ptype2(vseq_last, tseq_last, seq_ptype=seq_ptype)
        
            # tseq_last = transform_timestamp(tseq_last)

            if verify_doc and (j <= 2 or (j % 500 == 0) or (j > n_doc-3)): print('verify> vseq_last # %d => %s, %s' % (j, tseq_last, vseq_last))

            sequences.extend(vseq_last)  # last segment/visit of the per-patient sequence
            time_sequences.extend(tseq_last)

        # unique tokens
        tokens.update(sequences) 

        # per-patient doc
        assert len(sequences) == len(time_sequences)
        documents.append(sequences)
        timestamps.append(time_sequences)

    if verify_: 
        n_parsed, n_timestamp = len(documents), len(timestamps)
        assert n_doc == n_parsed == n_timestamp, "n_doc: %d <> n_parsed: %d, n_timestamps: %d" % \
            (n_doc, n_parsed, n_timestamp) # n_doc <- len(list(iterdocs()))

        div(message='Number of documents: %d\nNumber of unique tokens: %d' % (n_doc, len(tokens)), symbol='%')

        # if documents were dictionary: utils.sample_dict(documents, n_sample=10)
        
        print('  + First document:\n%s\n' % documents[0])
        print('  + Last document:\n%s\n' % documents[-1])

        # randomly sampled 
        n_sample = 5
        for i, doc_seq in enumerate(random.sample(documents, n_sample)): 
            print('   >>> Random Doc #%d:\n%s\n' % (i, doc_seq))

        # error report
        print('  + Found %d parsing errors in visits (unable to separate codes from times)' % n_error_separate_time)

    # model = gensim.models.Word2Vec(sequences)
    return (documents, timestamps)

def read(**kargs): 
    return readDocPerPatient(**kargs)
def readDoc(**kargs): 
    return readDocPerPatient(**kargs)
def read_doc(**kargs): 
    return readDocPerPatient(**kargs)
def readDocPerPatient(**kargs):  # coding sequence parser
    """
    Read medical coding sequence from .dat source files (created by seqMaker2) and convert 
    it into a set of patient documents in which each patient generally has multiple visits 
    and therefore a document consists of medical coding segments across multiple visists. 
    
    The end result is that each patient has a medical coding sequence that spans across all 
    his or her clinical visists or per-patient document. read() on the other hand generates 
    per-visit docuemnt/sequence. 
    
    Related
    -------
    read(): generate sequences in which each sequence corresponds to a clinical visit 
            (individuality of the patient is lost); that is, per-visit document

    readTimedDoc(): read docs with tagged timestamps
    read_doc(): old naming convention; synonymous to readDocPerPatient(), readDoc()

    seqAnalyzer.readDoc: returns (docuemnts, timestamps) when doc_type set to 'timed'

    Input
    -----
    ifile: single input
    ifiles: multiple inputs
       <code> 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat'
       e.g. condition_drug_seq.dat        //for diabetes cohort
            condition_drug_seq-PTSD.dat   //for PTSD cohort

    Memo
    ----
    1. default doc files: 
       condition_drug_seq.dat 
       condition_drug_timed_seq.dat  (with timestamps attached to codes)


    """
    def iterdocs(): 
        for fpath in docfiles: 
            assert os.path.exists(fpath) and os.path.getsize(fpath) > 0, "Invalid input: %s" % fpath
            # lines = []
            print('input> opening source document file at %s' % fpath)
            with open(fpath, 'r') as fp: # read .dat file content 
                for line in fp:   # assuming that each line ends with '\n' and each line corresponds to a single patient's doc
                    yield line  # line includes '\n'

    def summary(): 
        print('> reading from %s input files under:\n%s\n' % (len(docfiles), basedir))
        print('> number of patients/documents: %d' % n_doc)

        return
    def get_sources():
        docfiles = seqparams.TDoc.getPaths(cohort=cohort_name, basedir=basedir, doctype='doc', 
            ifiles=kargs.get('ifiles', []), verify_=True)  # if ifiles is given, then cohort is ignored
        # existence test 
        docfiles = [docfile for docfile in docfiles if os.path.exists(docfile)]
        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, basedir)
        return docfiles     
   
    # import seqparams
    # [params]
    test_, verify_ = False, True
    use_preprocessed = False 

    seq_compo = sequence_composition = kargs.get('seq_compo', 'condition_drug') # condition_drug would consist of diagnoses and medications
    cohort_name = kargs.get('cohort', None)  # default diabetes if None

    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
    doctype = '.dat'

    # default basedir
    default_basedir = sys_config.read('DataIn') if use_preprocessed else sys_config.read('DataExpRoot') # doc data dir
    basedir = kargs.pop('inputdir', default_basedir)
    verify_doc = kargs.get('verify_', True)

    # [params] ifiles, cohort_name, basedir
    docfiles = get_sources() # this specifies (coding sequence) source files
    print('io> reading sequences from %d documents' % len(docfiles))
    
    # [params] test
    iter_max_doc, iter_max_visit = kargs.get('max_doc', np.inf), kargs.get('max_visit', np.inf)

    # [params] doc (=> made global)
    # [note] ':' is reserved for prefixed strings like MED:1234, I9:309.81, etc. 
    token_sep = ','
    visit_sep = ';'
    token_end_history =  '$'
    doc_sep = patient_sep = '\n'

    # [note] use seqAlgo to simply the code if necessary
    simplify_code = False # kargs.get('simplify_code', False) or kargs.get('base_only', False)
    div(message='Simplify code? %s' % simplify_code, symbol='%')

    # sentences = CodeSeq(ipath) # a memory-friendly iterator
    # sequences = []  # that collects all visists across all patients 
    documents = []  # map from patient to their invidiual documents
    tokens = set() # unique tokens

    n_doc = len(list(iterdocs()))
    print('readDocPerPatient> found %d documents' % n_doc)
    vseq_last = None
    for j, pseq in enumerate(iterdocs()):  # pseq: patient sequence (combining all visits)
        if j >= iter_max_doc: break 

        # [note] pseq[-1] is newline
        sequences = [] # collect sequences (made out of visits)

        pseq = pseq.strip() # strip off '\n' 
        vseqx = pseq.split(visit_sep) 
        n_visits = len(vseqx)

        vseq_front, vseq_last = vseqx[:-1], vseqx[-1]
        for i, vseq in enumerate(vseq_front):  # vseq: visit sequence (i.e. a sequence of codes within a single visit)
            # if i >= iter_max_visit: break 

            # [key] operation
            vseq = transform(vseq.strip(), simplify_code=simplify_code, split=True, token_sep=token_sep) # split via token_sep
            vseq = transform_by_ptype(vseq, seq_ptype=seq_ptype)  # [todo] speed, factor out
            sequences.extend(vseq)  # combining visit-specific subsequence
            
        # add the last visit 
        vseq_last = vseq_last.strip()  # strip off white spaces   e.g. 125.0,137.0,11155.
        assert vseq_last[-1] == token_end_history, "diagnosis> last: %s ~? %s" % (vseqx[-1], vseq_last)
        vseq_last = vseq_last[:-1]
        assert len(vseq_last) > 0 
        vseq_last = transform(vseq_last, simplify_code=simplify_code, split=True, token_sep=token_sep)
        vseq_last = transform_by_ptype(vseq_last, seq_ptype=seq_ptype)

        if verify_doc and (j <= 2 or (j % 500 == 0) or (j > n_doc-3)): print('verify> vseq_last # %d => %s' % (j, vseq_last))

        sequences.extend(vseq_last)
        tokens.update(sequences)

        # patient doc
        documents.append(sequences)
        
    if verify_: 
        n_parsed = len(documents)
        assert n_doc == n_parsed, "n_doc: %d <> parsed result: %d" % (n_doc, n_parsed) # n_doc <- len(list(iterdocs()))
        summary()  # [log] number of patients/documents: 432000

        div(message='Number of documents: %d\nNumber of unique tokens: %d' % (n_doc, len(tokens)), symbol='%')

        # if documents were dictionary: utils.sample_dict(documents, n_sample=10)
        
        print('  + First document:\n%s\n' % documents[0])
        print('  + Last document:\n%s\n' % documents[-1])

        # randomly sampled 
        n_sample = 5
        for i, doc_seq in enumerate(random.sample(documents, n_sample)): 
            print('  >>> Random Doc #%d:\n%s\n' % (i, doc_seq))

    # sys.exit(0)
    # model = gensim.models.Word2Vec(sequences)
    return documents

def profileDocuments(D, **kargs): 
    """

    Input
    -----
    D: 

    Params
    ------
    seq_ptype 
    predicate 

    Output
    ------
    A dictionary containing the following keys/properties: 

    unique tokens: tokens_x, where x in {'regular', 'diag', 'med', }
    number of unique tokens: n_x, where x in {'regular', 'diag', 'med', }

    """
    def get_tokens(docs, test_=True): 
        tokens = set()
        for i, doc in enumerate(docs): 
            tokens.update(doc)
        if test_: 
            r = random.randint(0, len(D)-1)
            assert isinstance(docs[r], list), "ill-formatted input documents (example: %s)" % docs[r]
            # print("  + example codes:\n%s\n" % random.sample(tokens, min(len(tokens), 10)))
        return tokens
    def get_special_tokens(docs, ctype='regular'): 
        ctype = seqparams.normalize_ctype(ctype)
        docs, T, L = transform_docs(D=docs, ctype=ctype)
        # if ctype.startswith('reg'): 
        #     # noop
        #     pass 
        # else: 
        #     print('  + computing tokens | ctype=%s' % ctype)
        #     modified_docs = st.modify(docs, seq_ptype=ctype)
        #     return get_tokens(modified_docs) 
        return get_tokens(docs, test_=False) 
    def transform_docs(D, L=[], T=[], ctype='regular'): # params: seq_ptype, predicate, simplify_code
        # seq_ptype = kargs.get('seq_ptype', 'regular')
        # [params] items, policy='empty_doc', predicate=None, simplify_code=False
        # predicate = kargs.get('predicate', None)
        simplify_code = kargs.get('simplify_code', False)
        nD, nD0 = len(D), len(D[0])
        # this modifies D 
        D2, L2, T2 = st.transformDocuments(D, L=L, T=T, policy='empty_doc', seq_ptype=ctype, 
            predicate=None, simplify_code=simplify_code) # save only if doesn't exist 
        # D, labels = transformDocuments(D, L=labels, seq_ptype=seq_ptype)

        print('    + (after transform) nDoc: %d -> %d, size(D0): %d -> %d' %  (nD, len(D2), nD0, len(D2[0])))
        print('    + (after transform) nD: %d, nT: %d, nL: %d' % (len(D2), len(T2), len(L2)))

        return (D2, L2, T2)
    def do_splice(): 
        if not kargs.has_key('splice_policy'): return False
        if kargs['splice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True  # prior, posterior [todo] 'in-between'; e.g. from first occurence of X to last occurence of X
    def do_transform(): 
        if not seq_ptype.startswith('reg'): return True
        if do_splice(): return True
        if kargs.get('simplify_code', False): return True
        return True       

    import random
    from pattern import medcode as pmed
    import seqTransform as st

    ret = {}
    for ctype in ['regular', 'diag', 'med', ]:  # other types: 'lab'
        ret['tokens_%s' % ctype] = tokx = get_special_tokens(D, ctype=ctype)
        ret['n_%s' % ctype] = len(tokx)
        print("profileDocuments> example codes (ctype=%s):\n%s\n" % (ctype, random.sample(tokx, min(len(tokx), 10))))

    # pmed.classify_drug(alist) # keys: ['med', 'ndc', 'multum', 'unknown', ]
    return ret  # keys: 

def loadDocuments(**kargs):
    """
    Load documents (and labels). Adapted from seqClassify.loadDocuments()

    Params
    ------
    cohort
    read_dat: Read processed (.csv) or raw (.dat) documents? Set to False by default

    result_set: a dictionary with keys ['sequence', 'timestamp', 'label', ]
    ifiles: paths to document sources

    <not used>
    use_surrogate_label: applies only when no labels found in the dataframe

    Output: a 2-tuple: (D, l) where 
            D: a 2-D np.array (in which each document is a list of strings/tokens)
            l: labels in 1-D array

    """
    def load_docs(): 
        prefix = kargs.get('basedir', TDoc.prefix)  # basedir ~ inputdir in seqReader
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        if not ret:   # this is assuming that document sources exist in .csv format 
            ifiles = kargs.get('ifiles', [])
            ret = readDocFromCSV(cohort=cohort_name, inputdir=prefix, ifiles=ifiles, complete=True) # [params] doctype (timed)
        # seqx = ret['sequence'] # must have sequence entry
        # tseqx = ret.get('timestamp', [])  
        return ret
    def load_raw_docs(): 
        prefix = kargs.get('basedir', TDoc.prefix)  # basedir ~ inputdir in seqReader
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        if not ret:   # this is assuming that document sources exist in .csv format 
            print('loadDocuments> Warning: Reading from .dat source(s) ...')
            ifiles = kargs.get('ifiles', [])
            dx, tx = readTimedDocPerPatient(cohort=cohort_name, inputdir=prefix, ifiles=ifiles)
            assert len(dx) == len(tx), "The size between documents and timestamps is not consistent."
            ret['sequence'] = dx 
            ret['timestamp'] = tx 
        return ret
    def load_labeled_docs(): # read subset of the documents of the given label
        prefix = kargs.get('basedir', TDoc.prefix)
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        
        # [note] this function should be agnostic to seq_ptype
        # fpath = TDoc.getPath(cohort=kargs['cohort'], seq_ptype=kargs.get('seq_ptype', 'regular'), 
        #     doctype='labeled', ext='csv', basedir=prefix) 
        # assert os.path.exists(fpath), "labeled docuemnt source should have been created previously (e.g. by processDocuments)"
        # df_src = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # ['sequence', 'timestamp', 'label']

        if not ret: 
            ifiles = kargs.get('ifiles', [])
            ret = readDocFromCSV(cohort=cohort_name, inputdir=prefix, ifiles=ifiles, complete=True, 
                            stratify=True, label_name='label') # [params] doctype (timed)

            # this 'ret' is indexed by labels
        return ret
    def parse_row(df, col='sequence', sep=','): # [note] assuming that visit separators ';' were removed
        seqx = [] # a list of (lists of tokens)  
        
        isStr = False
        for i, row in enumerate(df[col].values):
            if isinstance(row, str): 
                isStr = True
            if i >= 3: break
            
        if isStr:  # sequence, timestamp
            for row in df[col].values:   # [note] label is not string
                tokens = row.split(sep)
                seqx.append(tokens)
        else: 
            # integer, etc. 
            seqx = list(df[col].values)
        return seqx
    def show_stats(): 
        print('seqClassify.loadDoc> nD: %d, nT: %d, nL: %d' % (len(D), len(T), len(L)))
        if len(L) > 0: 
            n_classes = len(set(L))  # seqparams.arg(['n_classes', ], default=1, **kargs) 
            print('  + Stats: n_docs: %d, n_classes:%d | cohort: %s' % (len(D), n_classes, cohort_name))
        else: 
            print('  + Stats: n_docs: %d, n_classes:? (no labeling info) | cohort: %s' % (len(D), cohort_name)) 
        return
 
    from seqparams import TSet, TDoc
    from labeling import TDocTag

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code
    #         

    # [params] cohort   # [todo] use classes to configure parameters, less unwieldy
    cohort_name = kargs.get('cohort', 'group-1')  
    tSingleLabelFormat = kargs.get('single_label_format', True)  # single label format ['l1', 'l2', ...] instead of multilabel format [['l1', 'l2', ...], [], [], ...]
    tReadDat = kargs.get('read_dat', False)  # read processed documents (.csv) by default 

    # [params]
    # read_mode = seqparams.TDoc.read_mode  # assign 'doc' (instead of 'seq') to form per-patient sequences
    # docSrcDir = kargs.get('basedir', TDoc.prefix) # sys_config.read('DataExpRoot')  # document source directory

    ### load model
    # 1. read | params: cohort, inputdir, doctype, labels, n_classes, simplify_code
    #         | assuming structured sequencing files (.csv) have been generated
    div(message='1. Read temporal doc files (cohort=%s) ...' % cohort_name)

    # [note] csv header: ['sequence', 'timestamp', 'label'], 'label' may be missing
    # [params] if 'complete' is set, will search the more complete .csv file first (labeled > timed > doc)
    
    # if result set (sequencing data is provided, then don't read and parse from scratch)
    ret = load_docs() if not tReadDat else load_raw_docs()
    assert len(ret) > 0
    D, T, L = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])
    nD, nT, nL = len(D), len(T), len(L) 

    ### determine labels and n_classes 
    print('info> document labeling is now automatic that depends on label_type (train, test, validation, unlabeled, etc.)')

    hasTimes = True if len(T) > 0 else False
    hasLabels = True if len(L) > 0 else False
    # [condition] len(seqx) == len(tseqx) == len(labels) if all available

    if hasTimes: assert len(D) == len(T), "Size inconsistency between (D: %d) and (T: %d)" % (len(D), len(T))

    # labels is not the same as tags (a list of lists)
    if hasLabels and tSingleLabelFormat: 
        # use the first label as the label by default (pos=0)
        L = TDocTag.toSingleLabel(L, pos=0) # to single label format (for single label classification)
        print('    + labels (converted to single-label format, n=%d): %s' % (len(np.unique(L)), np.unique(L)))
        assert len(D) == len(L), "Size inconsistency between (D: %d) and (L: %d)" % (len(D), len(L))

        # [condition] n_classes determined
    show_stats()
    return (D, L, T)

def readVisit(**kargs): 
    return readDocPerVisit(**kargs)  # one list per visit
def readDocPerVisit(**kargs): 
    """
    Read medical code time series from files and convert 
    it into a cananical form (e.g. a list of lists of medical codes):

    e.g. gensim's word2vec expects a sequence of sentences as its input. 
         Each sentence a list of words

    Can go through preprocess() first. 

    Input
    -----
    docfiles

    Memo
    ----
    1. default doc files: 
       condition_drug_seq.dat 
       condition_drug_timed_seq.dat  (with timestamps attached to codes)


    """
    def v_path(fname): 
        # [filter]
        if fname.find(doctype) <= 0: 
        	raise ValueError

        ipath = os.path.join(basedir, fname)
        assert os.path.exists(ipath) and os.path.getsize(ipath) > 0, "Invalid input: %s" % ipath

        return ipath

    def iterdocs(): # [key]
        for fname in docfiles: 
            if verify_doc: v_path(fname)
            lines = []
            fpath = os.path.join(basedir, fname)
            print('read> opening source document file at: %s' % fpath)
            with open(fpath, 'r') as fp: # read .dat file content 
                for line in fp:   # assuming that each line ends with '\n' and each line corresponds to a single patient's doc
                    yield line  # line includes '\n'

    def summary(): 
        print('> reading from %s input files under:\n%s\n' % (len(docfiles), basedir))
        print('> number of patients/documents: %d' % n_doc)

        return

    def get_sources(): 
        fp0 = kargs.get('ifile', None)
        if fp0 is not None: 
            assert isinstance(fp0, str) 
            docfiles0 = [fp0, ]
        else: 
            fp0 = os.path.join(basedir, seqparams.TDoc.getName(cohort=cohort_name))
            docfiles0 = [fp0, ]
        docfiles = kargs.get('ifiles', docfiles0)
        print('read> input files:\n%s\n' % docfiles) 

        # verfiy the source(s)
        for fp in docfiles: 
            assert os.path.exists(fp), "Invalid coding sequence source path: %s" % fp    

        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, basedir)
        return docfiles   
        
    # [params]
    test_, verify_ = False, True
    use_preprocessed = False 

    seq_compo = sequence_composition = kargs.get('seq_compo', 'condition_drug')
    cohort_name = kargs.get('cohort', None)  # None, default, diabetes

    # sequence pattern type: regular (ordering perserved) (use 'regular')
    #                        shuffled (removing ordering) (use 'random')
    #                        diagnostic codes only (use 'diag'), medications only (use 'med'), labs only, etc. 
    seq_ptype = kargs.get('seq_ptype', 'regular')
    # doc_basename = 'condition_drug'
    # if not seq_ptype.startswith('reg'): 
    #     doc_basename = '%s-%s' % (doc_basename, seq_ptype)

    doctype = '.dat'
    verify_doc = kargs.get('verify_', False)
        
    # default basedir
    default_basedir = sys_config.read('DataIn') if use_preprocessed else sys_config.read('DataExpRoot') # doc data dir
    basedir = kargs.get('inputdir', default_basedir)

    # [params] input
    docfiles = get_sources() # this specifies (coding sequence) source files

    # [params] test
    iter_max_doc, iter_max_visit = kargs.get('max_doc', np.inf), kargs.get('max_visit', np.inf)

    # [params] doc (=> made global)
    # [note] ':' is reserved for prefixed strings like MED:1234, I9:309.81, etc. 
    token_sep = ','
    visit_sep = ';'
    token_end_history =  '$'
    doc_sep = patient_sep = '\n'

    simplify_code = kargs.get('simplify_code', False) or kargs.get('base_only', False)
    div(message='Simplify code? %s' % simplify_code, symbol='%')

    # sentences = CodeSeq(ipath) # a memory-friendly iterator
    sequences = []  # that collects all visists across all patients 

    tokens = set() # unique tokens

    n_doc = len(list(iterdocs()))
    vseq_last = None

    for j, pseq in enumerate(iterdocs()):  # pseq: patient sequence (combining all visits)
        # print('test: j=%d, max=%d' % (j, iter_max_doc))
        if j >= iter_max_doc: break 

        # [note] pseq[-1] is newline
        pseq = pseq.strip() # strip off '\n' 
        vseqx = pseq.split(visit_sep)  # ';'
        n_visits = len(vseqx)

        vseq_front, vseq_last = vseqx[:-1], vseqx[-1]
        for i, vseq in enumerate(vseq_front):  # vseq: visit sequence (i.e. a sequence of codes within a single visit)
            # if i >= iter_max_visit: break 
            
            # [key]
            # string to a list of codes
            vseq = transform(vseq.strip(), simplify_code=simplify_code, split=True, token_sep=token_sep) # split via token_sep

            # process time stamps if any
            # times, vseq = strip_time(vseq)

            # perserve only the desired contents
            vseq = transform_by_ptype(vseq, seq_ptype=seq_ptype)  # transform the ouput sequence by seq_ptype (e.g. shuffle, diag codes only, etc.)
            
            # print('verify> vseq => %s' % vseq)
            tokens.update(vseq)
            sequences.append(vseq)
            
        # add the last visit 
        vseq_last = vseq_last.strip()  # strip off white spaces   e.g. 125.0,137.0,11155.
        assert vseq_last[-1] == token_end_history, "diagnosis> last: %s ~? %s" % (vseqx[-1], vseq_last)
        vseq_last = vseq_last[:-1]
        assert len(vseq_last) > 0 

        vseq_last = transform(vseq_last, simplify_code=simplify_code, split=True, token_sep=token_sep)
        vseq_last = transform_by_ptype(vseq_last, seq_ptype=seq_ptype)
        # => ['125', '137', '11155.'] if simplify_code <- True

        # assert vseq_last[-1] == ''  # note that '.'.split('.') => ['', '']
        # vseq_last = [e for e in vseq_last if e]
        # vseq_last = vseq_last[:-1]
        if verify_doc: 
            if j <= 2 or (j % 500 == 0) or (j > n_doc-3): print('verify> vseq_last # %d => %s' % (j, vseq_last))

        tokens.update(vseq_last)
        sequences.append(vseq_last)
        
    if verify_: 
        assert n_doc <= len(sequences)  # n_doc <- len(list(iterdocs()))
        summary()  # [log] number of patients/documents: 432000

        div(message='Number of documents: %d\nNumber of sentences: %d\nNumber of unique tokens: %d' % \
            (n_doc, len(sequences), len(tokens)), symbol='%')
        print('verify> example sequences:\n%s\n' % sequences[:10])

    # sys.exit(0)
    # model = gensim.models.Word2Vec(sequences)

    return sequences


def split(ratio=0.7, unit='doc', shuffle=True): # separate training and test sets in units of a. per-patient documents b. per-visist sequence
    """
    
    Input
    -----
    ratio: fraction of data (r) used for training, (1-r) for testing
    unit: 'doc' => separate training and test data according to per-patient document
          'visit' => separate training and test data according to visits (where each visit correspond to one sentence)

    """
    pass

def vectorize(sequences, **kargs): 
    """
    Subsummed by seqAnalyzer::vectorize()

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
    ----
    1. parameters
        size is the dimensionality of the feature vectors.

    """
    basedir = kargs.get('inputdir', sys_config.read('DataExpRoot'))
    ofile = kargs.get('output_file', 'test.word2vec')

    fpath = os.path.join(basedir, ofile)
    if kargs.get('load_model', True) and os.path.exists(fpath):
        model = gensim.models.Word2Vec.load(fpath)  # can continue training with the loaded model!
    else: 
        # [note] for workers to work, need cython installed
        model = gensim.models.Word2Vec(sequences, size=100, window=5, min_count=2, workers=4)  # [1]
        div(message='Model estimate complete', symbol='%')

    # model persistence
    if kargs.get('save_model', True): 
        model.save(fpath)

    return

def t_transform(**kargs): 
    codes = [481, '005', '039.1', 'V42.0', '010.2', '010.01', 11100, 'MED:31415', 'Test', "lepsis 20mg once every week", "S02.412A"]
    seq = ' '.join(str(c) for c in codes)
    print('input> seq:\n%s\n' % seq)
    oseq = icd9utils.getRootSequence(seq)
    print('output> seq:\n%s\n' % oseq)

    return

def t_vectorize(**kargs): 
    # preprocess(ofile='condition_drug.seq') # convert the input time series into a format where each visit is represented by a sentence (code sequence). 
    vectorize(read(**kargs))

    return

def t_make_doc(**kargs): 
    import seqparams
    cohort_name = kargs.get('cohort', 'PTSD')
 
    seq_compo = sequence_composition = kargs.get('seq_compo', 'condition_drug')
    default_ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo
    ifile = kargs.get('ifile', default_ifile) 
    ifiles = kargs.get('ifiles', None) # [todo]

    include_timestamps = True
    verify_doc = True

    # content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    # seqparams.normalize_ctype(ctype)    

    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))  # options: diag, regular, med

    print('info> make coding sequences and save it to .csv (cohort=%s)' % cohort_name)
    res = readToCSVDoc(load_=False, simplify_code=False, 
                verify_=verify_doc, include_timestamps=include_timestamps, 
                    seq_ptype=seq_ptype, cohort=cohort_name)
    assert not include_timestamps or len(res) == 2
    if include_timestamps: 
        n_docs = len(res[0])
        n_times = len(res[1])
        print('info> number of docs/patients: %d =?= number of tdocs: %d' % (n_docs, n_times))

    return

def t_sequencing(**kargs): 
    """
    Batch population-scale sequence reader. 
    For population-scale sequence maker, refer to seqMaker2

    Related
    -------
    seqMaker2.t_sequencing()

    Memo
    ----
    1. logging 
         test/log/population_stats.log

    """
    def overall_stats(): # <- tokens
        div(message='overall stats', symbol="%")
        n_diag = len(tokens['diag'])
        print('  + example diag codes:\n%s\n' % random.sample(tokens['diag'], min(n_diag, 20)))
        n_med = len(tokens['med'])
        print('  + example medication codes:\n%s\n' % random.sample(tokens['med'], min(n_med, 20)))

        n_total = len(tokens['regular'])
        print("  + tokens consistent? nT=%d =?= (nT':%d <- nD=%d + nM=%d)" % (n_total, n_diag+n_med, n_diag, n_med))

        n_unknown = len(tokens['unknown']) # n_total-(n_diag+n_med)
        print('  + example unknown codes:\n%s\n' % random.sample(tokens['unknown'], min(n_unknown, 20)))

        print('overall_stats> Among medical codes of ALL types (e.g. diag, med) ...')
        print('  + diagnostic | n=%d' % n_diag)
        print('  + medication | n=%d' % n_med)
        print('  + total      | n=%d' % n_total)
        print('  + unknown    | n=%d =?= %d (total-med-diag)' % (n_unknown, n_total-(n_diag+n_med)))
        return
    def med_stats(): # <- tokens 
        div(message='medication stats', symbol="%")
        adict = pmed.classify_drug(list(tokens['med'])) # keys: ['med', 'ndc', 'multum', 'unknown', ]
        n_MED = len(adict['med'])
        print('  + example MED codes:\n%s\n' % random.sample(adict['med'], min(n_MED, 15)))
        n_NDC = len(adict['ndc'])
        print('  + example NDC codes:\n%s\n' % random.sample(adict['ndc'], min(n_NDC, 15)))
        n_MULTUM = len(adict['multum'])
        print('  + example MULTUM codes:\n%s\n' % random.sample(adict['multum'], min(n_MULTUM, 10)))
        n_unknown = len(adict['unknown'])
        print('  + example Unknown codes:\n%s\n\n' % random.sample(adict['unknown'], min(n_unknown, 15)))

        print('med_stats> Among all medication codes ...')
        print('  + MED    | n=%d' % n_MED)
        print('  + NDC    | n=%d' % n_NDC)
        print('  + MULTUM | n=%d' % n_MULTUM)
        return adict
    def diag_stats(): 
        # pmed.classify_condition(tokens['diag'])
        return
    def load_and_profile(i): 
        cohort_name = 'group-%s' % (i+1)  # starting from 1
        div(message='t_sequencing> processing cohort=%s ...' % cohort_name)
        D, T, L = loadDocuments(cohort=cohort_name, basedir=basedir_sequencing, read_dat=True)
        assert len(D) > 0, "empty documents | cohort=%s, basedir=%s" % (cohort_name, basedir_sequencing)
        ret = profileDocuments(D, **kargs)

        # for ctype in ctypes: 
        #     tokens[ctype].update(ret['tokens_%s' % ctype])
        return ret

    from pattern import medcode as pmed
    import random
    # read population documents (generated via seqMaker2.t_sequencing())
    n_parts = 10
    basedir = sys_config.read('DataExpRoot')
    basedir_sequencing = os.path.join(basedir, 'sequencing') 

    ctypes = ['regular', 'diag', 'med', ] # other values: 'regular', 'lab'
    tokens = {ct: set() for ct in ctypes}
    unknown = set()
    for i in range(n_parts): # condition_drug_seq-group-{1 ... 10}.dat 
        ret = load_and_profile(i)
        for ctype in ctypes: 
            tokens[ctype].update(ret['tokens_%s' % ctype])
    
    # unknown tokens
    ctypes.remove('regular')
    assert not 'regular' in ctypes
    V = set()
    for ctype in ctypes:
        V.update(tokens[ctype])
    tokens['unknown'] = tokens['regular']-V
    ### statistics 
    # condition: tokens available
    overall_stats()
    med_stats()

    return

def test(**kargs): 
    """

    Memo
    ----
    1. Basic operations should go first. 
       e.g. t_transform() needs to work perfectly before even considering t_vectorize()

    """
    ### medical coding sequence processing
    # preprocess(**kargs)  # turn coding sequences into a human readable format

    # read(simplify_code=False)
    # t_transform(**kargs)
    
    # t_vectorize(**kargs)

    ### make temporal documents 
    # t_make_doc(**kargs)  # also see pathwayAnalyzer.py

    ### medical coding sequence statistics 
    t_sequencing(**kargs) 

    return

if __name__ == "__main__":
	test()

