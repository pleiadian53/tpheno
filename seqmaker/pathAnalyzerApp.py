# encoding: utf-8

# main module
import pathAnalyzer
import docProc

# supportive modules 
from batchpheno import icd9utils, sampling, qrymed2
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed
import seqparams  # all algorithm variables and settings
import vector 

### App Module of pathAnalyzer (related: seqAnalyzerApp)

def processDocuments(**kargs):
    """
    Load and transform documents (and ensure that labeled source file exist (i.e. doctype='labeled')). 

    Params
    ------
    1. for reading source documents
        cohort
        ifiles: paths to document sources (if given, the cohort is ignored)

    2. document transformation 
        seq_ptype 
        predicate 
        simplify_code

    Output: a 3-tuple: (D, T, l) where 
            D: a list of documents (in which each document is a list of strings/tokens)
            T: a list of timestamps
            l: labels in 1-D array

    Use 
    ---
    processDocuments(cohort)

    Note
    ----
    1. No need to save a copy of the transformed documents because the derived labeled source (doctype='labeled')
       subsumes it.  

    """
    import docProc as dp  # document processor module (wrapper app of seqReader, seqTransformer)
    return dp.processDocuments(**kargs)

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
    # import seqAnalyzer as sa 

    cohort_name = kargs.get('cohort', 'CKD')
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')) 
    d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
    # read_mode = 'timed'  # 'doc', 'seq'

    # data source from sequencing (entire population)
    n_parts = 1   # only used for global sequenced data

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
        for _ in range(n_parts):
            # cohort_name = 'group-%s' % (i+1)  
                
            ### read 
            # D: corpus, T: timestamps, L: class labels < load + transform + (labeled_seq file)
            D, T, L = processDocuments(cohort=cohort_name, seq_ptype=seq_ptype, 
                        predicate=kargs.get('predicate', None),
                        simplify_code=kargs.get('simplify_code', False), 
                        ifiles=kargs.get('ifiles', []), 

                        ### transformation
                        slice_policy=kargs.get('slice_policy', 'noop'), # slice operations {'noop', 'prior', 'posterior', }
                        slice_predicate=kargs.get('slice_predicate', None), 
                        cutpoint=kargs.get('cutpoint', None),
                        inclusive=True)  
            seqx, tseqx = D, T   # just aliasing 
        
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
        print('  + n_uniq_signatures: %d, n_persons: %d => ratio: %f' % (n_uniq_signatures, n_persons, r))

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

        print('status> analysis (policy=%s, ctype=%s) complete ------* '  % (policy, seq_ptype))

    return

def t_deidentify_global_core(**kargs):  # template function refactored from seqAnalyzer
    """
    Determine sequence uniqueness wrt global sequenced EHR data. 
    """
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
    import seqAnalyzer as sa 

    seq_ptype = seqparams.normalize_ptype(kargs.get('seq_ptype', 'diag'))
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
                
            seqx, tseqx = sa.read2(load_=False, save_=False, simplify_code=False, read_mode='timed', verify_=False, 
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

def t_deidentify_global(**kargs): 

    ctypes = ['regular', 'diag', 'med', ] # 'diag', 'med', 
    for ctype in ctypes: 
        t_deidentify_global_core(seq_ptype=ctype)

    return

def t_deidentify_cohort(**kargs):
    ctypes = ['regular', 'diag', 'med', ] # 'diag', 'med', 
    for ctype in ctypes: 
        t_deidentify(seq_ptype=ctype)  # cohort-specific

    return

def test(**kargs): 

    # singularity, uniqueness of coding sequences
    t_deidentify_global(**kargs)

    return