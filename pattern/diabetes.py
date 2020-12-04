import os, sys, re, random, collections

# install tsne via pip
# from tsne import bh_sne

import pandas as pd 
from pandas import DataFrame

# temporal sequence modules
from config import seq_maker_config, sys_config
from batchpheno.utils import div

try:
    import cPickle as pickle
except:
    import pickle

# from seqmaker import seqTransform as st  # [design] this leads to mutual dependency!

# for impl (approximate) phenotyping algorithm based on 
# from seqmaker import seqAnalyzer as sa
# from seqmaker import seqReader as sr

p_diabetes = re.compile(r"(?P<base>250)\.(?P<subclass>[0-9]{1,2})")
p_secondary_diabetes = re.compile(r"(?P<base>249)\.(?P<subclass>[0-9]{1,2})")
p_gestational_diabetes = re.compile(r"(?P<base>648)\.(?P<subclass>[0-9]{1,2})")


class Coding(object): 
    
    dmap = {'type_0': (1, 0, 0)}
    labelBitSize = bitSize = 3  # number of digits used to encode all diabetes types

    @staticmethod
    def decode(lx):
        assert len(lx) == Coding.bitSize 
        lx = tuple(lx)
        
        # ret = {}
        lv, lc = 'type_0', 0  # default: no diabetes
        if lx == (0, 0, 0):
            lv, lc = 'type_0', 0  # no diabetes
        elif lx == (1, 0, 0): 
            lv, lc = 'type_1', 1  # type I
        elif lx == (0, 1, 0): 
            lv, lc = 'type_2', 2   # type II
        elif lx == (0, 0, 1): 
            lv, lc = 'type_3', 3  # gestational
        elif lx == (1, 1, 0): 
            lv, lc = 'type_1_2', 4  # type I & II
        elif lx == (1, 0, 1): 
            lv, lc = 'type_1_3', 5  # type I & gestational
        elif lx == (0, 1, 1): 
            lv, lc = 'type_2_3', 6  # type II and gestational
        elif lx == (1, 1, 1):      
            lv, lc = 'type_1_2_3', 7

        # ret['label'] = lv 
        # ret['target_field'] = lc  # numeric value of the class label, this goes into the training set
    
        return (lv, lc)

# [design] primary API (e.g. transform, phenotype) should go on top

def transform(seq, **kargs):
    """

    Memo
    ----
    1. use case: 
       transform(seq, policy='noop', seq_ptype='diag')

    """
    # if not kargs.has_key('predicate'): kargs['predicate'] = is_diabetes
 
    # # [args] policy: noop to bypass cut op. 
    # if not kargs.has_key('policy'): kargs['policy'] = 'prior'  # i.e. only looking at the sequence prior to the first diabetes-related diagnosis
    # if not kargs.has_key('seq_ptype'): kargs['seq_ptype'] = 'regular'
    # return st.transform(seq, **kargs)  # operations: cut -> transform_by_ptype (e.g. diag, med)
    raise ValueError, "Use seqTransform for this operation!"

def cut(seq, **kargs): 
    """
    Don't define it here. Keep pattern module clean such that only disease-specific operations are allowed. 

    Input
    -----
    * policy: 
        'regular' => noop
        'prior' => coding sequence segment prior to the cut point determined by the predicate
        'posterior'
    """ 
    # if not kargs.has_key('predicate'): kargs['predicate'] = is_diabetes
    # if not kargs.has_key('policy'): kargs['policy'] = 'prior'
    # return st.cut(seq, **karsg)
    raise ValueError, "Use seqTransform for this operation!"

def generate(seqx, **kargs): 
    """
    Input: a list of documents (in list-of-tokens format)
    Ouput: a dictionary mapping from (class) labels to documents' position IDs

    Related
    -------
    label(seqx, **kargs): use this to generate a sequence of labels, one for each input document

    Module
        seqmaker.labeling

    """
    return genLabels(seqx, **kargs)
def genLabels(seqx, **kargs): # phenotypeDoc + phenotypeindex 
    """
    Input
    -----
    seqx: a list of medical coding sequences (e.g. temporal sequences of diagnoses and treatments)
          

    """
    labelsets = phenotypeDoc(docs, **kargs)
    return phenotypeindex(labelsets, **kargs)

def label(seqx, **kargs): 
    """

    Input: 

    Params: 
        min_count: minimum number of occurrences of disease codes before declaring positive

    Output: 

    """
    
    # encode documents in phenotypic/class labels 
    labelsets = phenotypeDoc(seqx, **kargs)  # [params] min_count, seq_ptype, cohort
    ret = labelToID(labelsets)  # this is only label-to-ID format

    # convert to a list of labels matching docs positions 
    labels = [None] * len(seqx)
    ulabels = set()
    if kargs.get('numeric_', True):
        for label, docIDs in ret.items():  # 
            if not isinstance(label, int): continue
            for i in docIDs: 
                labels[i] = label
            ulabels.add(label) # [test]
    else: # canonical
        for label, docIDs in ret.items(): 
            if isinstance(label, int): continue
            for i in docIDs: 
                labels[i] = label
            ulabels.add(label) # [test]

    assert len(ulabels) * 2 == len(ret), "Inconsistent canonical and numeric labels (got: %d but total: %d)" % \
        (len(ulabels), len(ret))
    # [condition] no unlabeled docs 
    unlabeled = [i for i, l in enumerate(labels) if l is None]
    assert len(unlabeled) == 0, "There exist unlabeled documents at positions: %s" % unlabeled

    return labels

def getNumericLabels(seqx, **kargs):
    kargs['numeric_'] = True
    return label(seqx, **kargs)

def getCanonicalLabels(seqx, **kargs):
    kargs['numeric_'] = False 
    return label(seqx, **kargs)
def getNamedLabels(seqx, **kargs):
    kargs['numeric_'] = False 
    return label(seqx, **kargs) 
 
def labelToID(labelsets):
    """
    Map labels in multilabel format (e.g. [1, 0, 0] as type I diabetes) to: 
        lv: canonical class label 
        lc: numerical class label

    """
    ret = {}
    for i, lset in enumerate(labelsets): 
        lkey = tuple(lset)
        
        lv, lc = Coding.decode(lkey)  # canonical class label, numeric class label 
        
        for l in [lv, lc, ]: 
            if not ret.has_key(l): ret[l] = [] 
 
        # canonical class label 
        ret[lv].append(i)

        # numeric class label
        ret[lc].append(i)
    return ret 

def phenotypeIndex(labelsets, **kargs):
    """
    Given phenotyped labels (upon completion of phenotypeDoc), find document indices 
    for different disease subtypes (e.g. type I, type II, type I + type II, etc.)

    Output
    ------
    a dictionary mapping from class labels to document (positional) IDs
    """
    # note: type 3 ~ gestational
    header = ['type_0', 'type_1', 'type_2', 'gestational', 'type_3', 'type_1_2', 'type_1_3', 'type_2_3', 'type_1_2_3']
    res = {h: [] for h in header} # result set 

    res = {}
    for i, lset in enumerate(labelsets): 
        lkey = tuple(lset)
        
        
        lv, lc = Coding.decode(lkey)  # canonical class label, numeric class label 
        
        for l in [lv, lc, ]: 
            if not res.has_key(l): res[l] = [] 
 
        # canonical class label 
        res[lv].append(i)

        # numeric class label
        res[lc].append(i)

        # res[lkey].append(i)  # class tuples => document positions

    # [old]
    # # rename keys: idiom > mydict[new_key] = mydict.pop(old_key)
    # for lx, idx in res.items():  # foreach multilabel (lx) and doc IDs (idx)
    # 	if lx == (0, 0, 0):
    # 		res['type_0'] = res.pop(lx)    # index by position
    #     elif lx == (1, 0, 0): 
    #     	res['type_1'] = res.pop(lx)
    #     elif lx == (0, 1, 0): 
    #     	res['type_2'] = res.pop(lx)
    #     elif lx == (0, 0, 1): 
    #     	res['gestational'] = res.pop(lx)
    #     	res['type_3'] = res['gestational'] # duplicate the entry
    #     elif lx == (1, 1, 0): 
    #     	res['type_1_2'] = res.pop(lx)
    #     elif lx == (1, 0, 1): 
    #     	res['type_1_3'] = res.pop(lx)
    #     elif lx == (0, 1, 1): 
    #         res['type_2_3'] = res.pop(lx)
    #     elif lx == (1, 1, 1):      
    #         res['type_1_2_3'] = res.pop(lx)
    #     else: 
    #     	raise ValueError, "Invalid label set: %s" % str(lx)

    # # binary classification considering type 1 and type 2 diabetes 
    # res[0] = res['type_1']
    # res[1] = res['type_2']

    # # 3-class problem 
    # res[2] = res['gestational']

    # # multiclass problem 
    # res[3] = res['type_1_2']
    # res[4] = res['type_1_3']
    # res[5] = res['type_2_3']
    # res[6] = res['type_1_2_3']

    return res   # map classe labels to indices (into temporal sequences)

# [policy]
def phenotype(doc, **kargs):
    min_count = kargs.get('min_count', 1) 

    # [policy]
    n_type1 = n_type2 = n_birth = 0
    labels = [0, 0, 0]  # default: False for all labels (in the order of type I, II, gestational)
    for e in doc: 
        if is_diabetes_type1(e): 
            n_type1 += 1 
            if n_type1 >= min_count: labels[0] = 1
        if is_diabetes_type2(e): 
            n_type2 += 1 
            if n_type2 >= min_count: labels[1] = 1
        if is_diabetes_gestational(e): 
            n_birth += 1
            if n_birth >= min_count: labels[2] = 1
        
        # [todo] add other conditions here after increasing the dim of labels appropriately


    return labels # multilabel

def toBinaryClass(docs, **kargs): 
    """

    Related
    -------
    phenotypeIndex(labelsets, **kargs)
    """
    pass


# [precond] input sequences/docs should contain diagnostic codes
def phenotypeDoc(docs, **kargs): # this produces mutliple labels
    """
    phenotype input documents (consisting of temporal sequences of codes) according to 
    the mention of related diagnostic codes. 

    This routine generates the labeling in a vector form (represented by tuple)

    Related 
    -------
    1. phenotypeIndex()

    """
    def v_labels(): 
        counter = collections.Counter(lD) 

        ndoc_t0 = counter[(0, 0, 0)]

        ndoc_t1 = counter[(1, 0, 0)]
        ndoc_t2 = counter[(0, 1, 0)]
        ndoc_t3 = counter[(0, 0, 1)]

        ndoc_t1t2 = counter[(1, 1, 0)]
        ndoc_t1t3 = counter[(1, 0, 1)]
        ndoc_t2t3 = counter[(0, 1, 1)]

        ndoc_all = counter[(1, 1, 1)]

        div(message='Count statistics ...')
        print('> unknown type (all false): %d' % ndoc_t0)
        print('> type I  only:             %d' % ndoc_t1)
        print('> type II only:             %d' % ndoc_t2)
        print('> birth-related:            %d' % ndoc_t3)

        print('> type I + type II:         %d' % ndoc_t1t2)
        print('> type I + birth:           %d' % ndoc_t1t3)
        print('> type II + birth:          %d' % ndoc_t2t3)

        print('> all types:                %d' % ndoc_all)

        return
    # def save(fname=None): 
    #     header = ['type_1', 'type_2', 'gestational']

    #     # given labelsets 
    #     adict = {h:[] for h in header}
    #     for i, lset in enumerate(labelsets): 
    #         adict['type_1'].append(lset[0])
    #         adict['type_2'].append(lset[1])
    #         adict['gestational'].append(lset[2])

    #     df = DataFrame(adict, columns=header)
    #     if fname is None: 
    #         fname = '%s_labels.%s' % (doc_basename, doctype)
    #     fpath = os.path.join(basedir, fname)
    #     print('output> saving labels to %s' % fpath)
    #     df.to_csv(fpath, sep=fsep, index=False, header=True)
    #     return df
    seq_ptype = kargs.get('seq_ptype', 'regular') # values: regular, random, diag, med, lab ... default: regular
    cohort_name = kargs.get('cohort', 'diabetes')
    doc_basename = 'phenotyping'
    
    identifier = "%s-%s" % (seq_ptype, cohort_name)

    lD = []  # labeled D (dataset)
    basedir = sys_config.read('DataExpRoot') # os.path.join(os.getcwd(), 'data')
    fpath = os.path.join(basedir, '%s-%s.pkl' % (doc_basename, identifier)) 
    if kargs.get('load_', True) and os.path.exists(fpath): 
        lD = pickle.load(open(fpath, "rb" ))
        if len(lD) > 0: 
            print('input> loaded phenotyped/labeled docs from %s' % fpath)
            return lD
        else: 
            print('phenotypeDoc> Could not load pre-computed phenotype labels from %s' % fpath)

    min_count = kargs.get('min_count', 1) # minimum count before concluding positive 
    stats = {}  #  
    for i, doc in enumerate(docs):  # order independent
        
        n_type1 = n_type2 = n_birth = 0

        # multilabel format: a label has more than 1 digit
        labels = [0, 0, 0]  # default: False for all labels (in the order of type I, II, gestational)
        for e in doc:   # for each token in the document 
            if is_diabetes_type1(e): 
                n_type1 += 1 
                if n_type1 >= min_count: labels[0] = 1
            if is_diabetes_type2(e): 
                n_type2 += 1 
            	if n_type2 >= min_count: labels[1] = 1
            if is_diabetes_gestational(e): 
                n_birth += 1
            	if n_birth >= min_count: labels[2] = 1
        lD.append(tuple(labels))

    if kargs.get('save_', True):
        pickle.dump(lD, open(fpath, 'wb'))  

    # [test]
    v_labels()

    # now we need to map this to a single label
    
    return lD

def has_type1(seq): 
    for e in seq: 
    	if is_diabetes_type1(e): 
    		return True 
    return False

def has_type2(seq):
    for e in seq: 
    	if is_diabetes_type2(e): 
    		return True 
    return False  

def has_gestational(seq):
    for e in seq: 
    	if is_diabetes_gestational(e): 
    		return True 
    return False 
def has_type3(seq): 
    return has_gestational(seq)    

def cohort_diabetes(code): 
    cmap = {}
    cmap['250'] = {}
    cmap['type_1'] = ['250.01', '250.03', '250.11', '250.13', '250.21', '250.23', '250.31', '250.33', ]   

    # .00 type II or unspecified type, not stated as uncontrolled .02  type II or unspecified type, uncontrolled
    cmap['type_2'] = ['250.00', '250.02', '250.10', '250.12', '250.20', '250.22', '250.30', '250.32', ] 

def is_diabetes_type1(code): 
    scode = str(code).strip()
    # print('type1> input: %s' % scode)
    m = p_diabetes.match(scode)
    # if m: print('m: %s, n: %d, last digit: %s' % (m.group(), len(scode), scode[-1]))
    # base, subclass = 
    if m:
        subc = m.group('subclass')
    	if len(subc) == 2 and subc[-1] in ('1', '3', ): 
    		return True 
    return False

def is_diabetes_type2(code): 
    scode = str(code)
    m = p_diabetes.match(scode)
    if m: 
        subc = m.group('subclass')
    	if len(subc) == 2 and subc[-1] in ('0', '2', ): 
    		return True 
    return False

def is_diabetes_secondary(code): 
    scode = str(code)
    if scode[:3] == '249': 
    	return True 
    return False

def is_diabetes_gestational(code): 
	# 64800 64801 64802 64803 64804 64880 64881 64882 64883 64884
    scode = str(code)
    m = p_gestational_diabetes.match(scode)
    # if scode[:3] == '648': 
    if m: 
        subc = m.group('subclass')
        if subc[0] in ('0', '8', ):
    	    return True 
    return False

def is_diabetes(code): 
    scode = str(code)
    if scode[:3] in ('249', '250', '648', ): 
    	return True 
    return False	

def has_abnormal_blood(code): 
    # 7902 79021 79022 79029 
    scode = str(code)
    if scode[:3] == '790': 
    	if scode in ('790.2', '790.21', '790.22', '790.29'): # abnormal glocose
            return True
    return False 

def has_abnormal_urine(code): 
    # 7915 7916 
    scode = str(code)
    if scode[:3] == '791': 
    	if scode in ('791.5', '791.6', ):  # glycosuria, acetonuria
    	    return True 
    return False

def involve_insulin_pump(code): 
	# V4585 V5391 V6546 
    # V45.85: presence of insulin pump
    # V53.91: fitting and adjustment of insulin pump	
    # V65.46: Encounter for insulin pump training
    scode = str(code)
    if scode in ('V45.85', 'V53.91', 'V65.46', ): 
    	return True
    return False 

def test(): 
    codes = ['250.03', '250.00', 250.2, 250.24, 249.00, 250.01, '648.1', '648.00', '648.82', '649.12']
    print('input> %s' % codes)
    adict = {'type I': is_diabetes_type1, 'type II': is_diabetes_type2, 'secondary': is_diabetes_secondary, 'gestational': is_diabetes_gestational}
    for typ, routine in adict.items(): 
    	print "> %s => %s" % (typ, [c for c in codes if routine(c)])

    # transform 
    seq = ['786.50', '786.50', '67894', '413.9', '250.00', '17.2']
    subseq = transform(seq, policy='prior', inclusive=True, seq_ptype='diag')
    print('> diag seq: %s' % subseq)


if __name__ == "__main__": 
    test()



