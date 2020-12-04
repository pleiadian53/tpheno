import re 
import string

# import all the disease patterns
import diabetes, ptsd

################################################
#  Package: pattern
#  Similar modules: icd9utils 
#
#  medcode is a tentative module that encapsulates 
#  all disease-specific modules (specially for 
#  creating labels for documents)
# 
#  Related 
#  -------
#    - batchpheno.icd9utils 
#
################################################

class ICD9: 
	pass 

class ICD10: 
	pass 

class SNOMED: 
	pass


### Global Parameters

prefix_prescription = 'prescription'
prefix_condition = 'condition'  # [todo] check consistency with seqMaker*

# General diagnostic codes
# p_icd9_base = re.compile(r'(?P<base>\d{3}\b)')
p_icd9_base = re.compile(r'^(V\d{2}|\d{3}|E\d{3})$')
p_icd9_sub = re.compile(r'\d{1,2}\b')  # ICD-9 decimal points (subcategory)
p_icd9 = re.compile(r'(?P<char>v|e)?(?P<base>\d{1,3}\b)(\.(?P<decimal>\d{1,2}\b))?', re.I) # re.compile(r'\d+\.\d+')

p_icd9_v2 = re.compile(r'^(V\d{2}(\.\d{1,2})?|\d{3}(\.\d{1,2})?|E\d{3}(\.\d)?)$')
# p_icd9_v2 = re.compile(r'([V\d]\d{2}(\.?\d{0,2})?|E\d{3}(\.?\d)?|\d{2}(\.?\d{0,2})?)')
# p_icd9_v2 = re.compile(r'^(?P<base>(V\d{2})(\.\d{1,2})?|(?P<base>\d{3})(\.\d{1,2})?|(?P<base>E\d{3})(\.\d)?)$')

# p_icd10 = re.compile(r'(?P<base>\w+)\.(?P<subcategory>\w+)') # [todo]
p_generic = re.compile(r'(?P<category>\w+)(\.(?P<subclass>\w+))?')  # [todo] this also matches S8.

# [note] suppose that c = 'S74.7A712' then the last '2' will not be matched
p_icd10_base = re.compile(r'(?P<category>[A-TV-Z][0-9][A-Z0-9])')
p_icd10_sub = re.compile(r'\w{3,4}\b')
p_icd10 = re.compile(r'(?P<category>[A-TV-Z][0-9][A-Z0-9])(\.(?P<subclass>\w{1,4}\b))?', re.I) # this matches S53.

# ODHSI
p_source_val = re.compile(r'(?P<prefix>(?P<type>\w+):)?(?P<code>\d{2,12}\b)', re.I) # MED: without following a code is not a match 

p_alpha = re.compile(r'[a-zA-Z]+')
p_alphanum = re.compile(r'\w+')

# drug med code (e.g. NDC: 11 digits, MED: at most 6 digits)
p_medcode = re.compile(r'\d{4,12}\b')  # \b, so no more than 12
p_MED = re.compile(r'\d{1, 7}\b')  # no more than 6 digits so far
p_medcode2 = re.compile(r'(?P<prefix>(?P<type>MULTUM|NDC|MED):)(?P<code>\d{2,12}\b)', re.I)

DrugPrefix = ('MULTUM', 'NDC', 'MED') # 'MULTUM', 'NDC', 'MED'
ConditionPrefix = ('I10', 'I9') 
p_cond_prefix = 'I(10|9):'

def getSurrogateLabels(docs, **kargs): 
    """

    disease modules/classes needs to provide generate() method

    """
    # label sequences in a manner that that adheres to the  phenotyping criteria ~ diagnostic codes
    cohort_name = kargs['cohort'].lower()
    if cohort_name.startswith('diab'): 
        
        # [params]
        # res key: ['type_0', 'type_1', 'type_2', 'gestational', 'type_3', 'type_1_2', 'type_1_3', 'type_2_3', 'type_1_2_3']
        return diabetes.getNumericLabels(docs, **kargs)
    elif cohort_name.startswith(('ptsd', 'post', )):
        return ptsd.getNumericLabels(docs, **kargs)
    else: 
        print('medcode.getSurrogateLabels> Unsupported cohort: %s > Assuming all cases are positive (label=1)' % cohort_name) 

    # default: all cases are positive (1)
    return {1: range(len(docs))}  # disease types to indices 

def label(docs, **kargs):
    # label sequences in a manner that that adheres to the  phenotyping criteria ~ diagnostic codes
    cohort_name = kargs['cohort'].lower()
    if cohort_name.startswith('diab'): 
        
        # [params]
        print('info> calling diabetes module (numerical labels)')
        return diabetes.getNumericLabels(docs, **kargs)  # 
    elif cohort_name.startswith(('ptsd', 'post', )):
        return ptsd.getNumericLabels(docs, **kargs)
    else: 
        print('medcode.getSurrogateLabels> Unsupported cohort: %s > Assuming all cases are positive (label=1)' % cohort_name) 

    # default: all cases are positive (1)
    return [1] * len(docs)  # disease types to indices 

def binarize(labels, positive):  # [refactor] labeling
    """
    Turn multiclass labels into binary labels 

    """
    # n_positive, n_negative = len(positive), len(negative)
    # assert n_positive > 0 or n_negative > 0
    # assert (n_positive + n_negative) == len(labels)

    n_active = 0

    # make deep copy
    # binLabels = np.empty_like(labels)
    binLabels = np.ones(len(labels))
    for i, label in enumerate(labels):     
        if label in positive: 
            # labels[i] = 1    # this modifies the list in place
            binLabels[i] = 1
            n_active += 1 
        else: 
            # labels[i] = 0
            binLabels[i] = 0
    assert n_active > 0, "None of the labels is detected positive defined by %s" % str(positive)
    return binLabels

def label_binarize(labels, classes): 
    # from sklearn.preprocessing import label_binarize
    import sklearn
    y = sklearn.preprocessing.label_binarize(labels, classes=classes) # classes e.g. [0, 1, 2] for 3-class Iris problem
    return y

def classify_drug(alist): 
    types = ['med', 'ndc', 'multum', 'unknown', ]  # 3 basic types of medication coding can be found in odhsi DB
    adict = {t: [] for t in types}
    for c in alist: 
        code0 = str(c).strip()
        crepr = code0.split(':'); nrepr = len(crepr)
        # assert nrepr in (1, 2), "invalid code: %s" % code0

        prefix = code = None
        if nrepr == 1: 
            code = crepr[0]   # code0
        elif nrepr == 2:  # is prefixed? 
            prefix, code = crepr[0], crepr[1]
            prefix = prefix.lower()
        else: 
            raise ValueError, "invalid code: %s" % code0

        # if medcode.isMED(code): 
        #     medx.append(code)
        
        if prefix is not None: 
            if prefix == 'med': 
                adict['med'].append(code)
            elif prefix == 'ndc': 
                adict['ndc'].append(code)
            elif prefix.startswith('mul'): 
                adict['multum'].append(code)
            else: 
                print('unusual prefix: %s' % prefix)
                adict['unknown'].append(code)
        else: 
            if p_MED.match(code): 
                adict['med'].append(code)
            else: # ambiguous coding 
                adict['unknown'].append(code)
    
    # [explore] min and max number of digits for each code type? 
    lengths = [len(c) for c in adict['med']]
    nd_med = {'min': min(lengths), 'max': max(lengths)} if lengths else {'min': -1, 'max': -1}  # min=1, max=6? 
    
    lengths = [len(c) for c in adict['ndc']]
    nd_ndc = {'min': min(lengths), 'max': max(lengths)} if lengths else {'min': -1, 'max': -1} # {'min': 1, 'max': 12}

    lengths = [len(c) for c in adict['multum']]
    nd_mul = {'min': min(lengths), 'max': max(lengths)} if lengths else {'min': -1, 'max': -1}

    lengths = [len(c) for c in adict['unknown']]
    nd_other = {'min': min(lengths), 'max': max(lengths)} if lengths else {'min': -1, 'max': -1}

    return adict

def classify_lab(alist): 
    adict = {}
    return adict

# predicates 
def isSourceVal(x): 
    if x.startswith(DrugPrefix): 
        return True
    if p_source_val.match(x): # this only applies to MED or NDC codes
        return True 
    return False

def isPrefixedDrugSourceVal(x):
    """
    Assuming codes have proper prefix
    """
    if x.find(':') > 0: 
        prefix = x.split(':')[0]
        prefix = prefix.lower()
        return prefix in DrugPrefix  # 'MED', 'MULTUM', "NDC"
    return False

def isSingletonDrugName(x): # single word 
    if p_alpha.match(x): 
        return True
    return False

def isDescription(x): # multiple words
    xt = x.split()
    tval = True
    cutoff = 5 
    for i, e in enumerate(xt): 
        if i > cutoff: break 
        if not p_alphanum.match(e): 
            tval = False
            break
    return tval


def removeDiagPrefix(x):
    return re.sub('^I(10|9):', '', x)  # I(10|9):
def removeDiagPrefix2(x, test_=True):
    code = x
    if x.startswith(ConditionPrefix):  
        elems = x.split(':')
        code = elems[1]
        if test_: assert elems[0] in ConditionPrefix, "Ill-formatted data element? %s (prefix=%s)" % (x, elems[0])
    return code

def isICD9v0(x): # 129. is a valid code
    if p_icd9.match(x): 
    	return True 
    return False

def isICD9(x):  # 129. is not valid (incomplete)
    if p_icd9_v2.match(x): 
        return True 
    return False     

def isICD9Base(x):
    if x.find('.') > 0: return False
    if p_icd9_base.match(x): 
        return True
    return False

def isICD10(x): 
    if p_icd10.match(x): 
        return True 
    return False

def isICD10Base(x): 
    if x.find('.') > 0: return False 
    if p_icd10_base.match(x): 
        return True 
    return False

def isICDBase(x): 
    return isICD9Base(x) or isICD10Base(x)

def convert(x, nocatch=True, n_base=3): # convert digit string to valid code
    """
    Attempt to correct an ill-formatted code.  

    Memo
    ----
    1. Example codes that cannot be fixed: 

       ['920.011', '.23', 'U23', 'U23.512A']

       two dots 
          72..30 

    """
    x = str(x)
    x = x.strip()
    code = x 
    dotpos = n_base
    err_level = 0
    msg = ''

    if x is None or len(x) == 0: 
        msg += "Invalid ICD (Null code: %s)" % x 
        err_level += 1 

        if nocatch: 
            print(msg) 
        else: 
            raise ValueError, msg
        return x # nothing to fix

    # try to fix lack of digits (e.g. 89.01 => 089.01)
    dpos = x.find('.')
    is_valid = True
    if dpos > 0: 
        base = sub = ''
        try: 
            # base, sub = x.split('.')
            base, sub = x[:dpos], x[dpos+1:]
        except: 
            err_level += 1 
            msg += "\nInvalid ICD format: %s" % x 

        # zero fill if not a V code or E code
        if err_level == 0: 
            if not base.lower().startswith(('e', 'v')): 
                base = base.zfill(n_base)  # 89(.11) => 089(.11)
        else: 
            pass 
            # base remains ''

        if len(base) == n_base:  # base is okay
            sub = sub.replace('.', '')  # fix multiple dots problem
            if len(sub) > 0: 
                code = '%s.%s' % (base, sub)
            else: 
                code = base # e.g. 491. => 491 without decimals
        else: 
            msg += "\nInvalid ICD format (no base): %s" % x 
            print msg 
            err_level += 1 

        print('  + (with dot) %s => %s' % (x, code))    
    elif x.find('.') == 0:
        msg += "\nInvalid ICD (missing first 3 digits): %s" % x
        print msg 
        err_level += 1 
    else:  
        # goal: 08901 => 089.01 

        x = x.translate(None, string.punctuation)  
        # print('  +++ input %s => %s' % (x0, x)) # +++ input 89.01 => 8901

        if x.find('.') < 0: 
            if len(x) > n_base: 
                base, sub = x[:n_base], x[n_base:]
                # cat = cat.zfill(n_base) 
                code = base + '.' + sub   # e.g. 4912
            else: 
                code = x  # e.g. 491
        else: 
            err_level += 1 
            msg += "\nInvalid ICD (impossible case: %s)" % x
        # # incomplete? 
        # if x[-1] == '.': 
        #     code = x[:-1]
        # else: 
        #     code = x

    if not isICD(code): 
        msg += '\nInput %s (=> %s) is not a valid diagnostic code.' % (x, code)

        # final attempt
        # x = x.translate(None, string.punctuation)  
    if err_level > 0: 
        if nocatch: 
            print(msg)
        else: 
            raise ValueError, msg

    return code

def containsBaseForm(codes): 
    """
    True if any element in 'alist' is a (diagnostic) 
    code in its base form only. 
    """
    has_base_only = False
    for c in codes: 
        if isICDBase(c): 
            has_base_only = True
            break
    return has_base_only

def isICD0(x): # icd9 or icd10
    """
    Older version of isICD 
    """
    if isICD9(x): # 129. is valid (even though it's incomplete)
    	return True 
    if isICD10(x):
    	return True
    return False

def isPrefixedICD(x): 
    pass

def isICD(x): 
    if isICD9(x): # 129. is not valid (incomplete)
        return True 
    if isICD10(x):
        return True
    return False    

# def isICDv2(x):  # isICD is now updated to this regex version ... 08.17
#     if isICD9(x): 
#         return True 
#     if isICD10(x):
#         return True
#     return False

def isWord(x): 
    if p_alpha.match(x): 
    	return True
    return False

def isICD9Sub(x): 
    if p_icd9_sub.match(x): 
    	return True
    return False

def isICD10Sub(x):
    if p_icd10_sub.match(x): 
    	return True
    return False

def isMedCode(x): # raw representation ()
    if p_medcode2.match(x):  # check first if the code is prefixed by {MED, MULTUM, NDC, }
        return True
    # if p_medcode.match(x): 
    #     return True
    if isPrefixedDrugSourceVal(x): 
        print('warning> unusual med code format: %s' % x)
        return True
    return False 
def isMed(x): # is medication (internal coded representation including customized prefix such as 'prescription')
    return isMedCode(x)
def isMedication(x):
    return isMedCode(x) 

# [old]   
def isMedication0(x): 
    if x.startswith(prefix_prescription): # drug or medical instructions 
        return True 
    if isMedCode(x): 
        return True 
    return False 

def isMED(x, sep=':'): # is a code from medical entities dictionary (MED)
    """
    Check if the input x is a MED code; MED is a type of med code 
    among at least two other known types i.e. MULTUM, NDC. 
    """
    alist = x.split(sep)  # assuming that x is a string (and white spaces are trimmed)
    prefix = code = None

    if len(alist) == 2: 
        prefix, code = alist[0], alist[1]
    elif len(alist) == 1: 
        code = alist[0]
    else: 
        raise ValueError, "Invalid code input: %s" % str(x)
    
    # check prefix 
    if prefix and not prefix.lower() == 'med': 
        return False
    if p_MED.match(code): 
        return True
    return False

def isCondition(x):
    # if x.startswith(prefix_condition): 
    #     return True
    if isICD(x): # v2 uses a better regex
        return True 
    return False

def t_predicate(**kargs): 
    
    codes = ['920.0', '920', 'S53.521A', 'S53521A', '4031', '52495', '715.,9', ' 524,95  ', '524955', '52495511', '89.01', '920.']
    for code in codes: 
        print '> (input) %s -> (valid) %s' % (code, convert(code, nocatch=True))
    
    code = '524.95511'
    print('> is %s a valid icd9? %s' % (code, isICD9(code)))

    return

def t_transform(**kargs): 
    codes = ['920.0', '920', 'S53.521A', 'S53521A', '4031', '52495', '715.,9', 'V25..23', 'V25.01', 'E334.5', ]
    print('  + before:\n%s\n' % codes)
    for i, code in enumerate(codes): 
        codes[i] = convert(code, nocatch=True, n_base=3)
    print('  + after:\n%s\n' % codes)

    return

def test(**kargs): 

    # [note] S53. is considered valid but 920. is not  ... 
    # codes = ['920.0', '920', 'S53.521A', 's53.521a', 'S53', 'xyz', '1126', '89.01', 'S53.', '920.']
    codes = []
    icd9x = ['920.0', '920.01', '920.011', '920', '89.01', '920.', '13..41', '.23']
    icd10x = ['S53.521A', 'S53.521ABAAAAA', 'S53', 'S53.', 'E23', 'U23', 'U23.512A']
    codes.extend(icd9x)
    codes.extend(icd10x)

    # '89.01' is not valid because only 2 digits in base
    # S53.521ABAAAAA, valid? True
    # S53, valid? True
    # S53., valid? True
    for code in codes: 
        print('> conversion: %s => %s' % (code, convert(code, nocatch=True)))

    print('\n')
    invalids = []
    for code in codes: 
    	print('> code: %s, valid? %s' % (code, isICD(code))) # 
    	print('+           icd9?  %s' % isICD9(code))
    	print('+           icd10? %s' % isICD10(code))
        
        if not isICD(code): 
            invalids.append(code)

    print('\n  + found %d invalid codes: %s\n' % (len(invalids), invalids))

    invalids2 = []
    for code in invalids: 
        code2 = convert(code, nocatch=True)
        if isICD(code2): 
            print('> invalid %s => %s (valid? %s)' % (code, code2, isICD(code2)))
        else: 
            invalids2.append(code2)
    print('\n + found %d invalid codes after conversion: %s\n' % (len(invalids2), invalids2))
            
    t_predicate(**kargs)

    return
def test2(**kargs): 
   
    ### transforming codes 
    t_transform(**kargs)

    return 

if __name__ == "__main__": 
	# test()
    test2()



