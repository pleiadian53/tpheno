import inspect, sys, commands, os
import random

# autovivification
class autovivify_list(dict):
    """Pickleable class to replicate the functionality of collections.defaultdict"""
    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


# system-wise helper functions
def whoami():
    return inspect.stack()[1][3]
def whosdaddy():
    return inspect.stack()[2][3]

def make_hashable(): 
    if isinstance(name, str): return name  # str object doesn't have '__iter__'
    if hasattr(name, '__iter__'): 
        return '_'.join(str(e) for e in name)
    return str(name)

### Function tools ### 
def arg(names, default=None, **kargs): 
    val, has_value = default, False
    if hasattr(names, '__iter__'): 
        for name in names: 
            try: 
                val = kargs[name]
                has_value = True 
                break
            except: 
                pass 
    else: 
        try: 
            val = kargs[names]
            has_value = True
        except: 
            print('warning> Invalid key value: %s' % str(names))

    if not has_value:    
        print('warning> None of the keys were given: %s' % names) 
    return val


### Search Tools ### 

def search(file_patterns=None, basedir=None): 
    """

    Memo
    ----
    1. graphic patterns 
       ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']

    """
    import fnmatch

    if file_patterns is None: 
        file_patterns = ['*.dat', '*.csv', ]
    if basedir is None: 
        basedir = os.getcwd()

    matches = []

    for root, dirnames, filenames in os.walk(basedir):
        for extensions in file_patterns:
            for filename in fnmatch.filter(filenames, extensions):
                matches.append(os.path.join(root, filename))
    return matches

### Formatting tools 

def indent(message, nfill=6, char=' ', mode='r'): 
    if mode.startswith('r'): # left padding 
        return message.rjust(len(message)+nfill, char)
    return message.ljust(len(message)+nfill, char)

def div(message=None, symbol='=', prefix=None, n=80, adaptive=False, border=0, offset=5): 
    output = ''
    if border is not None: output = '\n' * border
    # output += symbol * n + '\n'
    if message:
        if prefix: 
            line = '%s: %s\n' % (prefix, message)
        else: 
            line = '%s\n' % message
        if adaptive: n = len(line)+offset 

        output += symbol * n + '\n' + line + symbol * n
    else: 
        output += symbol * n
        
    if border is not None: 
        output += '\n' * border
    print(output)
    return

# resursively disply a dictionary or any iterable objects
def dumpclean(obj, n_sep=0):
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                if n_sep:
                    blank_lines = '\n' * n_sep
                    print "%s%s" % (blank_lines, k)
                else: 
                    print k
                dumpclean(v)
            else:
                print '%s : %s' % (k, v)
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print v
    else:
        print obj
def display(obj, n_sep=0): 
    return dumpclean(obj, n_sep=n_sep) 

def listToSting(alist, group_size=10, print_=False, sep=','):
    """


    Arguments
    ---------
    * group_size: output this many elements at a time

    """
    if isinstance(alist, str): 
        alist = alist.split(sep) 

    msg = '[' 
    q = "'"
    for i, e in enumerate(alist): 
        msg += q+e+q + sep
        if i > 0 and i % group_size == 0: msg += '\n'
    msg = msg[:-1] + ']'
    if print_: 
        print msg 
    return msg

def execute(cmd): 
    """
    
    [note]
    1. running qrymed with this command returns a 'list' of medcodes or None if 
       empty set
    """

    st, output = commands.getstatusoutput(cmd)
    if st != 0:
        raise RuntimeError, "Could not exec %s" % cmd
    #print "[debug] output: %s" % output
    return output  #<format?>

def fverify(**kargs):
    import inspect 
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print '> function name "%s"' % inspect.getframeinfo(frame)[2]
    for i in args:
        print "    %s = %s" % (i, values[i])
    return 

def multi_delete(alist, idx):
    """
    Remove elements in *alist where elements to remove are indexed by *idx
    """
    indexes = sorted(list(idx), reverse=True)
    for index in indexes:
        del list_[index]
    return list_

def file_path(file_, default_root=None, verify_=True):
    root, fname = os.path.dirname(file_), os.path.basename(file_)
    if not root: 
        if default_root is not None and os.path.exists(default_root): 
            root = default_root
    path = os.path.join(root, fname)
    if verify_: assert os.path.exists(path)
    return path

# [algorithm]
def lcs(S,T):
    """
    Find longest common substring
    """
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set

def size_hashtable(adict): 
    return sum(len(v) for k, v in adict.items())

def sample_dict(adict, n_sample=10): 
    """
    Get a sampled subset of the dictionary. 
    """
    import random 
    keys = adict.keys() 
    n = len(keys)
    keys = random.sample(keys, min(n_sample, n))
    return {k: adict[k] for k in keys} 

def sample_subset(x, n_sample=10):
    if len(x) == 0: return x
    if isinstance(x, dict): return sample_dict(x, n_sample=n_sample)
    
    # assume [(), (), ] 
    return random.sample(x, n_sample)

def pair_to_hashtable(zlist, key=1):
    vid = 1-key 
    adict = {}
    for e in zlist:
        if not adict.has_key(e[key]):   
            adict[e[key]] = [] 
        adict[e[key]].append(e[vid])    
    return adict

def sample_hashtable(hashtable, n_sample=10):
    import random, gc, copy
    from itertools import cycle

    n_sampled = 0
    tb = copy.deepcopy(hashtable)
    R = tb.keys(); random.shuffle(R)
    nT = sum([len(v) for v in tb.values()])
    print('sample_hashtable> Total keys: %d, members: %d' % (len(R), nT))
    
    n_cases = n_sample 
    candidates = set()

    for e in cycle(R):
        if n_sampled >= n_cases or len(candidates) >= nT: break 
        entry = tb[e]
        if entry: 
            v = random.sample(entry, 1)
            candidates.update(v)
            entry.remove(v[0])
            n_sampled += 1

    return candidates

def dictToList(adict):
    lists = []
    for k, v in nested_dict_iter(adict): 
        alist = []
        if not hasattr(k, '__iter__'): k = [k, ]
        if not hasattr(v, '__iter__'): v = [v, ]
        alist.extend(k)
        alist.extend(v)
        lists.append(alist)
    return lists

def nested_dict_iter(nested):
    import collections

    for key, value in nested.iteritems():
        if isinstance(value, collections.Mapping):
            for inner_key, inner_value in nested_dict_iter(value):
                yield inner_key, inner_value
        else:
            yield key, value

def dictSize(adict): # yeah, size matters  
    return len(list(nested_dict_iter(adict)))
def size_dict(adict): 
    """

    Note
    ----
    1. size_hashtable()
    """
    return len(list(nested_dict_iter(adict)))


def partition(lst, n):
    """
    Partition a list into almost equal intervals as much as possible. 
    """
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in xrange(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in xrange(n)]

def divide_interval(total, n_parts):
    pl = [0] * n_parts
    for i in range(n_parts): 
        pl[i] = total // n_parts    # integer division

    # divide up the remainder
    r = total % n_parts
    for j in range(r): 
        pl[j] += 1

    return pl 

def combinations(iterable, r):
    """

    Memo
    ----
    itertools.combinations

    Reference
    ---------
    https://docs.python.org/2/library/itertools.html#itertools.combinations
    """
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

# unbuffer output
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def test(): 
    S, T = ('mixed50-pp-5.csv', 'mixed50-pp-10.csv')
    x = lcs(S, T)
    print list(x)[0].rstrip('-') + '.csv'

def t_iter():
    nested = {'a':{'b':{'c':1, ('d', 'dog'):2}, 
             'e':{'f':3, 'g':4}}, 
             'h':{('i', 'ihop', 1000):(5, 'five'), 'j':6, 'k': {'l': (7, 'seven', 'eleven'), 'm': 8, 'o':{'p': 9} }}}
    l = list(nested_dict_iter(nested))
    l.sort()
    print('info> size: %d, flat:\n%s\n' % (len(l), l))

    lx = dictToList(nested)
    lx.sort()
    print('info> lists:\n%s\n' % lx)

    # size 
    print('info> size of dictionary: %d' % dictSize(nested))

    hashtb = {'a': range(10), 'b': range(100), ('a', 'b'): ('c', 'd', 'e')}
    print('info> size of hashtable: %d' % size_hashtable(hashtb))

    nested2 = {5: [('a', 5), ('b', 7)], 10: [('a', 1), ('b', -2), ('c', 10)]}
    print('info> size of hashtable: %d' % size_hashtable(nested2))
    print('info> size of dictionary: %d' % dictSize(nested2))

    return

def t_format(): 
    codes = ['047.8', '112.2', '038.10', '038.11', '112.5', '047.9', '090.9', '135', '041.9', '041.6', '041.7', '138', '001.1', '017.00', '112.4', '123.1', '003.0', '094.9', '098.0', '088.81', '054.2', '070.71', '038.19', '008.45', '010.10', '133.0', '040.82', '481', '090.1', '027.0', '041.3', '131.01', '041.89', '041.85', '049.9', '009.2', '009.3', '009.0', '009.1', '038.2', '038.3', '038.0', '011.93', '117.5', '038.8', '117.9', '054.10', '041.19', '136.3', '136.9', '041.11', '031.2', '031.0', '031.9', '031.8', '112.3', '033.9', '041.02', '041.01', '041.00', '079.0', '079.6', '041.09', '079.4', '054.13', '070.51', '007.1', '061', '070.32', '070.30', '117.3', '046.3', '038.43', '038.42', '038.40', '054.79', '053.19', '110.0', '110.3', '137.0', '075', '038.49', '057.9', '112.89', '112.84', '097.9', '097.1', '078.5', '078.0', '070.70', '054.3', '099.9', '127.4', '091.3', '005.9', '041.10', '053.9', '054.11', '083.2', '054.19', '034.0', '052.7', '130.7', '036.0', '130.0', '008.69', '053.79', '087.9', '008.61', '111.9']
    l = listToSting(alist=codes)
    print l 

def test3(): 
    # t_format()
    import string, random
    chars = []
    n = 10
    while n > 0: 
        chars.append(random.choice(string.letters))
        n -= 1

    pars = partition(chars, 3)
    for p in pars: 
        print('size: %d => %s' % (len(p), p))

    return

def test2(): 
    div(message='Hey you ~ what would you like me to do?', symbol='*', border=1)
    current = ['047.8','112.2','038.10','038.11','112.5','047.9','038.19','090.9','135','041.9','041.6',
'090.1','138','041.3','001.1','017.00','011.93','112.4','003.0','094.9','008.45',
'054.2','070.71','052.7','088.81','041.7','027.0','131.01','041.89','041.85','049.9',
'046.3','009.2','009.3','009.0','009.1','038.2','117.3','038.0','091.3','117.5',
'038.8','117.9','054.10','041.19','136.3','041.10','041.11','031.2','031.0','031.9',
'031.8','112.3','033.9','041.02','041.01','041.00','079.0','079.6','041.09','079.4',
'054.13','070.51','007.1','070.32','070.30','038.3','038.49','038.43','038.42','038.40',
'054.79','053.19','110.0','110.3','137.0','075','057.9','112.89','112.84','097.9',
'097.1','078.5','078.0','070.70','054.3','099.9','127.4','005.9','136.9','053.9',
'054.11','083.2','054.19','481','130.7','036.0','130.0','008.69','053.79','087.9',
'008.61','111.9']

    batch1 = ['078.2', '083.2', '017.00', '057.9', '033.9', '127.4','098.83','005.81','090.1','088.81','033.9','011.10','110.0','132.0']
    batch2 = ['098.0', '077.8', '074.8', '072.9', '010.10', '132.1', '058.10', '055.9', '034.0', '061', 
           '066.3', '084.6', '045.90', '129', '123.1', '040.82', '133.0', '026.9']

    print('n_current: %d' % len(current))
    otra = '112.1,112.0,112.9,072.9,096,056.9,041.8,098.86,041.4,041.5,041.2,041.0,011.12,091.0,026.9,001.9,091.9,123.1,003.1,074.0,003.9,074.8,077.99,098.0,008.6,098.2,054.0,054.6,008.8,099.40,099.41,052.9,129,088.82,057.0,039.9,008.43,010.10,131.9,039.1,133.0,079.53,040.82,099.50,099.53,099.55,099.54,039.8,090.2,035,092.9,010.01,010.00,041.1,094.0,131.00,079.51,079.83,041.86,131.09,079.88,079.89,049.8,048,042,038.1,038.9,094.89,136.1,136.8,031.1,079.98,066.3,139.8,033.0,070.54,041.04,041.03,074.3,079.2,079.1,070.22,054.40,054.43,007.4,045.90,007.2,070.59,061,078.19,077.8,070.31,078.10,078.11,004.9,046.1,038.44,038.41,058.10,053.12,053.11,084.0,084.6,110.1,070.41,110.2,110.5,110.4,110.9,110.8,054.8,134.0,054.9,010.90,057.8,078.89,078.88,040.0,055.9,112.81,078.8,097.0,078.2,078.1,111.0,002.0,127.2,099.1,099.0,099.3,054.12,053.21,070.3,053.0,034.0,034.1,130.9,111.8,036.2,132.9,088.8,008.62,132.2,132.1,132.0,088.0'
    # additional = ['078.2', '083.2', '017.00', '057.9', '033.9', '127.4','098.83','005.81','090.1','088.81','033.9','011.10','110.0','132.0']
    # allcodes = set(current).union(additional) 
    sumcodes = current + batch1 + batch2
    allcodes = set(sumcodes)
    print('sum of allcodes: %d =? %d' % (len(sumcodes), len(allcodes)))
    print('n_all: %d' % len(allcodes))
    l = listToSting(alist=allcodes)
    print l 

def test4(): 
    # iteration utility 
    t_iter()

    print('=' * 80)
    div(message='test message', symbol='*', border=2)

    return

if __name__ == "__main__":
    # test3()
    test4()



