####################################
#
# File I/O Utilities 
# 
import os, sys, re
import pandas as pd 
from pandas import DataFrame, Series

try:
    import cPickle as pickle
except:
    import pickle

def get_testdir(prefix=None, topdir='test'): 
    """
    Get module-wise test directory, which is located at <module>/<testdir> 
    where <module> in this case is 'seqmaker'
    """
    if prefix is None: prefix = os.getcwd()
    testdir = os.path.join(prefix, topdir)
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory
    return testdir

def get_basedir(prefix=None, suffix=None, topdir='data'): 
    if prefix is None: prefix = os.getcwd()
    if suffix is not None: 
        topdir = os.path.join(topdir, suffix)  # <topdir>/<suffix> on unix
    basedir = os.path.join(prefix, topdir) # [I/O] sys_config.read('DataExpRoot')
    if not os.path.exists(basedir): os.makedirs(basedir) # base directory
    return basedir

def splitext(path):
    """splitext for paths with directories that may contain dots."""
    li = []
    path_without_extensions = os.path.join(os.path.dirname(path), os.path.basename(path).split(os.extsep)[0])
    extensions = os.path.basename(path).split(os.extsep)[1:]
    li.append(path_without_extensions)
    # li.append(extensions) if you want extensions in another list inside the list that is returned.
    li.extend(extensions)
    return li

def save(obj, *args, **kargs):
    # e.g. cluster-tfidf-PV.csv
    outputdir = kargs.get('basedir', kargs.get('outputdir', os.getcwd())) 
    if not os.path.exists(outputdir):
        print('io> create new dir: %s' % outputdir) 
        os.makedirs(outputdir) 

    ext = kargs.get('ext', 'csv')
    identifier = 'test'
    id_sep = '-'

    if args: 
        identifier = args[0]
        for arg in args[1:]: 
            identifier += "%s%s" % (id_sep, str(arg))
    fname = '%s%s%s.%s' % (base, id_sep, identifier, ext)
    fpath = os.path.join(outputdir, fname)
    print('io> saving to %s' % fpath)

    if ext in ('csv', ): 
        sep = kargs.get('sep', kargs.get('delimit', ','))
        header = kargs.get('header', True)
        index = kargs.get('index', False)
        obj.to_csv(fpath, sep=sep, index=index, header=header)  # [todo]
    elif ext in ('pkl', ): 
        pickle.dump(obj, open(fpath, "wb" ))
    else: 
        raise NotImplementedError

    return

def test_split(): 
	p = '/path.with//dots./filename.ext1.ext2'
	pl = splitext(p)
	print "parts: %s" % pl

def test():
	test_split()

if __name__ == "__main__": 
	test()