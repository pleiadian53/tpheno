

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def t1(): 
    """

    Memo
    ----
    1. repr 
           t1  t2  t3 
        d1
        d2
        d3

    """
    corpus = ["This is very strange",
              "This is very nice", 
              "This is a lot better"]
    Xc = np.asarray(corpus)
    print('info> corpus dim: %s' % str(Xc.shape))

    vectorizer = TfidfVectorizer(min_df=1)

    X = vectorizer.fit_transform(corpus)
    print("info> X dim: %s" % str(X.shape)) # 2 by 5 (2 docs, 5 tokens)
    print('  + X:\n%s\n' % X)  #  sparse matrix, [n_samples, n_features]

    idf = vectorizer.idf_
    # [log] {u'this': 1.0, u'very': 1.0, u'is': 1.0, u'strange': 1.4054651081081644, u'nice': 1.4054651081081644}
    adict = dict(zip(vectorizer.get_feature_names(), idf))  
    print '  + feature vs idf:\n%s\n' % adict 

    for w in ['this', 'is', 'very', 'Very', 'meaningful', 'strange', 'nice', 'Nice', 'better']: 
        if adict.has_key(w): 
            print('  + %s => %f' % (w, adict[w]))
        else: 
            print('  + %s => %f' % (w, 0.0))

    return

def test(**kargs): 
    t1()

    return

if __name__ == "__main__": 
    test()