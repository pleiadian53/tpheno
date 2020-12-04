    

class TestDocs(object): 

    @classmethod
    def stratum_stats(cls, units): # <- units
        labels = units.keys() # documents = D; timestamps = T
        n_labels = len(set(labels))

        nD = nT = 0
        sizes = {}
        for label, entry in units.items():
            Di = entry['sequence']
            Ti = entry['timestamp']
            Li = entry['label']
            nD += len(Di)
            nT += len(Ti)
            sizes[label] = len(Di)
            nDi, nTi = len(Di), len(Ti)
            assert nDi == nTi, "nD=%d <> nT=%d" % (nDi, nTi)
        # assert nD == nT, "nD=%d <> nT=%d" % (nD, nT)
        print('TestDocs> nD=%d, nT=%d, n_labels=%d ...\n  + sizes:\n%s\n' % (nD, nT, n_labels, sizes))
        print('  + number of labels: %d' % n_labels)
        return