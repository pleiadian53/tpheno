

def kl_divergence(p, q): 
    """

    Reference
    ---------
    1. http://scipy.github.io/devdocs/generated/scipy.stats.entropy.html

    2. The bonus of this function as well is that it will normalize the vectors you pass it if they do not sum to 1 
       (though this means you have to be careful with the arrays you pass - ie, how they are constructed from data).

	"""
	scipy.stats.entropy(p, qk=q, base=None)

	return 


def cluster_similarity():
    """
    
    Reference
    ---------
    1. http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html

    """
    pass 