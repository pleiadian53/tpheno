# encoding: utf-8

import numpy as np
import scipy.stats
from scipy.interpolate import UnivariateSpline

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import os
import itertools

# configure global output dir
Basedir =os.path.join(os.getcwd(), 'test') # sys_config.read('')
if not os.path.exists(Basedir): os.mkdir(Basedir)

##############################################################################################
#
#  Module that extends beyond basic utils defined in sampling (e.g. pdf, plotting, etc.)
#  See doc/utils_pdf.txt for documentation
# 
#  Memo
#  ---- 
#  1. Subdirectories that hold graphics
#     plot (to keep) or test (for test only)
# 
#
##############################################################################################


def g_rv_data(**kargs):  # g: generate 
    # import scipy.stats

    # generate data samples
    # scipy.stats.expon: An exponential continuous random variable.
    data = scipy.stats.expon.rvs(loc=0, scale=1, size=1000, random_state=123)

    return


def p_func(**kargs):  # p: plot 
    def my_dist(x):
        return np.exp(-x ** 2)
    
    x = np.arange(-100, 100)
    p = my_dist(x)

    plt.clf()
    plt.plot(x, p) 
    plt.show()

    return

def t_smooth_hist_beta(**kargs):
    from scipy.stats import beta

    pdf_name = 'beta'
    graph_ext = 'tif'

    plt.clf()
    fig, ax = plt.subplots(1, 1)  

    a, b = 2.30984964515, 0.62687954301

    # first few moments
    mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

    # pdf 
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')

    # freeze RV 
    rv = beta(a, b)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    # check accuracy of cdf and ppf 
    vals = beta.ppf([0.001, 0.5, 0.999], a, b)
    print('cdf consist with ppf? %s' % np.allclose([0.001, 0.5, 0.999], beta.cdf(vals, a, b)))

    # [data] generate random numbers 
    r = beta.rvs(a, b, size=1000)

    # histogram 
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=False)
    # plt.show()

    fpath = os.path.join(Basedir, 'smooth_hist_%s.%s' % (pdf_name, graph_ext))
    plt.savefig(fpath, bbox_inches='tight') 
    plt.close()

    return 

def p_beta(**kargs):
    from scipy.stats import beta

    pdf_name = 'beta'
    graph_ext = 'tif'

    plt.clf()
    fig, ax = plt.subplots(1, 1)  

    # params = set(itertools.product(range(1,51), repeat=2))
    params = list((1,1.0+y*5.0) for y in range(6))
    n_params = len(params)
    legend_params = []
    for i, (a, b) in enumerate(params):     
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        
        # ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
        # ax.plot(x, beta.pdf(x, a, b), lw=1, alpha=0.6, label='beta pdf')

        rv = beta(a, b)
        ax.plot(x, rv.pdf(x), lw=1, label='frozen pdf')
        
        if i % 2 == 1 or i == 0 or i == n_params-1: 
            legend_params.append('(%f, %f)' % (a, b))

    ax.legend(legend_params, loc='upper left')
    fpath = os.path.join(Basedir, 'test_%s-1.%s' % (pdf_name, graph_ext))
    plt.savefig(fpath, bbox_inches='tight') 
    plt.close()

    return

def t_smooth_hist_expo(**kargs): 
    """
    Generate exponential RV and plot smoothed histogram
    
    Memo
    ----
    1. ppf: Percent point function (inverse of cdf: percentiles).

    2. parameter estimate 
       fit(data, loc=0, scale=1)	Parameter estimates for generic data.

    3. survival function: 1 - cdf 
       sf(x, loc=0, scale=1)	Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

    4. differential entropy 
       entropy(loc=0, scale=1)	(Differential) entropy of the RV.

    5. variance of the distribution 
       var(loc=0, scale=1)	Variance of the distribution.

    6. alpha percent interval 
       interval(alpha, loc=0, scale=1)	Endpoints of the range that contains alpha percent of the distribution

    """
    from scipy.stats import expon   # exponential continuous random variable. expon.pdf(x) = exp(-x)

    fig, ax = plt.subplots(1, 1)    # subplotting 

    # compute first few moments 
    mean, var, skew, kurt = expon.stats(moments='mvsk')

    # display pdf
    x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), 100)
    ax.plot(x, expon.pdf(x), 'r-', lw=5, alpha=0.6, label='expon pdf')


    ### frozen RV 
    # Alternatively, the distribution object can be called (as a function) to fix the shape, 
    # location and scale parameters. This returns a “frozen” RV object holding the given parameters fixed.

    # Freeze the distribution and display the frozen pdf
    rv = expon()
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    # Check accuracy of cdf and ppf
    vals = expon.ppf([0.001, 0.5, 0.999])
    np.allclose([0.001, 0.5, 0.999], expon.cdf(vals))

    # [data] generate random numbers 
    r = expon.rvs(size=1000)

    # compute histogram 
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=False)
    # plt.show()

    fpath = os.path.join(Basedir, 'smooth_hist_expo.tif')
    plt.savefig(fpath, bbox_inches='tight') 
    plt.close()

    return

def t_smooth_hist(**kargs): # t: template
    """

    Memo
    ----
    1. numpy histogram 
       density: 
   
       weights:

    """
    # from scipy.interpolate import UnivariateSpline

    basedir = kargs.get('output_dir', os.path.join(os.getcwd(), 'test')) # sys_config.read('')
    if not os.path.exists(basedir): os.mkdir(basedir)

    N = 1000
    n = N/10
    s = np.random.normal(size=N)   # generate your data sample with N elements

    # p: histogram values (e.g. number of samples in each bin) see density and weights
    # x: edges
    p, x = np.histogram(s, bins=n) # bin it into n = N/10 bins

    # x[:-1]: all but the last
    x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
    f = UnivariateSpline(x, p, s=n)

    # plotting
    fpath = os.path.join(basedir, 'smooth_hist.tif')

    plt.clf()
    plt.plot(x, f(x))
    # plt.show()
    plt.savefig(fpath, bbox_inches='tight') 
    plt.close()

    return

def t_kde(**kargs): 
    """

    Memo
    ----
    [1] gaussian_kde uses a changable function, covariance_factor to calculate it's bandwidth. 
        Before changing the function, the value returned by covariance_factor for this data was about .5. 
        Lowering this lowered the bandwidth. I had to call _compute_covariance after changing that function so that 
        all of the factors would be calculated correctly.

    """
    # gaussian kernel density estimate 

    from scipy.stats.kde import gaussian_kde
    from numpy import linspace

    basedir = kargs.get('output_dir', os.path.join(os.getcwd(), 'test')) # sys_config.read('')
    if not os.path.exists(basedir): os.mkdir(basedir)

    # create fake data
    data = randn(1000)
    
    # this create the kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde( data )
 
    # these are the values over wich your kernel will be evaluated
    dist_space = linspace( min(data), max(data), 100 )
    
    # plot the results
    plt.clf()
    fpath = os.path.join(basedir, 'gaussian_kde_example.tif')
    plt.plot( dist_space, kde(dist_space) )

    # plt.show()
    plt.savefig(fpath, bbox_inches='tight') 
    # plt.close()

    ### 2. changing the bandwith param 
    plt.clf()

    density = gaussian_kde(data)   # gaussian_kde tries to infer the bandwidth automatically
    xs = np.linspace(min(data), max(data), 200)
    
    density.covariance_factor = lambda : .25  # [1]
    density._compute_covariance()
    plt.plot(xs,density(xs))
    # plt.show()
    
    fpath = os.path.join(basedir, 'gaussian_kde_bandwith.tif')
    plt.savefig(fpath, bbox_inches='tight') 
    plt.close()

    return

def t_kde_expo(**kargs):

    return 


def test(**kargs): 
    # t_smooth_hist(**kargs)

    # smoothed histogram using expoential RV as an example 
    t_smooth_hist_expo(**kargs)

    # beta distribution 
    p_beta(**kargs)

    return 

if __name__ == "__main__":
    test()




