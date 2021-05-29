import sys
from EM_twodirection_inference import EM_algo, initialize_params, calc_expected_states
import numpy as np
from scipy.stats import multivariate_normal
import copy

def fitModel_2d_util(y, max_bins=4):
    """
    Allows you to call the one-directional EM model
    Parameters
    ----------
    y : 2D Array (nsamples x ndim)
        Array containing all of the samples
    max_bins : int, optional
        Longest number of mixtures you want to test. The default is 4 (corresponds to 4 by 2 model).
    Returns
    -------
    list
        The fitted model for each choice of mixtures 
    """
    ntaxa = len(y[0])
    best_dataset = None
    best_bins = []
    best_ds = []
    for K in range(2, max_bins+1):
        highest_ll=None
        max_b = 200
        np.random.seed(5)
        for b in range(max_b):
            print(b)
            results = EM_algo(y, K)
            mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = results
            Z = np.zeros((y.shape[0],2*K))
            for n in range(y.shape[0]):
                for k in range(2*K):
                    Z[n,k]=mixing[k]*multivariate_normal.pdf(y[n], mu[k], sigma[k])
            best_bins = np.argmax(Z,axis=1)
            print(best_bins) #best_bins is the bin assignments 
            if highest_ll is None or likelihoods[-1] > highest_ll:
                #choose model by best likelihood
                best_dataset = copy.deepcopy(results)
                highest_ll = likelihoods[-1]
            print(highest_ll)
        best_ds.append(best_dataset)
    sys.stdout.flush()
    return best_ds