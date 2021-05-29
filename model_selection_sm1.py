# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:46:47 2021

@author: amey_admin
"""

import sys
from test_onedirection import fitModel_1d_util
from test_twodirection import fitModel_2d_util
import numpy as np
from scipy.stats import multivariate_normal
import copy
from multiprocessing import Pool
import pickle as pkl
from skbio.stats.composition import ilr, ilr_inv

def one_d_AIC(model, y):
    mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations = model
    print(np.mean([np.linalg.norm(mu[i]-mu[i-1]) for i in range(1, len(mixing))]))
    log_likelihood=0
    nsamples = len(y)
    num_bins = len(mixing)
    #expected_states = calc_expected_states(num_bins//2, nsamples, y, mu, sigma, mixing)
    for n in range(nsamples):
        ll=0
        for k in range(num_bins):
            ll+=mixing[k]*multivariate_normal.pdf(y[n], mu[k], sigma[k])
        log_likelihood+=np.log(ll)
    ntaxa = len(sigma[0])
    
    num_params = num_bins*((ntaxa**2-ntaxa)/2+2*ntaxa+1)-1
    return num_params*2-2*log_likelihood

def two_d_AIC(model, y):
    mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = model

    log_likelihood=0
    nsamples = len(y)
    num_bins = len(mixing)

    for n in range(nsamples):
        ll=0
        for k in range(num_bins):
            ll+=mixing[k]*multivariate_normal.pdf(y[n], mu[k], sigma[k])
        log_likelihood+=np.log(ll)
    ntaxa = len(sigma[0])
    num_params = num_bins*((ntaxa**2-ntaxa)/2+2*ntaxa+1)-1 #-1 since mixing has num_bins-1 degrees of freedom
    
    return num_params*2-2*log_likelihood

def test_ds(region_identifier=0):
    """
    Fit all models to data, calculate the AIC score for each model to determine best model
    Parameters
    ----------
    region : 2=colon, 1=cecum, 0=intestine, -1 for all regions
        DESCRIPTION. The default is colon.
    Returns the data, basis
    """
    np.random.seed(2191)
    #Load the data
    loc, rel_abun_data, ilr_data, philr_data, basis = pkl.load(open("sm1_data.p", "rb"))
    #loc is if a sample is from the intestine (0), cecum (1), colon (2)
    #rel_abun_data is the relative abundance data
    #ilr_data is the data transformed under the default ilr basis
    #philr_data is the data transformed under the phylogentically derived basis
    #basis is the basis used for philr

    if region_identifier>=0:
        region = np.where(loc==region_identifier)[0] #intestine (0), cecum (1), colon (2)
        philr_data = philr_data[region]

    two_dim = fitModel_2d_util(philr_data)
   
    print([two_d_AIC(two_dim[i], philr_data) for i in range(len(two_dim))])
    one_dim = fitModel_1d_util(philr_data)

    print([one_d_AIC(one_dim[i], philr_data) for i in range(len(one_dim))])
    print([two_d_AIC(two_dim[i], philr_data) for i in range(len(two_dim))])
    
    

    result = philr_data, one_dim, two_dim, basis
    if region_identifier==0:
        pkl.dump(result, open("model_selection_ileum.p", "wb"))
    if region_identifier==1:
        pkl.dump(result, open("model_selection_cecum.p", "wb"))
    if region_identifier==2:
        pkl.dump(result, open("model_selection_colon.p", "wb"))
    return result

if __name__ == "__main__":
    #result = test_colon_phyla()
    result = test_ds(1)
    #result = test_ds(1)
    #result = test_ds(2)
    philr_data, one_dim, two_dim, selection = result
    
    print([two_d_AIC(two_dim[i], philr_data) for i in range(len(two_dim))])
    print([one_d_AIC(one_dim[i], philr_data) for i in range(len(one_dim))])
    