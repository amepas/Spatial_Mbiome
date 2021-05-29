# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:39:45 2020

@author: amey_admin
"""

#from util import parse_data, parse_data_sm1, parse_data_lf
import sys
from simulation_ilr import simulate_gmm, simulate_direction, simulate_twolayer, sim_twolayer_delta
from pick_nec_singledirection import fitModel_1d_util
from pick_nec_doubledirection import fitModel_2d_util
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold, train_test_split
import copy
from multiprocessing import Pool
from sklearn.decomposition import PCA
#from sklearn.metrics.cluster import v_measure_score, homogeneity_score
#from sklearn.cluster import KMeans
import pickle as pkl
#from sklearn import svm
#from sklearn.model_selection import cross_val_score
from skbio.stats.composition import ilr, ilr_inv, sbp_basis
import matplotlib.pyplot as plt
#import scipy.stats
import argparse
from validity_indexes import silhouette 

def one_d_BIC(dataset, y):
    mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations = dataset
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
    #num_params = num_bins*(ntaxa+1)-1
    #num_params = num_bins*(2*ntaxa+1)-1
    return num_params*np.log(nsamples)-2*log_likelihood

def one_d_AIC(dataset, y):
    mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations = dataset
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
    #num_params = num_bins*(ntaxa+1)-1
    #num_params = num_bins*(2*ntaxa+1)-1
    #print(likelihoods[-1]-log_likelihood)
    return num_params*2-2*log_likelihood

def BIC(dataset, y):
    mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = dataset
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
    num_params = num_bins*((ntaxa**2-ntaxa)/2+2*ntaxa+1)-1 #-1 since mixing has num_bins-1 degrees of freedom
    #num_params = num_bins*(ntaxa+1)-1
    #num_params = num_bins*(2*ntaxa+1)-1
    return num_params*np.log(nsamples)-2*log_likelihood

def AIC(dataset, y):
    mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = dataset
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
    num_params = num_bins*((ntaxa**2-ntaxa)/2+2*ntaxa+1)-1 #-1 since mixing has num_bins-1 degrees of freedom
    #num_params = num_bins*(ntaxa+1)-1
    #num_params = num_bins*(2*ntaxa+1)-1
    #print(likelihoods[-1]-log_likelihood)
    return num_params*2-2*log_likelihood

# =============================================================================
# def twod_spatial(dataset):
#     mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = dataset
#     log_likelihood=0
#     log_likelihood += multivariate_normal.logpdf(mu[0], edge_mean.flatten(), Q_edge)
#     log_likelihood += multivariate_normal.logpdf(mu[1], mu[0]+delta_perp, Q)
#     for k in range(1, K):
#         log_likelihood += multivariate_normal.logpdf(mu[2*k], mu[2*(k-1)]+delta, Q) 
#         log_likelihood += multivariate_normal.logpdf(mu[2*k+1], 0.5*(mu[2*k]+mu[2*k-1]+delta+delta_perp), Q)
#     return log_likelihood
# =============================================================================
def entropy(mixing, sigma, mu, y):
        K = len(mu)
        y = np.array(y)
        Z = np.zeros((y.shape[0],K))
        for n in range(y.shape[0]):
            for k in range(K):
                Z[n,k]=mixing[k]*multivariate_normal.pdf(y[n], mu[k], sigma[k])
            Z[n,:]/=np.sum(Z[n,:])
        #print(Z)
        entropy = 0
        #print(Z)
        for n in range(y.shape[0]):
            for k in range(K):
                if Z[n,k]<1e-10:
                    entropy+=0
                else:
                    entropy +=(-Z[n,k]*np.log(Z[n,k]))
        return entropy

def one_d_AWE(dataset, y):
    mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations = dataset
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
    #num_params = num_bins*(ntaxa+1)-1 
    #num_params = num_bins*(ntaxa+1)+(ntaxa**2-ntaxa)/2 + ntaxa-1 
    #num_params = num_bins*(2*ntaxa+1)-1
    #print(np.mean([np.trace(sigma[i]) for i in range(len(sigma))]))
    print(entropy(mixing, sigma,mu, y))
    return 2*num_params*(np.log(nsamples)+1.5)+2*entropy(mixing, sigma, mu, y)-2*log_likelihood

def two_d_AWE(dataset, y):
    mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = dataset
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
    num_params = num_bins*((ntaxa**2-ntaxa)/2+2*ntaxa+1)-1 #-1 since mixing has num_bins-1 degrees of freedom
    #num_params = num_bins*(ntaxa+1)-1 
    #num_params = num_bins*(ntaxa+1)+(ntaxa**2-ntaxa)/2 + ntaxa-1 
    #num_params = num_bins*(2*ntaxa+1)-1
    print(entropy(mixing, sigma,mu, y))
    return 2*num_params*(np.log(nsamples)+1.5)+2*entropy(mixing, sigma, mu)-2*log_likelihood

def sim_ll(y, sigma, mu):
    mixing = [1/len(mu) for i in range(len(mu))]
    log_likelihood=0
    nsamples = len(y)
    num_bins = len(mixing)
    for n in range(nsamples):
        ll=0
        for k in range(num_bins):
            ll+=mixing[k]*multivariate_normal.pdf(y[n], mu[k], sigma[k])
        log_likelihood+=np.log(ll)
    Z = np.zeros((y.shape[0],num_bins))
    for n in range(y.shape[0]):
        for k in range(num_bins):
            Z[n,k]=mixing[k]*multivariate_normal.pdf(y[n], mu[k], sigma[k])
    best_bins = np.argmax(Z,axis=1)
    #print(best_bins)
    return log_likelihood

def test_ds_1d(i, num_bins):
    """
    i defines the seed, num_bins defines the number of bins. We only test 1d datasets here
    """
    np.random.seed(2191+i)
    simulated_dataset = simulate_direction(num_bins, ntaxa=47, nsamples=int(360/num_bins), Sigma_trace=1)
    X, K, sigma, mu = simulated_dataset
    y = np.zeros((X.shape[0]*X.shape[1], np.shape(X)[2])) #reformat data for model
    for i in range(len(X)):
        for j in range(len(X[0])):
            y[X.shape[1]*i+j] = X[i,j]
    no_struc = 1
    one_dim = fitModel_1d_util(y)
    for i in range(2):
        print([one_d_AIC(one_dim[1][i], y) for i in range(len(one_dim[1]))])
    #for i in range(2):
    #    print([one_d_AWE(one_dim[1][i], y) for i in range(len(one_dim[1]))])
    #print("silhouette")
    #for i in range(len(one_dim[1])):
    #    mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations = one_dim[1][i]
    #    print(silhouette(mixing, sigma, mu, y))
    two_dim = fitModel_2d_util(y)
    for i in range(2):
        print([one_d_AIC(one_dim[1][i], y) for i in range(len(one_dim[1]))])
        print([AIC(two_dim[1][i], y) for i in range(len(two_dim[1]))])
    #one_dim_scores = one_dim[0] #Scores start at 2 bins
    #two_dim_scores = two_dim[0]
    selection = 1 #if selection is negative just assume i'm referring to the 2d case
    return simulated_dataset, one_dim, two_dim, selection

def test_ds_2d(i, num_bins):
    """
    i defines the seed, num_bins defines the number of bins. We only test 2d datasets here
    """
    np.random.seed(2191+i)
    #simulated_dataset = simulate_twolayer(num_bins, ntaxa=47, nsamples=int(180/num_bins), Sigma_trace=10, scaling_factor=1)
    simulated_dataset = sim_twolayer_delta(num_bins, ntaxa=47, nsamples=int(180/num_bins), Sigma_trace=1, angle_threshold=np.pi/4, Q_tr=5, scaling_factor=1)
    #print(num_bins)
    X, K, sigma, mu = simulated_dataset
    y = np.zeros((X.shape[0]*X.shape[1], np.shape(X)[2])) #reformat data for model
    for i in range(len(X)):
        for j in range(len(X[0])):
            y[X.shape[1]*i+j] = X[i,j]
    p = PCA(n_components=2)
    y_trans = p.fit_transform(y)
    plt.scatter(y_trans[:,0], y_trans[:,1])
    plt.show()
    print(sim_ll(y, sigma, mu))
    no_struc = 1
    two_dim = fitModel_2d_util(y, 2)
    for i in range(2):
        print([BIC(two_dim[1][i], y) for i in range(len(two_dim[1]))])
        print([AIC(two_dim[1][i], y) for i in range(len(two_dim[1]))])
    one_dim = fitModel_1d_util(y, 6)
    #for i in range(2):
    #    print([one_d_AIC(one_dim[1][i], y) for i in range(len(one_dim[1]))])
    #for i in range(2):
    #    print([one_d_AWE(one_dim[1][i], y) for i in range(len(one_dim[1]))])
    one_dim_scores = one_dim[0] #Scores start at 2 bins
    two_dim_scores = two_dim[0]
    selection = 1 #if selection is negative just assume i'm referring to the 2d case

    for i in range(2):
        print([one_d_AIC(one_dim[1][i], y) for i in range(len(one_dim[1]))])
        print([AIC(two_dim[1][i], y) for i in range(len(two_dim[1]))])
    return simulated_dataset, one_dim, two_dim, selection

def util_1d(seed):
    num_bins=2
    #results = test_ds_1d(seed, num_bins)
    #ds, one_dim, two_dim, selection = results
    
    #pkl.dump(results, open("simulation_experiments/1d_data/300results"+str(seed)+"_"+str(num_bins)+"bins.p", "wb"))
    if num_bins==2:
        ds, one_dim, two_dim, selection = test_ds_1d(seed, num_bins+1)
        results = ds, one_dim, two_dim, selection 
        
        pkl.dump(results, open("simulation_experiments/1d_data/300results"+str(seed)+"_"+str(num_bins+1)+"bins.p", "wb"))

def select_model(aic1, aic2):
    if min(aic1)<min(aic2):
        selection = " by 1"
        return str(aic1.index(min(aic1))+2)+selection
    else:
        selection = " by 2"
        return str(aic2.index(min(aic2))+2)+selection
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", metavar="seed/ds number", type=int)
    parser.add_argument("K", metavar="K", type=int, help="number of bins, must be >2")
    args = parser.parse_args()
    seed = args.seed
    num_bins = args.K
    
    results = test_ds_2d(seed, num_bins)
    ds, one_dim, two_dim, selection = results
    
    #pkl.dump(results, open("simulation_experiments/1d_data/results"+str(seed)+"_"+str(num_bins)+"bins.p", "wb"))
    """
    selections = []
    for i in range(20,50):
        print("dataset: ", i)
        results = pkl.load(open("simulation_experiments/1d_data/results"+str(i)+"_"+str(num_bins)+"bins_SigTr1.p", "rb"))
        simulated_dataset, one_dim, two_dim, selection = results 
        X, K, sigma, mu = simulated_dataset
        y = np.zeros((X.shape[0]*X.shape[1], np.shape(X)[2])) #reformat data for model
        for i in range(len(X)):
            for j in range(len(X[0])):
                y[X.shape[1]*i+j] = X[i,j]
        #print([one_d_AIC(one_dim[1][i], y) for i in range(len(one_dim[1]))])
        #print([AIC(two_dim[1][i], y) for i in range(len(two_dim[1]))])
        selection = select_model([one_d_AIC(one_dim[1][i], y) for i in range(len(one_dim[1]))], [AIC(two_dim[1][i], y) for i in range(len(two_dim[1]))])
        selections.append(selection)
        mixing, sigma, delta, Q, Q_edge, edge_mean, mu, likelihoods, iterations = one_dim[1][2]
        #print(np.trace(Q))
        print(likelihoods[-1])
        #one_d_AIC(one_dim[1][4], y)
        mixing, sigma, delta, delta_perp, Q, Q_edge, edge_mean, mu, likelihoods, iterations = two_dim[1][0]
        print(likelihoods[-1])
        #AIC(two_dim[1][1], y)
    unique, counts = np.unique(selections, return_counts=True)
    """
    
    