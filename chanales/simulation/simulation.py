import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, invgamma, invwishart, wishart
import pickle
import sys
import pymc3 as pm
import theano.tensor as tt
import argparse

from pymc3.model import modelcontext
from scipy import dot
import scipy.stats as st
from scipy.linalg import cholesky as chol
import warnings
import arviz as az
import os

def sample_gmm(means, covars, 
               n_samples = 1000, 
               weights = [0.5,0.5]):
    rng = np.random.mtrand._rand
    n_samples_comp = rng.multinomial(n_samples, weights)
    dat = np.vstack([
                rng.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    means, covars, n_samples_comp)])
    return dat
    
def sample_mvn(mean, cov, n_samples = 10000):
    rng = np.random.mtrand._rand
    dat = rng.multivariate_normal(mean, cov, n_samples)
    return dat

def fit_mvn_model(dat, n_samples):
    p = 0.5
    nu_0, inv_lambda_0, mu_0, kappa_0 = (5, np.eye(2), np.array([-2, -2]), 1)
    model = pm.Model()
    with model:
        #comp = pm.Bernoulli("comp", p)

        sd_dim1 = pm.Exponential('sd1', 0.25, shape=1)
        sd_dim2 = pm.Exponential('sd2', 0.25, shape=1)
        mean = pm.MvNormal('mean', mu = mu_0,  cov=np.eye(2), shape=2)
        vals = pm.MvNormal('vals', mu=mean, chol = tt.diag(tt.concatenate([sd_dim1,sd_dim2])), observed=dat)

        trace = pm.sample(n_samples, cores=4, chains=4, progressbar=True)
    return trace, model

def fit_gmm_model(dat, n_samples):
    model = pm.Model()
    k = 2
    ndata = dat.shape[0]
    nu_0, inv_lambda_0, mu_0, kappa_0 = (5, np.eye(2), np.array([0, -3.5]), 1)
    nu_1, inv_lambda_1, mu_1, kappa_1 = (5, np.eye(2), np.array([-3.5, 0]), 1)
    with model:
        # cluster centers
        meandim1 = pm.Normal("meandim1", mu=0, sigma=np.array([1]))
        meandim2 = pm.Normal("meandim2", mu=-3.5, sigma=np.array([1]))

        # measurement error
        sd_dim1 = pm.Exponential('sd1', 0.25, shape=1)
        sd_dim2 = pm.Exponential('sd2', 0.25, shape=1)

        comp1 = pm.MvNormal.dist( mu = [meandim1, meandim2], chol = tt.diag(tt.concatenate([sd_dim1,sd_dim2])))
        comp2 = pm.MvNormal.dist( mu = [meandim2, meandim1], chol = tt.diag(tt.concatenate([sd_dim2,sd_dim1])))

        # latent cluster of each observation
        #category = pm.Categorical("category", p=p, shape=ndata)

        # likelihood for each observed value
        #points = pm.Normal("obs", mu=means[category], sigma=sd, observed=dat)
        likelihood = pm.Mixture(name = "lh", w = [0.5, 0.5], comp_dists = [comp1,comp2], observed=dat)

        # trace = pm.sample_smc(5000, parallel=False)
        trace = pm.sample(draws=n_samples, cores=1, chains=4, progressbar=True)
    return trace, model

def Marginal_llk(mtrace, model=None, logp=None, maxiter=1000):
    """The Bridge Sampling Estimator of the Marginal Likelihood.

    Parameters
    ----------
    mtrace : MultiTrace, result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    logp : Model Log-probability function, read from the model by default
    maxiter : Maximum number of iterations

    Returns
    -------
    marg_llk : Estimated Marginal log-Likelihood.
    """
    r0, tol1, tol2 = 0.5, 1e-10, 1e-4

    model = modelcontext(model)
    if logp is None:
        logp = model.logp_array
    vars = model.free_RVs
    print("vars", vars)
    # Split the samples into two parts  
    # Use the first 50% for fiting the proposal distribution and the second 50% 
    # in the iterative scheme.
    len_trace = len(mtrace)
    nchain = mtrace.nchains
    
    N1_ = len_trace // 2
    N1 = N1_*nchain
    N2 = len_trace*nchain - N1

    neff_list = dict() # effective sample size

    arraysz = model.bijection.ordering.size
    samples_4_fit = np.zeros((arraysz, N1))
    samples_4_iter = np.zeros((arraysz, N2))
    # matrix with already transformed samples
    for var in vars:
        varmap = model.bijection.ordering.by_name[var.name]
        # for fitting the proposal
        x = mtrace[:N1_][var.name]
        samples_4_fit[varmap.slc, :] = x.reshape((x.shape[0], 
                                                  np.prod(x.shape[1:], dtype=int))).T
        # for the iterative scheme
        x2 = mtrace[N1_:][var.name]
        samples_4_iter[varmap.slc, :] = x2.reshape((x2.shape[0], 
                                                    np.prod(x2.shape[1:], dtype=int))).T
        # effective sample size of samples_4_iter, scalar
        


    # median effective sample size (scalar)
    esses = az.stats.ess(mtrace[N1_:], method='median')
    neff = np.median(np.array(esses.to_array()).ravel()) #pm.stats.dict2pd(neff_list,'temp').median()
    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    V = np.cov(samples_4_fit)
    L = chol(V, lower=True)

    # Draw N2 samples from the proposal distribution
    gen_samples = m[:, None] + dot(L, st.norm.rvs(0, 1, 
                                         size=samples_4_iter.shape))

    # Evaluate proposal distribution for posterior & generated samples
    q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, V)
    q22 = st.multivariate_normal.logpdf(gen_samples.T, m, V)

    # Evaluate unnormalized posterior for posterior & generated samples
    q11 = np.asarray([logp(point) for point in samples_4_iter.T])
    q21 = np.asarray([logp(point) for point in gen_samples.T])

    # Iterative scheme as proposed in Meng and Wong (1996) to estimate
    # the marginal likelihood
    def iterative_scheme(q11, q12, q21, q22, r0, neff, tol, maxiter, criterion):
        l1 = q11 - q12
        l2 = q21 - q22
        lstar = np.median(l1) # To increase numerical stability, 
                              # subtracting the median of l1 from l1 & l2 later
        s1 = neff/(neff + N2)
        s2 = N2/(neff + N2)
        r = r0
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol

        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            rold = r
            logmlold = logml
            numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
            deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
            if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                warnings.warn("""Infinite value in iterative scheme, returning NaN. 
                Try rerunning with more samples.""")
            r = (N1/N2) * np.sum(numi)/np.sum(deni)
            r_vals.append(r)
            logml = np.log(r) + lstar
            i += 1
            if criterion=='r':
                criterion_val = np.abs((r - rold)/r)
            elif criterion=='logml':
                criterion_val = np.abs((logml - logmlold)/logml)

        if i >= maxiter:
            return dict(logml = np.NaN, niter = i, r_vals = np.asarray(r_vals))
        else:
            return dict(logml = logml, niter = i)

    # Run iterative scheme:
    tmp = iterative_scheme(q11, q12, q21, q22, r0, neff, tol1, maxiter, 'r')
    if ~np.isfinite(tmp['logml']):
        warnings.warn("""logml could not be estimated within maxiter, rerunning with 
                      adjusted starting value. Estimate might be more variable than usual.""")
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'logml')

    return dict(logml = tmp['logml'], niter = tmp['niter'], method = "normal", 
                q11 = q11, q12 = q12, q21 = q21, q22 = q22)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description='Parameter search, Emergent model')
    parser.add_argument('idx', type=int,
                        action='store', help='Name of the parameter search')
    args = parser.parse_args()

    sds = list(range(1, 20))
    sds.insert(0,0.1)
    mvn_traces_new = np.zeros((len(sds),len(sds)))
    gmm_traces_new = np.zeros((len(sds),len(sds)))
    #mvn_traces_new[:19, :19] = mvn_traces
    #gmm_traces_new[:19, :19] = gmm_traces
    for idx, sd1 in enumerate(sds):
        for idy, sd2 in enumerate(sds):
            #if not (sd1 == 19 and sd2 == 19):
            #    continue
            if not (idx == args.idx):
                continue
            if os.path.exists(f'results/mvn_sd1_{sd1}_sd2_{sd2}') or os.path.exists(f'results/gmm_sd1_{sd1}_sd2_{sd2}'):
                continue
            print("sd1", sd1, "sd2", sd2)
            means = np.array([[-3.5,0], [0,-3.5]])
            cov1 = np.diag([sd1**2,sd2**2])
            cov2 = np.diag([sd2**2,sd1**2])
            covars = [cov1, cov2]
            dat = sample_gmm(means, covars,n_samples=50000)
            mvn_trace, mvn_model = fit_mvn_model(dat, n_samples=50000)
            mvn_dic = Marginal_llk(mvn_trace, model=mvn_model)
            mvn_traces_new[idx, idy] = mvn_dic["logml"]
            print("MVN", mvn_dic["logml"])
            gmm_trace, gmm_model = fit_gmm_model(dat, n_samples=50000)
            gmm_dic = Marginal_llk(gmm_trace, model=gmm_model)
            gmm_traces_new[idx, idy] = gmm_dic["logml"]
            print("GMM", gmm_dic["logml"])
            with open(f'results_50000/mvn_sd1_{sd1}_sd2_{sd2}', 'wb') as handle:
                pickle.dump(mvn_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'results_50000/gmm_sd1_{sd1}_sd2_{sd2}', 'wb') as handle:
                pickle.dump(gmm_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            