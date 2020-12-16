'''
Functions for calculating posterior distribution in Laplace Approximation
'''

import gpy
import GPy
import numpy as np
import torch



lik = GPy.likelihoods.Bernoulli()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
def error(x, y):
    return torch.norm(x - y) / torch.norm(x), torch.norm(x - y)

def get_inverse_d2posterior(W, var, G):
    n, m = W.shape[0], G.shape[0]

    W = W + torch.eye(n).to(device) / var
    W = torch.clamp(W, 1e-10, 1e4)
    W_inv = torch.diag(torch.pow(torch.diag(W), -1))
    
    W_inv_G = torch.mm(W_inv, G.T)
    
    middle_factor = torch.eye(m).to(device) + torch.mm(G, G.T) / var - torch.mm(G, W_inv_G) / (var**2)
    middle_factor = torch.inverse(middle_factor)
    
    result = torch.mm(W_inv_G, torch.mm(middle_factor, G))
    result = W_inv + torch.mm(result, W_inv) / (var ** 2)
    
    return result

def posterior(y, likelihood, G, var, *f):
    '''
        Compute posterior parameters with approximations
            y - targets
            G - help matrix from decomposition,
            var - variance,
            f - mean
        returns : f, (K+W^{-1})^{-1} - inverse matrix of second derivative of posterior, W
    '''
    
#     f = torch.zeros_like(y, dtype = torch.float)
    f = torch.normal(torch.zeros_like(y, dtype = torch.float), 0.5 * torch.ones_like(y, dtype = torch.float))
#     W = torch.zeros((G.shape[1], G.shape[1]))
#     d2posterior_inv = torch.zeros_like(W)
    
    n_steps = 30
    
    n, m = y.shape[0], G.shape[0]

    for i in range(n_steps):
        # if nans appeare you will need to go inside derivatives and change clip value from 1e-9 to 1e-7
        W = torch.tensor(-gpy.d2logpdf_df2(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood), dtype=torch.float)
        W = torch.diag(W.squeeze()).to(device)

        # count (K^-1 + W)^ -1
        d2posterior_inv = get_inverse_d2posterior(W, var, G)
        
        dlogpdf = torch.tensor(gpy.dlogpdf_df(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood),dtype=torch.float)
        update = torch.mm(W, f) + dlogpdf.to(device)

        f = torch.mm(d2posterior_inv, update)
    
    return f, d2posterior_inv, W

def straight_posterior(y, likelihood, G, var, *f):
    ''' 
        Calculate posterior parameters without approximations
            y - targets
            G - help matrix from decomposition,
            var - variance,
            f - mean
        returns : f, (K+W^{-1})^{-1} - inverse matrix of second derivative of posterior, W
    '''
    f = torch.zeros_like(y, dtype = torch.float)
#     W = torch.zeros((G.shape[1], G.shape[1]))
#     d2posterior_inv = torch.zeros_like(W)
    
    n_steps = 50
    
    n, m = y.shape[0], G.shape[0]
    K = torch.mm(G.T, G)
    I = torch.eye(n).to(device)

    for i in range(n_steps):
        # if nans appeare you will need to go inside derivatives and change clip value from 1e-9 to 1e-7
        W = torch.tensor(-gpy.d2logpdf_df2(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood), dtype=torch.float)
        W = torch.diag(W.squeeze()).to(device)

        # count (K^-1 + W)^ -1
        d2posterior_inv = torch.inverse(torch.inverse(K + var * I) + W) 
        
        dlogpdf = torch.tensor(gpy.dlogpdf_df(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood),dtype=torch.float)
        update = torch.mm(W, f) + dlogpdf.to(device)

        f = torch.mm(d2posterior_inv, update)
    
    return f, d2posterior_inv, W

def predict(x_train, y_train, x_test, kernel, f_posterior, likelihood, sigma):
    '''
        prediction function
        returns: mean and covariance of predictions
    '''
    
    k = torch.tensor(kernel.K(x_train, x_test), dtype = torch.float)
    K = torch.tensor(kernel.K(x_train), dtype = torch.float) + torch.eye(x_train.shape[0]) * sigma**2
    
    mean = torch.mm(k.T, torch.tensor(gpy.dlogpdf_df(f_posterior, y_train, likelihood), dtype = torch.float))
    
    k_new = torch.tensor(kernel.K(x_test))
    
    W = torch.tensor(-likelihood.d2logpdf_df2(f_posterior, y_train), dtype = torch.float)
    W = torch.diag(W.squeeze())
    W_inv = torch.inverse(W)

    covar = k_new - torch.mm(k.T, torch.mm(torch.inverse(K + W_inv + 1e-2 ** 2 * torch.eye(K.shape[0])), k))
    
    return mean, covar

# unused
def marginal_log_likelihood(G, var, W, f_posterior, y):
    '''
    calculating logarithm of marginal likelihood with approximations
        G - help matrix from decomposition,
        var - variance,
        f_posterior - posterior mean
        y - target
     returns : - logarithm of marginal likelihood
    '''
    n, m = W.shape[0], G.shape[0]

    varWI = var * W + torch.eye(n).to(device)
    varWI = torch.clamp(varWI, 1e-10, 1e10)
    varWI_inv = torch.diag(torch.pow(torch.diag(varWI), -1))

    log_lik = -0.5 * torch.logdet(varWI)

    varIGG = var * torch.eye(m).to(device) + torch.mm(G, G.T)

    log_lik -= 0.5 * torch.det(varIGG - torch.mm(G, torch.mm(varWI_inv, G.T)))
    log_lik = log_lik.reshape((-1))

    middle_term = torch.eye(n).to(device) - torch.mm(G.T, torch.mm(torch.inverse(varIGG), G))
    middle_term = torch.mm(f_posterior.T, torch.mm(middle_term, f_posterior)) / (2 * var)
    log_lik -= middle_term.reshape((-1))

    log_lik += 0.5 * m * torch.log(var)

    log_lik += torch.mean(torch.tensor(gpy.logpdf(f_posterior.detach().cpu().numpy(), y.detach().cpu().numpy(), lik)))

    return -log_lik

def straight_marginal_log_likelihood(G, var, W, f_posterior, y):
    '''
    calculating logarithm of marginal likelihood without approximations
        G - help matrix from decomposition,
        var - variance,
        f_posterior - posterior mean
        y - target
     returns : - logarithm of marginal likelihood
     '''
    n, m = W.shape[0], G.shape[0]

    K = torch.mm(G.T, G)
    K_inv = torch.inverse(K + var * torch.eye(n).to(device))
    log_lik = -0.5 * torch.mm(f_posterior.T, torch.mm(K_inv, f_posterior))

    log_lik += torch.sum(torch.tensor(gpy.logpdf(f_posterior.detach().cpu().numpy(), y.detach().cpu().numpy(), lik)))

#         log_lik -= 0.5 * torch.log(torch.det(K + sigma * torch.eye(n)) * torch.det(K_inv + W + sigma * torch.eye(n)))
    W12 = torch.sqrt(W)
    log_lik -= 0.5 * torch.logdet(torch.eye(n).to(device) + torch.mm(W12, torch.mm(K, W12))) 

    return -log_lik