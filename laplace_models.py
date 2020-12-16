'''
use Laplace Approximation for building a model
'''


import torch
import torch.nn as nn
import gpytorch
import GPy
import numpy as np

import gpy
import laplace

lik = GPy.likelihoods.Bernoulli()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class MarginalLogLikelihood(nn.Module):
    '''
    calculating logarithm of marginal likelihood with approximations
        G - help matrix from decomposition,
        var - variance,
        f_posterior - posterior mean
        y - target
     forward returns : - logarithm of marginal likelihood
    '''
    def __init__(self):
        super(MarginalLogLikelihood, self).__init__()
        
    def forward(self, G, var, W, f_posterior, y):
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
        
class StraightMarginalLogLikelihood(nn.Module):
    '''
    calculating logarithm of marginal likelihood without approximations
        G - help matrix from decomposition,
        var - variance,
        f_posterior - posterior mean
        y - target
     forward returns : - logarithm of marginal likelihood
     '''
    def __init__(self):
        super(StraightMarginalLogLikelihood, self).__init__()
        
    def forward(self, G, var, W, f_posterior, y):
        n, m = W.shape[0], G.shape[0]
        
        K = torch.mm(G.T, G)
        K_inv = torch.inverse(K + var * torch.eye(n).to(device))
        log_lik = -0.5 * torch.mm(f_posterior.T, torch.mm(K_inv, f_posterior))

        log_lik += torch.sum(torch.tensor(gpy.logpdf(f_posterior.detach().cpu().numpy(), y.detach().cpu().numpy(), lik)))
        
#         log_lik -= 0.5 * torch.log(torch.det(K + sigma * torch.eye(n)) * torch.det(K_inv + W + sigma * torch.eye(n)))
        W12 = torch.sqrt(W)
        log_lik -= 0.5 * torch.logdet(torch.eye(n).to(device) + torch.mm(W12, torch.mm(K, W12))) 
        
        return -log_lik
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class LargeFeatureExtractor(nn.Module):
    '''Network trained for reducing dimension of featire'''
    def __init__(self, data_dim, predictions=False, base_size=32, hidden_layer=5):
        super(LargeFeatureExtractor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(data_dim, 2*base_size),
            nn.BatchNorm1d(2*base_size),
            nn.ReLU(),
            nn.Linear(2*base_size, 2*base_size),
            nn.BatchNorm1d(2*base_size),
            nn.ReLU(),
            nn.Linear(2*base_size, base_size),
            nn.BatchNorm1d(base_size),
            nn.ReLU(),
            nn.Linear(base_size, hidden_layer),
        )
        self.last_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_layer, 2)
        )    
        self.log_soft_max = nn.LogSoftmax(dim=0)
        self.soft_max = nn.Softmax(dim=1)
        self.predictions = predictions
        
    def forward(self, x):
        out = self.main(x)
        if self.predictions:
            out = self.last_linear(out)
            
        return out
    
    def predict(self, x):
        out = self.main(x)
        out = self.last_linear(out)
        out = self.soft_max(out)
            
        return out

# we will think about predictions later

class DeepGPClassificationModel(nn.Module):
    '''model for classification task, build on NN feature extractor and gaussian processes as decision method'''
    def __init__(self, feature_extractor, y, straight):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.straight = straight
        f = torch.normal(torch.zeros_like(y, dtype = torch.float), 0.5 * torch.ones_like(y, dtype = torch.float))
        self.f = nn.Parameter(f, requires_grad=True) 
        
        # this is uneeded for now
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.LinearKernel() # RBFKernel()

    def forward(self, x, y, var):
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1
        projected_x = projected_x * torch.sqrt(self.covar_module.variance)
        
        if self.straight:
            f, covar, W = self.straight_posterior(y, lik, projected_x.T, var)
        else:
            f, covar, W = self.posterior(y, lik, projected_x.T, var)
        
        # makes sense only while training n the whole dataset at once!
        self.G = projected_x.T
        self.W = W
        self.f = nn.Parameter(f)
        
        return self.f, covar, W, projected_x.T
#         return gpytorch.distributions.MultivariateNormal(f, covar)

    def posterior(self, y, likelihood, G, var):
        '''
        Calculate posterior parameters with approximations
            y - targets
            G - help matrix from decomposition,
            var - variance,
        returns : f, (K+W^{-1})^{-1} - inverse matrix of second derivative of posterior, W
        '''
        f = self.f
        n_steps = 30

        n, m = y.shape[0], G.shape[0]

        for i in range(n_steps):
            # if nans appeare you will need to go inside derivatives and change clip value from 1e-9 to 1e-7
            W = -gpy.d2logpdf_df2(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood)
            W = torch.tensor(W, dtype=torch.float)
            W = torch.diag(W.squeeze()).to(device)

            # count (K^-1 + W)^ -1
            d2posterior_inv = laplace.get_inverse_d2posterior(W, var, G)

            dlogpdf = gpy.dlogpdf_df(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood)
            dlogpdf = torch.tensor(dlogpdf, dtype=torch.float)
            update = torch.mm(W, f) + dlogpdf.to(device)

            f = torch.mm(d2posterior_inv, update)

        return f, d2posterior_inv, W
    
    def straight_posterior(self, y, likelihood, G, var):
        '''
        Calculate posterior parameters without approximations
            y - targets
            G - help matrix from decomposition,
            var - variance,
        returns : f, (K+W^{-1})^{-1} - inverse matrix of second derivative of posterior, W
        '''
        f = self.f

        n_steps = 50

        n, m = y.shape[0], G.shape[0]
        K = torch.mm(G.T, G)
        I = torch.eye(n).to(device)

        for i in range(n_steps):
            # if nans appeare you will need to go inside derivatives and change clip value from 1e-9 to 1e-7
            W = -gpy.d2logpdf_df2(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood)
            W = torch.tensor(W, dtype=torch.float)
            W = torch.diag(W.squeeze()).to(device)

            # count (K^-1 + W)^ -1
            d2posterior_inv = torch.inverse(torch.inverse(K + var * I) + W) 

            dlogpdf = gpy.dlogpdf_df(f.detach().cpu().numpy(), y.detach().cpu().numpy(), likelihood)
            dlogpdf = torch.tensor(dlogpdf, dtype=torch.float)
            update = torch.mm(W, f) + dlogpdf.to(device)

            f = torch.mm(d2posterior_inv, update)
    
        return f, d2posterior_inv, W

    def predict(self, x_test, y_train, lik, var):
        '''
        predict probabilities with mean and covariance
        '''
        
        net_pred = self.feature_extractor(x_test)
        
        k_new = torch.diag(torch.mm(net_pred, net_pred.T))
        K = torch.mm(self.G.T, self.G)
        k = torch.mm(self.G.T, net_pred.T)

        # if we do not remember W during train we should count it this way
#         W = torch.FloatTensor(-gpy.d2logpdf_df2(f.detach().numpy(), y_train, lik))
#         W = torch.diag(W.squeeze())
        W = self.W
        # GPy way of prediction
        # W12 and B needed for numerical stability. Do not use (K + W_inv) ^ -1
        W12 = torch.sqrt(W)
        B = torch.eye(K.shape[0]).to(device) + torch.mm(W12, torch.mm(K, W12))
        WBW = torch.mm(W12, torch.mm(torch.inverse(B), W12))

        # our way
#         W_star = torch.inverse(W) + var * torch.eye(W.shape[0]).to(device)
#         W_star = torch.inverse(W_star)
#         GW_star = torch.mm(self.G, W_star)
#         middle = torch.inverse(torch.eye(self.G.shape[0]).to(device) + torch.mm(GW_star, self.G.T))
#         WBW = W_star - torch.mm(W_star, torch.mm(self.G.T, torch.mm(middle, GW_star)))
        
        # covar = k_new - torch.mm(k.T, torch.mm(torch.inverse(K + W_inv), k))
        covar = k_new - torch.sum(torch.mm(WBW.T, k) * k, 0)
        covar = torch.clamp(covar, 1e-15, float("inf"))

        mean = torch.tensor(gpy.dlogpdf_df(self.f.detach().cpu().numpy(), y_train.detach().cpu().numpy(), lik),dtype=torch.float)
        mean = torch.mm(k.T, mean.to(device)).reshape((-1))

        proba = mean / torch.sqrt(1 + covar)
        proba = gpy.std_norm_cdf(proba.detach().cpu().numpy().reshape((-1)))

        return mean, covar, proba