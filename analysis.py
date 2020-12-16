import numpy as np
import torch

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def predictive_entropy(y_pred):
    return -y_pred * torch.log(y_pred) - (1 - y_pred) * torch.log(1 - y_pred)
    
def roc_curve_with_k(uncertainty, X_test, y_test, net_proba, k):
    ''' k is the percent of points which taken from prediction'''
    pred_n = X_test.shape[0] - int(X_test.shape[0] * k)
    _, least_sure_indices = torch.topk(uncertainty, pred_n, largest=True, dim=0)

    modified_proba = net_proba
    modified_proba[least_sure_indices] = y_test[least_sure_indices]
    
    fpr, tpr, treshholds = roc_curve(y_test.detach().cpu(), modified_proba.detach().cpu())

    return np.trapz(tpr, fpr)

def get_uncertainty_curve(uncertainty, X_test, y_test, prediction, k_s):
    return [roc_curve_with_k(uncertainty(prediction), X_test, y_test, prediction, k) for k in k_s]

def get_accuracy_info(model, X_test, y_test, sigma, y_train, lik):
    plt.figure()
        
    mean, covar, net_prediction = model.predict(X_test, y_train, lik, sigma**2)
#     plt.plot(net_prediction, 'o')

    print("Range of network predictions: ")
    print(net_prediction.min(), " - ", net_prediction.max())
    print("mean: ", net_prediction.mean())

    net_proba = torch.tensor(net_prediction).to(device)
    net_prediction = torch.tensor(net_prediction).ge(0.5).float()

    accuracy = accuracy_score(y_test.detach().cpu().numpy(), net_prediction)
    print("accuracy: ", accuracy)
    return accuracy, net_proba.reshape(-1, 1)