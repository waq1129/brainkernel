# factor modeling demo for the working memory task in HCP. This demo runs for subject 100307 using the brain kernel prior. You can change method to "se" or "ridge" to run the comparisons across all three prior methods.

import os
from sklearn.metrics import r2_score

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import scipy.io
from sklearn.metrics import r2_score
import timeit
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_l_bfgs_b, minimize
import sys
import time
from sklearn.model_selection import KFold

from scipy.stats import ortho_group  # Requires version 0.18 of scipy

# import matplotlib.pyplot as plt
# %matplotlib inline

sublist = ['100307']

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def rbf_covariance(kernel_params, diffs2):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    return output_scale * np.exp(-0.5 * diffs2/lengthscales/lengthscales)

def run(taskname, method, rid, nf, ld_flag, nfold, rflag, subid):
    print(taskname, method, rid, nf, ld_flag, nfold, rflag, subid, flush=True)
    dataset = 'hcp'

    np.random.seed(0)

    data = scipy.io.loadmat('../HCP_data/hcp_sub'+sublist[subid]+'_'+taskname+'.mat')
    X = data['wholebrain'].astype(np.float32)
    X = X.T
    xgrid = data['coords'].astype(np.float32)
    print(xgrid.shape, flush=True)
    print(X.shape, flush=True)

    FF = scipy.io.loadmat('../brainkernel/brainkernel_prior.mat')
    F = FF['F']
    print(F.shape)

    mask = scipy.io.loadmat('../HCP_data/'+taskname+'_mask.mat')['mask'].flatten().astype(np.bool)
    print(mask.shape)
    mid = np.where(mask)[0]
    print('mid',mid.shape)
    F_bk = F[mid,:].astype(np.float32)
    xgrid = xgrid[mid,:]
    X = X[:,mid]
    print(F_bk.shape, flush=True)
    print(xgrid.shape, flush=True)
    print(X.shape, flush=True)
    
    # split data
    if rflag:
        kf = KFold(n_splits=nfold, shuffle=True, random_state=0)
    else:
        kf = KFold(n_splits=nfold)
    X_train_arr = []
    X_test_arr = []
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        X_train_arr += [X_train]
        X_test_arr += [X_test]

    X_train = X_train_arr[rid]
    X_test = X_test_arr[rid]

    xm = X_train.mean(0)
    xs = X_train.std(0)

    X_train = (X_train-xm)/(xs+0.000001)
    X_test = (X_test-xm)/(xs+0.000001)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    XX = X_test.T
    X_test_train_arr = []
    X_test_test_arr = []
    train_arr = []
    test_arr = []
    for train_index, test_index in kf.split(XX):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_test_train, X_test_test = XX[train_index].T, XX[test_index].T
        X_test_train_arr += [X_test_train]
        X_test_test_arr += [X_test_test]
        train_arr += [train_index]
        test_arr += [test_index]


    nt = X_train.shape[0]
    nd = X_train.shape[1]
    print('nt:',nt,'  nd:',nd,'  nf:',nf)
    if method=='bk':
        xinput = F_bk
    elif method=='se':
        xinput = xgrid
    else:
        xinput = xgrid
    print(xinput.shape)  

    diffs = np.expand_dims(xinput, 1) - np.expand_dims(xinput, 0)
    diffs2 = np.sum(diffs**2, axis=2)

    Cx = X_train.T.dot(X_train)
    Cov_x_time = X_train.dot(X_train.T)
    Cten = torch.tensor(Cov_x_time).float()

    ep = 1e-1
    Cov_L = np.linalg.cholesky(Cov_x_time+ep*np.eye(Cov_x_time.shape[0]))
    Cov_L_inv = np.linalg.inv(Cov_L+ep*np.eye(Cov_L.shape[0]))

    X_train_white = X_train
    X_train_ten = torch.tensor(X_train_white).float()

    W = np.random.randn(nt,nt)#ortho_group.rvs(dim=nt)#
    init_W = W[:nf,:].dot(Cov_L_inv)
    init_Winv = np.linalg.pinv(init_W)

    Cd = X_train.T.dot(X_train)
    Ct = X_train.dot(X_train.T)
    trx = np.trace(Ct)
    tra = trx/nd
    Cd1 = Cd/tra

    ep = np.mean(np.diag(Cd/(nt-1))-1)
    def runC(x_len, ld_flag):

        kx_params = [0,x_len]
        K = rbf_covariance(kx_params, diffs2)
        Kinv = np.linalg.inv(K+ep*np.eye(K.shape[0]))
        if ld_flag:
            K_L = np.linalg.cholesky(K+ep*np.eye(K.shape[0]))
            dd = np.trace(Cd1.dot(Kinv)) + 2.0 * np.sum(np.log(np.diag(K_L)))
        else:
            dd = np.trace(Cd1.dot(Kinv))

        return dd

    if method=='ridge':
        xl_min = -30
    else:
        xlenlist = np.linspace(-5,3,60)
        lls = []
        for xl in xlenlist:
            nll = runC(xl, ld_flag)
            print('xl',xl,' nll', nll,flush=True)
            lls += [nll]
        xl_min = xlenlist[np.argmin(lls)]


    x_len = xl_min
    print('x_len ',x_len,flush=True)

    kx_params = [0,x_len]
    K = rbf_covariance(kx_params, diffs2)
#     plt.imshow(K)
#     plt.show()
#     plt.imshow(Cd)
#     plt.show()
    kinv = np.linalg.inv(K+ep*np.eye(K.shape[0]))
    Kinv = torch.tensor(kinv).float()

    def test_r2(S_cf, train_arr, test_arr, X_test_train_arr, X_test_test_arr):
        cov_r2s = []
        x_r2s = []

        for trainid, testid, X_test_train, X_test_test in zip(train_arr, test_arr, X_test_train_arr, X_test_test_arr):
            X_test_train_ten = torch.tensor(X_test_train).float()

            Cx_test = X_test_test.T.dot(X_test_test)

            S_test_train = S_cf[:,trainid]
            S_test_test = S_cf[:,testid]
            SX = S_test_train.dot(X_test_train.T)
            SS = S_test_train.dot(S_test_train.T)

            D = np.linalg.inv(S_test_train.dot(S_test_train.T)+ep*np.eye(nf))
            F_train_test = D.dot(S_test_train).dot(X_test_train.T)

            Winv_test = F_train_test.T

            Xr = Winv_test.dot(S_test_test)
            Cxr = Xr.T.dot(Xr)

            cov_r2 = r2_score(Cx_test, Cxr)
            x_r2 = r2_score(X_test_test, Xr)

            cov_r2s += [cov_r2]
            x_r2s += [x_r2]

        print('cov_r2s: ', cov_r2s, flush=True)
        cov_r2 = np.mean(cov_r2s)
        x_r2 = np.mean(x_r2s)
        return cov_r2,x_r2

    def find_init_L(K,Cd,ep):
        kinv = np.linalg.inv(K+ep*np.eye(K.shape[0]))
        L = np.linalg.cholesky(kinv)
        P = 0.25*(K+ep*np.eye(K.shape[0]))+Cd/nt
        A = L.T.dot(P).dot(L)
        U,s,Uh = np.linalg.svd(A)
        S = np.diag(s**0.25)
        Q = U.dot(S).dot(Uh)
        B = np.linalg.inv(L).T.dot(Q)

        Q1 = B.dot(B.T)-0.5*(K+ep*np.eye(K.shape[0]))-ep*np.eye(nd)
        U,s,Uh = np.linalg.svd(Q1)
        L = U[:,:nf].dot(np.diag(s[:nf]**0.5))

        return L

    init_L = find_init_L(K,Cd,ep)

    class FA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.L = torch.nn.Parameter(torch.tensor(init_L).float())

        def forward(self, X_white, Kinv):
            L = self.L
            LL = torch.matmul(L.t(),L)+ep*torch.eye(nf)
            LLinv = torch.inverse(LL)
            LX = torch.matmul(L.t(),X_white.t())
            LXXL = torch.matmul(LX,LX.t())

            LLchol = torch.cholesky(LL)
            LL_ld = 2.0 * torch.sum(torch.log(torch.diag(LLchol)))

            l1 = -0.5*torch.trace(torch.matmul(LXXL,LLinv))/ep/nt

            l2 = 0.5*LL_ld

            l3 = 0.5*torch.trace(torch.matmul(torch.matmul(L.t(),Kinv),L))

            ll = l1+l2+l3
            return ll, l1, l2, l3

    # Construct our model by instantiating the class defined above
    model = FA()
    model.float()

    param = list(model.parameters())
    optimizer1 = torch.optim.LBFGS(param, lr=1, max_iter=200)

    ############
    Cx = X_train.T.dot(X_train)
    model.train()
    def closure1():
        optimizer1.zero_grad()
        output, l1, l2, l3 = model(X_train_ten, Kinv)
        loss_train = output
        loss_train.backward()
        return loss_train

    t = time.perf_counter()
    ll_u = []
    r2s = []
    r2s_x = []
    epochs = 50
    cov_r2 = 0
    x_r2 = 0
    for epoch in range(epochs):
        if epoch<20:
            loss_train = optimizer1.step(closure1)
        else:
            loss_train = optimizer1.step(closure1)

        if np.mod(epoch,1)==0:
            ll_u += [loss_train.item()]

            ll, l1, l2, l3 = model(X_train_ten, Kinv)

            L = model.L.detach().numpy()
            S = L.T

            cov_r2, x_r2 = test_r2(S, train_arr, test_arr, X_test_train_arr, X_test_test_arr)
            print('@@@@@@ NEW EPOCH', epoch, flush=True)
            print('loss',ll.item(),l1.item(),l2.item(),l3.item(), '  ##cov_r2:', cov_r2, '  ##x_r2:', x_r2,flush=True)

            r2s += [cov_r2]
            r2s_x += [x_r2]

    print('cov_r2:',cov_r2, '   x_r2:',x_r2, flush=True)

    r2s_smooth = []
    ws = 10
    for i in range(ws,len(r2s)):
        r2s_smooth += [np.mean(r2s[i-ws:i])]
    cov_r2 = r2s_smooth[-1]
    
    r2s_x_smooth = []
    for i in range(ws,len(r2s_x)):
        r2s_x_smooth += [np.mean(r2s_x[i-ws:i])]
    x_r2 = r2s_x_smooth[-1]
        
    return cov_r2, x_r2, x_len

dataset = 'hcp' 
taskname = 'WM' # working memory task
method = 'bk' # ridge, se, bk
subid = 0 # index of subject

ld_flag = 0
nf = 30 # number of latent dimension in factor modeling
nfold = 2 # nfold cross validation
rid = 0 # index of split, when nfold=2, rid=[0,1]
rflag = 0 # randomness

print('taskname: ', taskname, ' method: ', method, ' rid: ', rid, ' nf: ', nf, ' ld_flag: ', ld_flag, ' nfold: ', nfold, ' rflag: ', rflag, ' subid: ', subid, flush=True)

r2s = run(taskname, method, rid, nf, ld_flag, nfold, rflag, subid)
cov_r2, x_r2, x_len = r2s[0], r2s[1], r2s[2] # cov_r2: r2 between test sample cov and predicted cov; x_r2: r2 between test x and predicted x













