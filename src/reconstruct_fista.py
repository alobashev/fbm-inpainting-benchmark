import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import cv2
import os


#proximal gradient for non smooth function
def singular_value_soft_thresholding(X,beta):
    
    #svd
    U,D,Vt=scipy.linalg.svd(X,full_matrices=False)
    
    #compute soft threshold
    diag_threshold=np.max(np.c_[D-beta,np.zeros(D.shape)],axis=1)
    
    return U@np.diag(diag_threshold)@Vt

#ista with nesterov acceleration
def reconstruct_fista(Y,omega,beta=0.9,max_itr=1000,epsilon=1e-8,diagnosis=False):
    Y = Y**2
    #initialize
    X=Y.copy()
    Z=X.copy()
    X_prev=X.copy()
    t_prev=1
    counter=0
    cost=float('inf')
    stop=False
    
    X_history  = []
    
    while not stop:    
        X_history.append(Z)
        #update
        Z[omega]=Y[omega]
        X=singular_value_soft_thresholding(Z,beta)
        
        
        
        #nesterov acceleration
        t=(1+(1+4*(t_prev**2))**0.5)/2
        Z=(t_prev-1)*(X-X_prev)/t+X
        
        t_prev=t
        X_prev=X
        counter+=1
        
        #compute cost
        cost_prev=cost
        cost=((X[omega]-Y[omega])**2).sum()+beta*np.linalg.norm(X,ord='nuc')
        
        #maximum iteration check
        if counter>=max_itr:
            if diagnosis:
                print(f'Not converged after {counter} iterations')
            stop=True
        
        #convergence check
        if abs(cost/cost_prev-1)<epsilon:
            if diagnosis:
                print(f'Converged after {counter} iterations')
            stop=True
            
    def rank_truncated_svd(X, r=5):  
        U, S, V = np.linalg.svd(X, full_matrices=False)  
        U = U[:, :r]  
        S = S[:r]  
        V = V[:r, :]  
        recs_hat = U @ np.diag(S) @ V  
        return recs_hat  

    X = np.sqrt(np.abs(rank_truncated_svd(X, r=5)))
    return X