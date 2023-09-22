import os
import shutil
import numpy as np

from tqdm import tqdm
from tempfile import mkdtemp
from scipy.stats import kendalltau
from joblib import Parallel, delayed, Memory

def dep2mi(x):
    MAX_VAL = 0.999999;
    if(x>MAX_VAL):
        x = MAX_VAL
    elif(x<-MAX_VAL):
        x = -MAX_VAL
    
    y = -0.5*np.log(1-x*x);

    return y

def mi_tau(x,y):
    tau_val,p_val = kendalltau(x,y)
    if(np.isnan(tau_val)):
        return 0
    return dep2mi(tau_val)

def generic_combined_scorer(x1, o1, i_1, x2, o2, i_2, y, h):
    s1 = h(x1, y)
    s2 = h(x2, y)
    o1[i_1] = s1
    o2[i_2] = s2

def feature_select(X, Y, num_features_to_select=None, K_MAX=1000, estimator=mi_tau, n_jobs=-1,verbose=True):
    '''
    Implements the MRMR algorithm for feature-selection: http://ieeexplore.ieee.org/document/1453511/

    Inputs:
        X - A feature-matrix, of shape (N,D) where N is the number of samples and D is the number
            of features
        Y - A vector of shape (N,1) which represents the output.  Each index in Y is assumed to
            correspond to the row with the same index in X.
        num_features_to_select - the number of features to select from the provided X matrix.  If None
                                 are provided, then all the features that are available are ranked/ordered.
                                 (default: None)
        K_MAX - the maximum number of top-scoring features to consider.
        estimator - a function handle to an estimator of association (that theoretically should
                    follow the DPI assumptions)
        n_jobs - the numer of processes to use with parallel processing in the background
        verbose - if True, show progress

    Output:
        a vector of indices sorted in descending order, where each index represents the "importance"
        of the feature, as computed by the MRMR algorithm.
    '''
    num_features = X.shape[1]

    if(num_features_to_select is not None):
        num_selected_features = min(num_features, num_features_to_select)
    else:
        num_selected_features = num_features
    
    K_MAX_internal = min(num_features,K_MAX)

    initial_scores = Parallel(n_jobs=n_jobs)(delayed(estimator)(X[:,i],Y) for i in range(num_features))
    
    # rank the scores in descending order
    sorted_scores_idxs = np.flipud(np.argsort(initial_scores))
    
    # subset the data down so that joblib doesn't have to transport large matrices to its workers
    X_subset = X[:,sorted_scores_idxs[0:K_MAX_internal]]

    tmp_folder = mkdtemp()

    selected_feature_idxs    = np.zeros(num_selected_features,dtype=int)
    remaining_candidate_idxs = list(range(1,K_MAX_internal))
    
    relevance_vec_fname = os.path.join(tmp_folder, 'relevance_vec')
    feature_redundance_vec_fname = os.path.join(tmp_folder, 'feature_redundance_vec')
    mi_matrix_fname = os.path.join(tmp_folder, 'mi_matrix')
    relevance_vec = np.memmap(relevance_vec_fname, dtype=float, shape=(K_MAX_internal,), mode='w+')
    feature_redundance_vec = np.memmap(feature_redundance_vec_fname, dtype=float, shape=(K_MAX_internal,), mode='w+')
    mi_matrix = np.memmap(mi_matrix_fname, dtype=float, shape=(K_MAX_internal,num_selected_features-1), mode='w+')
    mi_matrix[:] = np.nan

    with tqdm(total=num_selected_features,desc='Selecting Features ...',disable=(not verbose)) as pbar:
        pbar.update(1)
        for k in range(1,num_selected_features):
            ncand = len(remaining_candidate_idxs)
            last_selected_feature = k-1

            Parallel(n_jobs = n_jobs)(delayed(generic_combined_scorer)(Y,relevance_vec, i,
                                                                       X_subset[:,selected_feature_idxs[last_selected_feature]],
                                                                       feature_redundance_vec,i,X_subset[:,i], estimator) 
                                      for i in remaining_candidate_idxs)

            # copy the redundance into the mi_matrix, which accumulates our redundance as we compute
            mi_matrix[remaining_candidate_idxs, last_selected_feature] = feature_redundance_vec[remaining_candidate_idxs]
            redundance_vec = np.nanmean(mi_matrix[remaining_candidate_idxs,:], axis=1)

            tmp_idx = np.argmax(relevance_vec[remaining_candidate_idxs]-redundance_vec)
            selected_feature_idxs[k] = remaining_candidate_idxs[tmp_idx]
            del remaining_candidate_idxs[tmp_idx]
            
            pbar.update(1)
    
    # map the selected features back to the original dimensions
    selected_feature_idxs = sorted_scores_idxs[selected_feature_idxs]

    # clean up
    try:
        shutil.rmtree(tmp_folder)
    except:
        pass

    return selected_feature_idxs

