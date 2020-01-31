#!/usr/bin/env python
# coding: utf-8
# ==========================================================================================================
# Created By  : Chima Eke
# Creation Date  :  10/12/2019
# ==========================================================================================================

# =========================================IMPORTS==========================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, make_scorer
from datetime import datetime
# ==========================================================================================================


# ============================================== DEF OF FUNCTIONS ==========================================
def calc_avg_KI(listofsigs,s,N):                                   # Returns each signatures's average KI (KItot) and its std
    avg_KI_per_sig = []
    std_avg_KI_per_sig = []
    for i in range(len(listofsigs)):
        KI = []
        for j in range(len(listofsigs)):
            if j!=i:
                r = np.intersect1d(listofsigs[i],listofsigs[j], assume_unique=False, return_indices=False).shape[0]
                KI.append((r-s**2/N)/(s-s**2/N))
            else:pass
        avg_KI_per_sig.append(np.mean(KI))
        std_avg_KI_per_sig.append(np.std(KI,ddof=1))                    # ddof=1 ensures sample (not population) standard deviation is calculated.
    return (avg_KI_per_sig,std_avg_KI_per_sig)

def Stot(list_avg_KI_per_sig):                                          # Returns Stot (total stabilty for a given signature size s). 
    return np.mean(list_avg_KI_per_sig)


# =================================== PATH FOR WRITING OUTPUT ================================================
filepath = 'directory_path'                              
filename = filepath + 'output_file_prefix'          

# ============================================  INPUT DATA ===================================================
data = pd.read_csv('input_csv_file.csv', delimiter=',')
class_labels = 'target_label'                       
y_orig = data[class_labels].values
cols_to_drop = ['list_of_colums_in_input_csvfile_to_drop_if_any']
X_orig = data.drop(columns=cols_to_drop) 
pos_neg_others_labels = [1, 0]                          # list of positive and negative labels
features = X_orig.columns
feature_inds = np.arange(features.shape[0])

# ==================================================== FEATURE SCALING METHOD ================================
scaler = StandardScaler()
 

# ============================================= CONFIG OF EXPERIMENTAL PARAMETERS ============================
clf = LinearSVC(max_iter = 10000000, dual=True)                                  
rfe = RFE(clf,n_features_to_select=20, step=0.2)                                     
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10,random_state=30)
cv_clf = make_pipeline(scaler,clf)
scoring = {'ACC':  make_scorer(accuracy_score),
           'SN': make_scorer(recall_score, labels = [pos_neg_others_labels[0]], pos_label=pos_neg_others_labels[0], average='binary'),
           'SP': make_scorer(recall_score, labels = [pos_neg_others_labels[1]], pos_label=pos_neg_others_labels[1], average='binary'),
           'PPV': make_scorer(precision_score, labels= [pos_neg_others_labels[0]], pos_label=pos_neg_others_labels[0], average='binary'),
           'NPV': make_scorer(precision_score, labels = [pos_neg_others_labels[1]], pos_label=pos_neg_others_labels[1], average='binary'),
           'roc_auc': make_scorer(roc_auc_score, needs_threshold=True)}    

n_subsamps = 500                                                                      # number of subsamples
n_boots = 50                                                                          # number of bootstraps
subsamp_frac = 0.80                                                                   # fraction of the original dataset to use in forming a subsample
subsamp_size = int(subsamp_frac*X_orig.shape[0])                                      # floors to the nearest int. prefers to importing math.floor
n_feats_to_select = ['list_of_signature_sizes']

perform_per_n_feats_to_select = {'avg_auc_CLA':[],            
                                 'avg_auc_CWA':[],
                                 'Stot_CLA':[],
                                 'std_Stot_CLA':[],
                                 'Stot_CWA':[],
                                 'std_Stot_CWA':[]
                                 }

#Beginning of main codes
for n in n_feats_to_select:
    print('n_feats_to_select: ' + str(n) + ';' + '  Time: ' + str(datetime.now()))
    rfe.n_features_to_select = n
    sig_per_subsamp_CLA = []
    sig_per_subsamp_CWA = []
    perform_per_subsamp_CLA = []
    cv_perform_per_subsamp_CLA = []
    perform_per_subsamp_CWA = []
    cv_perform_per_subsamp_CWA = []
    
    sum_auc_all_subsamps = {'sum_auc_all_subsamps_CLA':0,                                 # stores the sum of all aucs from all the subsamples for a given n
                          'sum_auc_all_subsamps_CWA':0}

    for subsamp in range(n_subsamps):
        print('subsample: ' + str(subsamp+1) + ';' + '  Time: ' + str(datetime.now()))
        xtrain_subsamp, xtest_subsamp, ytrain_subsamp, ytest_subsamp = train_test_split(X_orig.values, y_orig, test_size=1-subsamp_frac, stratify=y_orig, random_state=subsamp)
        auc_per_boot=[]
        panel_per_boot = []
        feature_CLA = []    # stores the CLA of each feature, with index corresponding to the index of each feature in features
        feature_CWA = []    # stores the CWA of each feature, with index corresponding to the index of each feature in features
        feature_rankings = {feature:{'rank': [], 'weighted_rank':[]} for feature in features}
    
        # perform expt per bootstrap
        for boot in range(n_boots):
            train_inds_boot = resample(np.arange(subsamp_size), replace=True, n_samples=subsamp_size, random_state=boot, stratify=ytrain_subsamp).astype(int)  # randomly selected indices with replacement to be used for getting training samples for a bootstrap. Added .astype(int) because without it, I found that the values were floats and thus failed when used to index an array.
            test_inds_boot = np.setxor1d(np.arange(subsamp_size), train_inds_boot, assume_unique=False).astype(int)                              # np.setxor1d finds the set exclusive-or of two arrays; the return array is sorted.
            xtrain_boot,xtest_boot = scaler.fit_transform(xtrain_subsamp[train_inds_boot]),scaler.transform(xtrain_subsamp[test_inds_boot])
            ytrain_boot,ytest_boot = ytrain_subsamp[train_inds_boot],ytrain_subsamp[test_inds_boot]
            rfe.fit(xtrain_boot,ytrain_boot)
            yscore_boot = rfe.estimator_.decision_function(xtest_boot[:,feature_inds[rfe.get_support()]])                               # feature_inds[rfe.get_support] are the indices of features selected by rfe  
            auc_per_boot.append(roc_auc_score(ytest_boot, yscore_boot)) 
            panel_per_boot.append(features[rfe.get_support()])

            for (feature,rank) in zip(features,rfe.ranking_):                                 
                feature_rankings[feature]['rank'].append(rank)
                if feature in panel_per_boot[boot]:
                    feature_rankings[feature]['weighted_rank'].append(rank*(1-auc_per_boot[boot])) #  higher AUC ==> smaller wi ==> smaller rank ==> better feature
                else:
                    feature_rankings[feature]['weighted_rank'].append(rank)                        # rank here correponds to 1*rank ===> wi=1
  
        #SELECT FINAL SIGNATURES (one from CLA and one from CWA) FROM THIS SUBSAMPLE
        for feature in features:
            feature_CLA.append(sum(feature_rankings[feature]['rank']))
            feature_CWA.append(sum(feature_rankings[feature]['weighted_rank']))
        CLA_top_n_features_inds = np.argsort(feature_CLA)[:n]        # selects indices of the best n_feats_to_select features ==> those with lowest CLA values
        CWA_top_n_features_inds = np.argsort(feature_CWA)[:n]
        sig_per_subsamp_CLA.append(features[CLA_top_n_features_inds].tolist())
        sig_per_subsamp_CWA.append(features[CWA_top_n_features_inds].tolist())
        xtrain_subsamp_CLA,xtest_subsamp_CLA = xtrain_subsamp[:,CLA_top_n_features_inds],xtest_subsamp[:,CLA_top_n_features_inds] #np.sort(CLA_top_n_features_inds) will sort the indices in ascending order
        xtrain_subsamp_CWA,xtest_subsamp_CWA = xtrain_subsamp[:,CWA_top_n_features_inds],xtest_subsamp[:,CWA_top_n_features_inds]
    
        #EVALUATE THE PERFORMANCE OF THE TWO SIGNATURES ON THE OUT-OF-SUBSAMPLE SET
        for (xtrain,xtest,id) in [(xtrain_subsamp_CLA,xtest_subsamp_CLA,'CLA'),(xtrain_subsamp_CWA,xtest_subsamp_CWA,'CWA')]:
            clf.fit(scaler.fit_transform(xtrain),ytrain_subsamp)
            ypred_subsamp = clf.predict(scaler.transform(xtest))
            yscore_subsamp = clf.decision_function(scaler.transform(xtest))
            perform_subsamp = {'sig_'+id:str(features[CLA_top_n_features_inds].tolist()),
                       'coeff_abs': str(abs(clf.coef_).tolist()),
                       'AUC_'+id:roc_auc_score(ytest_subsamp,yscore_subsamp),
                       'ACC_'+id:accuracy_score(ytest_subsamp, ypred_subsamp),
                       'SN_'+ id:recall_score(ytest_subsamp, ypred_subsamp, labels = [pos_neg_others_labels[0]], pos_label=pos_neg_others_labels[0], average='binary'),
                       'SP_'+id:recall_score(ytest_subsamp, ypred_subsamp, labels = [pos_neg_others_labels[1]], pos_label=pos_neg_others_labels[1], average='binary'),  
                       'PPV_'+id:precision_score(ytest_subsamp, ypred_subsamp, labels = [pos_neg_others_labels[0]], pos_label=pos_neg_others_labels[0], average='binary'),
                       'NPV_'+id:precision_score(ytest_subsamp, ypred_subsamp, labels = [pos_neg_others_labels[1]], pos_label=pos_neg_others_labels[1], average='binary')}   
    
            #EVALUATE THE CROSS-VALIDATED PERFORMANCE OF THE SELECTED SIGNATURE ON THE CURRENT SUB-SAMPLE  
            scores = cross_validate(cv_clf, xtrain, ytrain_subsamp, scoring=scoring, cv=cv)        
            cv_perform_subsamp = {'AUC_cv_'+id: scores['test_roc_auc'].mean(),
                                  'ACC_cv_'+id: scores['test_ACC'].mean(),  
                                  'SN_cv_'+id: scores['test_SN'].mean(),
                                  'SP_cv_'+id: scores['test_SP'].mean(),
                                  'PPV_cv_'+id: scores['test_PPV'].mean(),
                                  'NPV_cv_'+id: scores['test_NPV'].mean()}
    
            if id=='CLA': 
                perform_per_subsamp_CLA.append(perform_subsamp)
                cv_perform_per_subsamp_CLA.append(cv_perform_subsamp)
            elif id=='CWA':
                perform_per_subsamp_CWA.append(perform_subsamp)
                cv_perform_per_subsamp_CWA.append(cv_perform_subsamp)
            sum_auc_all_subsamps['sum_auc_all_subsamps_'+id] += perform_subsamp['AUC_'+id]

    ### RECORD OUTCOMES PER n in n_feats_to_select
    perform_per_n_feats_to_select['avg_auc_CLA'].append(sum_auc_all_subsamps['sum_auc_all_subsamps_CLA']/n_subsamps)        # Record avg auc per subsamp
    perform_per_n_feats_to_select['avg_auc_CWA'].append(sum_auc_all_subsamps['sum_auc_all_subsamps_CWA']/n_subsamps)
    feats_from_all_sigs_CLA = np.array(sig_per_subsamp_CLA).flatten().tolist()
    feats_from_all_sigs_CWA = np.array(sig_per_subsamp_CWA).flatten().tolist()
    feat_count_CLA = {feature:count for feature in features for count in [feats_from_all_sigs_CLA.count(feature)]}            # Counts frequency of all the items in features;
    feat_count_CWA = {feature:count for feature in features for count in [feats_from_all_sigs_CWA.count(feature)]}

    avg_KI_per_sig_CLA,std_avg_KI_per_sig_CLA = calc_avg_KI(sig_per_subsamp_CLA,n,len(features))
    avg_KI_per_sig_CWA,std_avg_KI_per_sig_CWA = calc_avg_KI(sig_per_subsamp_CWA,n,len(features))
    Stot_CLA,Stot_CWA = Stot(avg_KI_per_sig_CLA),Stot(avg_KI_per_sig_CWA)
    perform_per_n_feats_to_select['Stot_CLA'].append(Stot_CLA)
    perform_per_n_feats_to_select['Stot_CWA'].append(Stot_CWA)
    perform_per_n_feats_to_select['std_Stot_CLA'].append(np.std(2*np.array(avg_KI_per_sig_CLA),ddof=1)) 
    perform_per_n_feats_to_select['std_Stot_CWA'].append(np.std(2*np.array(avg_KI_per_sig_CWA),ddof=1))                                         

    list_dicts_CLA = [cv_perform_per_subsamp_CLA, perform_per_subsamp_CLA,{'avg_KI_per_sig_CLA':avg_KI_per_sig_CLA},{'std_avg_KI_per_sig_CLA':std_avg_KI_per_sig_CLA},{'Stot_CLA':[Stot_CLA]}]
    list_dicts_CWA = [cv_perform_per_subsamp_CWA, perform_per_subsamp_CWA,{'avg_KI_per_sig_CWA':avg_KI_per_sig_CWA},{'std_avg_KI_per_sig_CWA':std_avg_KI_per_sig_CWA},{'Stot_CWA':[Stot_CWA]}]
    
    # WRITE RESULTS OF THE CLA AND CWA ENSEMBLE METHODS TO CSV FILES
    pd.concat([pd.DataFrame(f) for f in list_dicts_CLA], axis=1, sort=False).to_csv(filename + 'CLA_perf_n_'+str(n) + '.csv', index=False, encoding='utf-8')
    pd.concat([pd.DataFrame(f) for f in list_dicts_CWA], axis=1, sort=False).to_csv(filename + 'CWA_perf_n_'+str(n) + '.csv', index=False, encoding='utf-8')
 
    pd.DataFrame({'feat_count_CLA'+str(n):feat_count_CLA}).to_csv(filename + 'CLA_feat_count_n_' + str(n) + '.csv', index_label='feature', encoding='utf-8')
    pd.DataFrame({'feat_count_CWA'+str(n):feat_count_CWA}).to_csv(filename + 'CWA_feat_count_n_' + str(n) + '.csv', index_label='feature', encoding='utf-8')

# WRITE SUMMARY RESULTS (INCLUDING TOTAL AVERAGE AUC AND STABILITY AND THEIR STANDARD DEVIATIONS) FOR EACH SIZE OF SIGNATURE
pd.concat([pd.DataFrame({'n_features_to_select':n_feats_to_select}),pd.DataFrame(perform_per_n_feats_to_select)], axis=1, sort=False).to_csv(filename + 'perf_of_each_n_1.csv', index=False, encoding='utf-8')
