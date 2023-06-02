import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
import json

SEED=42
OPT_INITS = 3
OPT_ITERS = 50
        
def main():

    hrvs=pd.read_csv(f'dataset.csv', index_col=0)
       
    labels = hrvs['label']
    stayids = hrvs['stayid']
    tests = hrvs['test']
    
    train_idx = hrvs[hrvs['test']==0].index
    test_idx = hrvs[hrvs['test']==1].index
    
    
    inputs = hrvs.drop(columns=['stayid', 'label', 'time','test'])# 'binary'


    X, x_test, Y, y_test = inputs.loc[train_idx], inputs.loc[test_idx], labels.loc[train_idx], labels.loc[test_idx]

    nsmap = len(labels)
    ntest = len(x_test)
    ntrain = len(X)
    
    bayes_dtrain = lgb.Dataset(X, Y)
    bayes_dtest = lgb.Dataset(x_test, y_test)

    param_bounds = {'num_leaves': (16, 32),
                    'lambda_l1': (0.7, 0.9),
                    'lambda_l2': (0.9, 1),
                    'feature_fraction': (0.6, 0.7),
                    'bagging_fraction': (0.6, 0.9),
                    'min_child_samples': (6, 10),
                    'min_child_weight': (10, 40)}
    
    fixed_params = {'objective': 'binary',
                    'learning_rate': 0.005,
                    'bagging_freq': 1,
                    'force_row_wise': True,
                    'max_depth': 5,
                    'verbose': -1,
                    'random_state': SEED,
                    'n_jobs':32,
                    }
    
    def auprc(preds, dtrain):
        labels = dtrain.get_label()
        return 'auprc', average_precision_score(labels, preds), True 

    params = {'num_leaves': int(round(num_leaves)),
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'min_child_samples': int(round(min_child_samples)),
            'min_child_weight': min_child_weight,
            'feature_pre_filter': False,
    }

    params.update(fixed_params)
    
    print('Hyperparameters :', params)    

    lgb_model = lgb.train(params=params, 
                        train_set=dtrain,
                        num_boost_round=2000,
                        valid_sets=dvalid,
                        feval=auprc, 
                        early_stopping_rounds=300, 
                        verbose_eval=False,)
    
    preds = lgb_model.predict(X_valid) 
    score = average_precision_score(y_valid, preds)
    print(f'AUPRC : {score}\n')
    return score

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    X_, Y_ = X.values.astype(float), Y.values.flatten().astype(bool)

    if not os.path.exists(f'./bestparams.json'):    
        oof_max_params = []
        oof_max_scores = []
        for idx, (train_idx, valid_idx) in enumerate(folds.split(X_, Y_)):

            print('#'*40, f'Fold {idx+1} / Folds {folds.n_splits}', '#'*40)

            X_train, y_train = X_[train_idx], Y_[train_idx]
            X_valid, y_valid = X_[valid_idx], Y_[valid_idx]

            dtrain = lgb.Dataset(X_train, y_train)
            dvalid = lgb.Dataset(X_valid, y_valid)
            
            optimizer = BayesianOptimization(f=eval_function, 
                                            pbounds=param_bounds, 
                                            random_state=SEED)

            optimizer.maximize(init_points=OPT_INITS, n_iter=OPT_ITERS)
            
            max_params = optimizer.max['params']
            min_score = optimizer.max['target']
            oof_max_params.append(max_params)
            oof_max_scores.append(min_score)
            
            print(optimizer.max)

        max_params = oof_max_params[np.argmax(oof_max_scores)]

        max_params['num_leaves'] = int(round(max_params['num_leaves']))
        max_params['min_child_samples'] = int(round(max_params['min_child_samples']))

        max_params.update(fixed_params)
        
        with open(f'./bestparams.json', 'w') as f:
            json.dump(max_params, f)
    else:
        with open(f'./bestparams.json', 'r') as f:
            max_params = json.load(f)
            print('Loaded best parmas :', max_params)
    
    final_iter=2000
    if not os.path.exists(f'bestmodel.txt'):
        best_iters=[]
        for idx, (train_idx, valid_idx) in enumerate(folds.split(X_, Y_)):
            print('#'*40, f'Fold {idx+1} / Folds {folds.n_splits}', '#'*40)
            
            X_train, y_train = X_[train_idx], Y_[train_idx]
            X_valid, y_valid = X_[valid_idx], Y_[valid_idx]

            dtrain = lgb.Dataset(X_train, y_train)
            dvalid = lgb.Dataset(X_valid, y_valid)
            
            lgb_model = lgb.train(params=max_params,   
                            train_set=dtrain,   
                            num_boost_round=2000, 
                            valid_sets=dvalid, 
                            feval=auprc,        
                            early_stopping_rounds=300,
                            ) 

            best_iter = lgb_model.best_iteration
            best_iters.append(best_iter)
            print('done')
        
        final_iter = int(np.mean(best_iters))
        print(f'The final best iteration for lgb model might be {final_iter}')

        dtest = lgb.Dataset(x_test)

        lgb_model = lgb.train(params=max_params, 
                            train_set=bayes_dtrain,
                            num_boost_round=final_iter)
    
        lgb_model.save_model(f'bestmodel.txt')
        print('model saved')
    else:
        lgb_model = lgb.Booster(model_file=f'bestmodel.txt')
        print('model loaded')

    y_prob = lgb_model.predict(x_test,  num_iteration=final_iter)

    auroc = roc_auc_score(y_test, y_prob)
    print(f'AUROC score {auroc}')
    
    auprc = average_precision_score(y_test, y_prob)
    print(f'AUPRC score {auprc}')
    
if __name__ == '__main__':
    main()