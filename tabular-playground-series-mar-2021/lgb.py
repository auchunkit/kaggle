#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split

from xgboost import DMatrix
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor, Pool

import optuna

from sklearn.metrics import roc_auc_score


# In[ ]:


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


# In[ ]:


y = train.target
train = train.drop(['id','target'],axis=1)


# In[ ]:


cat_features = [col for col in train.columns if 'cat' in col]

for col in cat_features: 
    le = LabelEncoder()
    le.fit(list(train[col].values) + list(test[col].values))
    train[col] = le.transform(train[col].values)
    test[col] = le.transform(test[col].values)


# In[ ]:


def objective_lgbm(trial,data=train, target=y):
    x_train, x_validation, y_train, y_validation = train_test_split(data,
                                                        target, 
                                                        test_size=0.2,
                                                        random_state=1688)
    lgbm_params = {'max_depth': trial.suggest_int('max_depth',5,20),
               'subsample': trial.suggest_float('subsample',0.3,1,step=0.1), 
               'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3,1, step=0.1),
               'learning_rate': trial.suggest_float('learning_rate',0.006,0.01,step=0.002),
               'reg_lambda': trial.suggest_loguniform('reg_lambda',1e-3,100),
               'reg_alpha': trial.suggest_loguniform('reg_alpha',1e-3,10),
               'min_child_samples': trial.suggest_int('min_child_samples',1,300), 
               'num_leaves': trial.suggest_int('num_leaves',2,100, step=2),
               'max_bin': trial.suggest_int('max_bin',10,800),
               'cat_smooth': trial.suggest_int('cat_smooth',10,100),
               'cat_l2': trial.suggest_loguniform('cat_l2',1e-3,100),
               'metric': 'auc',
               'n_jobs': -1, 
               'n_estimators': 200000
              }
    model = LGBMClassifier(**lgbm_params)
    model.fit(x_train,
               y_train,
               eval_set=[(x_validation, y_validation)],
               eval_metric=['auc'],
               early_stopping_rounds=2000, 
               verbose=0)
    y_pred = model.predict_proba(x_validation)[:,1]
    score = roc_auc_score(y_validation, y_pred)
    return score


# In[ ]:


if False:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_lgbm,n_trials=100)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)


# In[ ]:


lgbm_params = {'max_depth': 14,
           'subsample': 0.7, 
           'colsample_bytree': 0.3,
           'learning_rate': 0.006,
           'reg_lambda': 26.696886704344863,
           'reg_alpha': 4.6998938209396615,
           'min_child_samples': 143, 
           'num_leaves': 82,
           'max_bin': 382,
           'cat_smooth': 79,
           'cat_l2': 4.379086950854495,
           'metric': 'auc',
           'n_jobs': -1, 
           'n_estimators': 200000
          }


# In[ ]:


kfold = KFold(n_splits = 10, shuffle = True, random_state = 1688)
test_preds = []
scores = []
x_test = test.copy().drop(['id'],axis=1)
for fold, (train_idx, test_idx) in enumerate(kfold.split(train)):
    x_train, x_validation = train.iloc[train_idx], train.iloc[test_idx]
    y_train, y_validation = y.iloc[train_idx], y.iloc[test_idx]

    
    model = LGBMClassifier(**lgbm_params)
    model.fit(x_train,
               y_train,
               eval_set=[(x_validation, y_validation)],
               eval_metric=['auc'],
               early_stopping_rounds=2000, 
               categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18],
               verbose=0)
    y_pred = model.predict_proba(x_validation)[:,1]
    score = roc_auc_score(y_validation, y_pred)
    print("Fold-%s, AUC:%s" %(fold, score))
    scores.append(score)
    
    y_test_preds = model.predict_proba(x_test)[:,1]
    test_preds.append(y_test_preds)


# In[ ]:


print("Scores list: ", scores)
print("Mean of scores: %s" %(np.mean(scores)))


# In[ ]:


test['target'] = np.mean(test_preds, axis=0)


# In[ ]:


test[["id","target"]].to_csv("lightgbm_submission.csv",index=False)

