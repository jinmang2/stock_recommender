import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
from sklearn import metrics

xgb_clf = XGBClassifier(random_state=42)
alg = XGBClassifier(
 learning_rate =0.15,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=2,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

def modelfit(alg, dtrain, predictors, target, useTrainCV = True,
             cv_folds = 5, early_stopping_rounds = 50, top = 10,
             figure = True, model_report = True):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    if model_report:
        #Print model report:
        print("Model Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    imp_fear_name = pd.Series(predictors)[alg.feature_importances_ > 0.01]
    imp_fear_name.index = range(len(imp_fear_name))
    imp_fear = pd.concat((imp_fear_name,
               pd.Series(alg.feature_importances_[alg.feature_importances_ > 0.01])),axis=1).sort_values(by=1,ascending=True)
    imp_fear.index = range(len(imp_fear))
    imp_fear.columns = ["feature_name", "feature_importance"]
#     {i:j for i,j in zip(imp_fear.index, imp_fear["feature_name"])}
    if figure:
        if top:
            imp_fear.iloc[-top:].plot(y='feature_importance', x='feature_name', kind='barh', legend=False, figsize=(8,5))
        else:
            imp_fear.plot(y='feature_importance', x='feature_name', kind='barh', legend=False, figsize=(8,5))

#     feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances', figsize=(200,5))
#     plt.ylabel('Feature Importance Score')

    return alg, imp_fear

def temp_calc_feature_importance(alg, train, test, predictors, target):
    test_result = {}
    fi_dict = {}
    for i in train.keys():
        model, fear_importance = modelfit(alg, train[i],
                                            predictors=predictors,
                                            target=target,
                                                  cv_folds=5, early_stopping_rounds=20,
                                                  figure=False, model_report=False)
        pred = model.predict(test[i][predictors])
        # Calculating Score
        test_result[i] = np.array([
            metrics.accuracy_score(pred, test[i][target]),
            metrics.f1_score(pred, test[i][target]),
            metrics.precision_score(pred, test[i][target]),
            metrics.recall_score(pred, test[i][target]),
            metrics.roc_auc_score(pred, test[i][target])
        ])
        fi_dict[i] = {feature:value for feature,value in fear_importance.iloc[-10:].values}
    return test_result, fi_dict
