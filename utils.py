import optuna
#import xgboost as xgb
#import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
from sklearn.base import BaseEstimator

class trials:
    def __init__(self, X, y, score = None, folds = 1, transformation= None):
        self.X = X
        self.y = y
        self.score = score
        self.folds = folds
        self.transformation = transformation

        if transformation is not None:
            self.X_proc = transformation.fit_transform(X)

    def objxgb(self,trial):
        dtrain = xgb.DMatrix(self.X_proc, label=self.y)
        
        params = dict(
            verbosity=0,
            objective="reg:squarederror",
            eval_metric="rmse",
            eta=trial.suggest_float("eta", 0.05, 0.5),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 5),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.8, 1),
            subsample=trial.suggest_float("subsample", 0.55, 0.95),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
        )
    
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
        history = xgb.cv(
            params,
            dtrain,
            num_boost_round= trial.suggest_int("num_boost_round", 1000, 2500),
            early_stopping_rounds=50,
            folds=self.folds,
            stratified=False,
            shuffle=False,
            callbacks=[pruning_callback]
        )
        
        trial.set_user_attr("n_estimators", len(history))
        
        best_score = history["test-rmse-mean"].values[-1]
        return best_score

    def objsvp(self,trial):
        
        degree = trial.suggest_int('degree',2,5)
        epsilon = trial.suggest_float('epsilon', 0.009,0.05)
        gamma = trial.suggest_float("gamma", 0.00008,0.0005)
        C = trial.suggest_float("C", 70, 200 )
        coef0 = trial.suggest_float('coef0',-1,1) 
        
        clf = make_pipeline(StandardScaler(),SVR(kernel = 'poly', epsilon= epsilon, degree = degree, C=C, gamma= gamma, coef0= coef0))
        
        
        return self.score(clf, self.X, self.y)

    def objsvm(self,trial):
        
        epsilon = trial.suggest_float('epsilon', 0.009,0.05)
        gamma = trial.suggest_float("gamma", 0.00008,0.0005)
        C = trial.suggest_float("C", 70, 200 )
        
        clf =  make_pipeline(StandardScaler(),SVR(kernel='rbf', gamma = gamma, epsilon= epsilon, C=C))
        

        scr = self.score(clf, self.X, self.y)
            
        return scr

    def objlgbm(self,trial):
        
        dtrain = lgb.Dataset(self.X_proc, label=self.y)
        
        param = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
        gbm = lgb.cv(param, dtrain, num_boost_round= trial.suggest_int("num_boost_round",500, 2000),
                    folds= self.folds, callbacks=[pruning_callback])

        best_score = np.array(gbm['rmse-mean']).mean()
        return best_score




class ModelMix(BaseEstimator):
    
    def __init__(self, models= None, params= None):
        self.models = models
        self.params = params
        self.trained_models = {}
        
        
    def fit(self, X,y):
        if self.models is not None:
            for i in self.models:
                if i == 'xgb':
                    self.trained_models[i]= xgb.train({j: self.models[i][j] for j in self.models[i] if j!='num_boost_round'}, 
                    xgb.DMatrix(RobustScaler().fit_transform(X), label=y), num_boost_round =self.models[i]['num_boost_round'])
                elif i == 'lgbm':
                    self.trained_models[i]= lgb.train({j: self.models[i][j] for j in self.models[i] if j!='num_boost_round'}, 
                    lgb.Dataset(RobustScaler().fit_transform(X), label=y), num_boost_round =self.models[i]['num_boost_round'])
                else:
                    self.trained_models[i]= self.models[i].fit(X,y)
                    
    
    def predict(self, X):
        pred = 0
        for i in self.trained_models:
            if i == 'xgb':
                pred += self.trained_models[i].predict(xgb.DMatrix(RobustScaler().fit_transform(X)))*self.params[i]
            elif i == 'lgbm':
                pred += self.trained_models[i].predict(RobustScaler().fit_transform(X))*self.params[i]
            else:
                pred += self.trained_models[i].predict(X)*self.params[i]
        return pred   