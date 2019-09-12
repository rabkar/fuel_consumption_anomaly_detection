import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class EnsureDataTypes(BaseEstimator):

    def __init__(self, features, dtypes=None):
        self.features = features
        self.dtypes = dtypes
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.dtypes is None:
            X.loc[:, self.features] = X.loc[:, self.features].astype(np.double)
        return X


class ConvertMinutesToHours(BaseEstimator):

    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for c in self.features:
            X.loc[:,c] = X.loc[:,c].astype(float)/60.0
        return X

class DeriveFeatures(BaseEstimator):

    def __init__(self, functions):
        self.functions = functions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if len(self.functions) > 1:
            for f in self.functions:
                f(X)
        else:
            self.functions[0](X)
        return X


class WorkingBehaviorCategorization(BaseEstimator):

    def __init__(self, category_matrix, mode_threshold, prod_threshold):
        self.category_matrix = category_matrix
        self.mode_threshold = mode_threshold
        self.prod_threshold = prod_threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['working_char'] = X.apply(
            lambda x: self.categorize_working_behaviors(
                mode=x["delta_pe_mode"], 
                prod=x["daily_productivity"], 
                matrix=self.category_matrix,
                mode_thres=self.mode_threshold,
                prod_thres=self.prod_threshold
            ), axis =1 
        )
        return X

    def categorize_working_behaviors(self, mode, prod, matrix, mode_thres, prod_thres):
        lmt, umt = mode_thres
        lpt, upt = prod_thres

        if mode >= umt:
            working_mode = 2
        elif mode >= lmt:
            working_mode = 1
        else:
            working_mode = 0
        
        if prod >= upt:
            working_load = 2
        elif prod >= lpt:
            working_load = 1
        else:
            working_load = 0
        
        return matrix[working_load, working_mode] 


class ZTestAnomaly(StandardScaler):

    def __init__(self, features, direction, threshold=(-2,2), min_sample=20):
        super().__init__(self)
        self.features = features
        self.direction = direction
        self.threshold = threshold
        self.min_sample = min_sample
        self.num_sample = 0

    def fit(self, X, y=None):
        self.num_sample = len(X)
        if len(X) >= self.min_sample:
            super().fit(X[self.features])
        return self
    
    def transform(self, X, y=None):
        for i, c in enumerate(self.features):
            if self.num_sample >= self.min_sample:
                if self.var_[i] > 0:
                    Z = (X.loc[:,c] - self.mean_[i])/np.sqrt(self.var_[i])
                else:
                    Z = (X.loc[:,c] - self.mean_[i])
                    
                X["{}_anomaly".format(c)] = [
                    self.check_if_anomaly(
                        z, direction=self.direction, thres=self.threshold) 
                        for z in Z]
            else:
                X["{}_anomaly".format(c)] = None
        return X
    
    def check_if_anomaly(self, z, direction, thres):
        lt, ut = thres if type(thres)==type(()) else (thres, thres)
        
        if direction=='both':
            if z >= ut:
                return "upper anomaly"
            elif z >= lt:
                return None
            else:
                return "lower anomaly"
            
        elif direction == 'right':
            return ("upper anomaly" if z > ut else None)
        
        elif direction == 'left':
            return ("lower anomaly" if z < lt else None)


class FuelAnomalyDetection(BaseEstimator):

    def __init__(self, features, target, anomaly_detector, regressor, min_sample=20, anomaly_distance=-5):
        self.features = features
        self.target = target
        self.anomaly_detector = [clone(anomaly_detector), clone(anomaly_detector), clone(anomaly_detector)]
        self.regressor = [clone(regressor), clone(regressor), clone(regressor)]
        self.working_categories = ['light', 'moderate', 'heavy']
        self.min_sample = min_sample
        self.num_sample = [0, 0, 0]
        self.anomaly_distance = anomaly_distance if anomaly_distance < 0 else -1*anomaly_distance
    
    def fit(self, X, y=None):
        for i, cat in enumerate(self.working_categories):
            idx = X[X['working_char']==cat].index
            self.num_sample[i] = len(idx)
            if self.num_sample[i] >= self.min_sample:
                self.anomaly_detector[i].fit(X.loc[idx, self.features + [self.target]])
                self.regressor[i].fit(X=X.loc[idx, self.features], 
                                      y=X.loc[idx, self.target])
        return self
    
    def transform(self, X, y=None):
        for i, cat in enumerate(self.working_categories):
            idx = X[X['working_char']==cat].index
            if self.num_sample[i] >= self.min_sample and len(idx) > 0:
                label = {0:'low', 1:'good', -1:'high', -2:'very high'}
                # predict anomaly and mark anomaly as 'rather high'
                X.loc[idx, 'fuel_index'] = \
                    self.anomaly_detector[i].predict(X.loc[idx, self.features + [self.target]])

                # compute distance of the anomaly
                X.loc[idx, 'distance'] = \
                    self.anomaly_detector[i].decision_function(X.loc[idx, self.features + [self.target]])
                
                # use the distance to correct label 'rather high' to be 'high'
                X.loc[idx, 'fuel_index'] = \
                    X.loc[idx].apply(
                        lambda x: -2 if x['distance'] < self.anomaly_distance else x['fuel_index'], axis=1)
                
                # if the anomaly is bellow linear regression curve, correct label to 'good'
                X.loc[idx, 'fuel_index'] = \
                    X.loc[idx].apply(
                        lambda x: 1 if x[self.target] <= self.regressor[i].predict(x[self.features].values.reshape(-1, 1)) 
                            else x['fuel_index'], axis=1)
                
                # if the anomaly is bellow linear regression curve, but distance is too far, correct label to "low"
                X.loc[idx, 'fuel_index'] = \
                    X.loc[idx].apply(
                        lambda x: 0 if (x['fuel_index']==1 and 
                                        x['distance']< self.anomaly_distance and 
                                        x[self.target] <= self.regressor[i].predict(x[self.features].values.reshape(-1, 1)))
                                    else x['fuel_index'], axis=1)
                
                # Finally convert integer label to text label
                X.loc[idx, 'fuel_index'] = X.loc[idx, 'fuel_index'].map(lambda x: label.get(x))
                
            else:
                X.loc[idx, 'fuel_index'] = 'good'
        return X
    

class ColumnSelector(BaseEstimator):
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.columns].values


class MapIndustrySector(BaseEstimator):
    
    def __init__(self, sector_map=None):
        if sector_map is None:
            self.sector_map = \
                { "AGR": "AGR", "CON": "CON", "FOR": "FOR", "GOV": "OTH", 
                "MIM": "OTH", "MNG": "MNG", "MNS": "MNG", "SOF": "OTH",
                "SON": "OTH"}
        else:
            self.sector_map = sector_map

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if 'industry_sector' in X.columns:
            X['industry_sector'] = X['industry_sector'].map(
                lambda x: self.sector_map.get(x) if x in self.sector_map.keys() else 'OTH' )
        else:
            X['industry_sector'] = 'OTH'
        return X