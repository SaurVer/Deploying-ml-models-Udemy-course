import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

#temporal elapsed time transformer
class TemporalVariableTransformers(BaseEstimator, TransformerMixin):

    def __init__(self, variables, reference_variable) :
       
       if not isinstance(variables, list):
           raise ValueError('The input is not a list, give list')
       
       self.variables= variables
       self.reference_variable= reference_variable

    
    def fit(self, X, y=None):
        return self
        #this step is required to make compatible with sklearn pipeline

    def transform(self,X,y=None):
        
        X=X.copy()

        for var in self.variables:
            X[var]= X[self.reference_variable]-X[var]

        return X

#mapper pf categorical values
class Mapper(BaseEstimator,TransformerMixin):

    def __init__(self, variables, mappings):
        self.variables= variables
        self.mappings= mappings

    def fit(self,X,y=None):
        return self
    

    def transform(self,X,y=None):
        X=X.copy()

        for var in self.variables:
            X[var]= X[var].map(self.mappings)

        return X




#numerical missing value imputer
#class that learns and then transforms

class MissingInputer(BaseEstimator,TransformerMixin):

    def __init__(self, variables):
        self.variables= variables

    def fit(self,X,y=None):
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X,y=None):

        for var in self.variables:
            X[var].fillna(self.imputer_dict_[var], inplace= True)

        return X
    


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables, threshold=.05):
        self.threshold = threshold
        self.variables = variables

    def fit(self, X,y=None):
        self.encoder_dict_ = {}

        for var in self.variables:
            t= pd.Series(X[var].value_count(normalize= True))

            self.encoder_dict_[var]= list(t[t < self.threshold].index)
        return self

    
    def transform(self, X,y=None):
        X= X.copy()
        for var in self.variables:
            X[var]= np.where(~X[var].isin(self.encoder_dict_[var]),X[var], 'Rare')


# Ordinal encoding- Most interesting

class OrdinalCategoryEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    

    def fit(self,X,y):
        df= pd.concat([X,y], axis=1)
        df.columns= list(X.columns)+ 'target'

        for var in self.variables:
            t= df.groupby[var]["target"].mean().sort_values(ascending= True).index
            self.encoder_dict_[var]= {val:i for i, val in enumerate(t,0)}
        return self
    

    def transform(self, X,y=None):
        for var in self.variables:
            X[var]= X[var].map(self.encoder_dict_[var])
        return X




        


