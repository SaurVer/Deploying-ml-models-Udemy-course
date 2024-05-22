from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalVariableTransformation(BaseEstimator, TransformerMixin):

    def __init__(self,variables: List[str], reference_variable: str):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        
        self.variables= variables
        self.reference_variable = reference_variable
        
    def fit(self, X: pd.DataFrame, y: pd.Series= None):
        return self
    

    def transform(self, X: pd.Dataframe) -> pd.DataFrame:

        df= X.copy()

        for features in self.variables:
            df[features]= df[self.reference_variable]- df[features] 
        
        return df
    



class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str], mapping: dict):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        
        self.variables = variables
        self.mapping = mapping
        
    def fit(self,X: pd.DataFrame, y: pd.Series = None ):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        df= X.copy()
        
        for feature in self.variables:
            df[feature]= df[feature].map(self.mapping)

        return df


