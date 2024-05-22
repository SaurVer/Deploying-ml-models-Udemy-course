from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from regression_model.config.core import config
from regression_model.preprocessing import features as pp


price_pipe = Pipeline(
    [
        #imputation
        # impute categorical variables with string missing
        ("missing_imputation",
         CategoricalImputer(imputation_method='missing',
                            variables=config.model_config.categorical_vars_with_na_missing
                            ),
        ),


        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method='frequent',
                variables=config.model_config.categorical_vars_with_na_frequent
            )
        ),

        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars_with_na
                                ,)
        ),

        (
            "missing_imputation",
            MeanMedianImputer(variables=config.model_config.numerical_vars_with_na)
        ),

        (
            "elapsed_time",
            pp.TemporalVariableTransformation(variables= config.model_config.temporal_vars , reference_variable=config.model_config.ref_var)
        ),

        (
            "drop_features",
            DropFeatures(features_to_drop=[config.model_config.ref_var])
        ),

        (
            "log_transformation",
            LogTransformer(variables=config.model_config.numericals_log_vars)
        ),


        (
            "binarizer",
            SklearnTransformerWrapper(
                transformer = Binarizer(threshold=0),
                variables= config.model_config.binarize_vars,
            ),
        ),
        #mappers

        (
            "mapping_exposure",
            pp.Mapper(variables= config.model_config.exposure_vars,
                      mapping=config.model_config.exposure_mappings

                    ),
        ),
        (
            "mapper_finish",
            pp.Mapper(
                variables=config.model_config.finish_vars,
                mappings=config.model_config.finish_mappings,
            ),
        ),
        (
            "mapper_garage",
            pp.Mapper(
                variables=config.model_config.garage_vars,
                mappings=config.model_config.garage_mappings,
            ),
        ),
        (
            "mapper_qual",
            pp.Mapper(
                variables=config.model_config.qual_vars,
                mappings=config.model_config.qual_mappings,
            ),
        ),
        #Rare Label
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol= 0.01, n_categories=1, variables=config.model_config.categorical_vars
            )
        ),
        (
            'Ordinal Encoding',
            OrdinalEncoder(
                encoding_method="ordered",
                variables=config.model_config.categorical_vars,
            )
        ),
        (
            "scaler", MinMaxScaler()
        ),

        (
            "lasso",
            Lasso(alpha= config.model_config.alpha,
                  random_state=config.model_config.random_state,)
         
         ),


    ]

)