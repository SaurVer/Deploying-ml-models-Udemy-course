from regression_model.config.core import config
from regression_model.preprocessing.features import TemporalVariableTransformation


def test_temporal_variable_transformer(sample_input_data):
    #Given 
    transformer = TemporalVariableTransformation(variables=config.model_config.temporal_vars,
                                                 reference_variable=config.model_config.ref_var
                                                 )
    
    assert sample_input_data["YearRemodAdd"].iat[0] == 1961

    subject = transformer.fit_transform(sample_input_data)

    assert subject["YearRemodAdd"].iat[0]==49