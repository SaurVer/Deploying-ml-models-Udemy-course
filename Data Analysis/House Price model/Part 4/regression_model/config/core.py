from pathlib import Path
from typing import Dict, List,Optional,Sequence

#Optional: This represents an optional type in Python. It's used to indicate that a parameter or return value may be either of a certain type or None. For example, Optional[str] would represent either a string or None
#Sequence is for general type of list.

from pydantic import BaseModel
from strictyaml import YAML, load

import regression_model #this is the same folder

#Project Directories
PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
ROOT= PACKAGE_ROOT.parent
CONFIG_FILE_PATH= PACKAGE_ROOT / "config.yml"
DATASET_DIR= PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR= PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application Level Config.
    """
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str



class ModelConfig(BaseModel):
    """
    Model level configuration; all configuration 
    related to model training and feature engineering.
    """
    target: str
    variables_to_rename= Dict
    features: List[str]
    test_size= float
    random_state: int
    alpha= float
    categorical_vars_with_na_frequent: List[str]
    categorical_vars_with_na_missing: List[str]
    numerical_vars_with_na: List[str]
    temporal_vars: List[str]
    ref_var: str
    numericals_log_vars: Sequence[vars]
    binarize_vars: Sequence[str]
    qual_vars: List[str]
    exposure_vars: List[str]
    finish_vars: List[str]
    garage_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]
    exposure_mappings: Dict[str, int]
    garage_mappings: Dict[str, int]
    finish_mappings: Dict[str, int]

class Config(BaseModel):
    """ Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path: #return type hint
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path= find_config_file()
    
    if cfg_path:
        with open(cfg_path, 'r') as config_file:
            parse_config= load(config_file.read())
            return parse_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML =None) -> Config:
    """Run validation on config values."""

    if parsed_config is None:
        parsed_config= fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config

config = create_and_validate_config()