"""
Every test session runs in an isolated temporary project root with its own
SQLite MLflow store, so tests never touch your real data or models.
Model sizes are reduced to keep the suite fast.
"""

import os
import shutil
import sys
import warnings
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("platform")
    shutil.copytree(REPO / "config", root / "config")
    (root / "data" / "raw").mkdir(parents=True)
    shutil.copy(REPO / "data" / "raw" / "raw_data.csv", root / "data" / "raw" / "raw_data.csv")
    shutil.copytree(REPO / "samples", root / "samples")

    settings_file = root / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_file.read_text())
    settings["models"]["random_forest"]["n_estimators"] = 30
    settings["models"]["xgboost"]["n_estimators"] = 40
    settings["models"]["neural_network"]["max_iter"] = 200
    settings_file.write_text(yaml.safe_dump(settings))

    os.environ["MLOPS_PROJECT_ROOT"] = str(root)
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    return root


@pytest.fixture(scope="session")
def trained(project):
    """Preprocessed data + one champion per disease + one clean batch each."""
    from pipeline import setup

    return setup(create_first_batch=True)
