"""Train a model, while logging metrics to mlflow and storing the result as an artifact.
(code adapted from https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_diabetes/linux/train_diabetes.py)
"""

import time
import mlflow

# Configure mlflow to automatically log metrics and the trained model
# NOTE: it needs to be done before importing sklearn metrics to work properly
#pylint: disable=wrong-import-position
mlflow.sklearn.autolog(log_models=True)

import hydra
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import datasets
from omegaconf import DictConfig

# Evaluate metrics
def eval_metrics(actual, pred):
    """Evaluate metrics by comparing actual and predicted values.
    """

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

@hydra.main(version_base="1.2", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """main function
    """

    # Wrap everything in a mlflow run
    with mlflow.start_run():

        # Set random seed
        np.random.seed(cfg.random_seed)
        mlflow.log_param("random_seed", 42)

        # Load dataset
        #pylint: disable=no-member
        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target

        # Create pandas DataFrame for sklearn ElasticNet linear_model
        Y = np.array([y]).transpose()
        cols = diabetes.feature_names + ["target"]
        data = pd.DataFrame(np.concatenate((X, Y), axis=1), columns=cols)

        # Split the data into training and test sets
        train, test = train_test_split(data)

        # The target is a quantitative measure of disease progression one year after baseline
        train_x = train.drop(["target"], axis=1)
        train_y = train[["target"]]
        test_x = test.drop(["target"], axis=1)
        test_y = test[["target"]]

        # Run ElasticNet
        start_time = time.time()
        model = ElasticNet(
            alpha=cfg.params.alpha,
            l1_ratio=cfg.params.l1_ratio,
            random_state=cfg.random_seed
        )
        model.fit(train_x, train_y)
        mlflow.log_metric("fit_time", time.time() - start_time)
        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        mlflow.log_metric("root_mean_squared_error_test_x", rmse)

        # Print out ElasticNet model metrics
        print(f"""Elasticnet model (alpha={cfg.params.alpha}, l1_ratio={cfg.params.l1_ratio}):
            RMSE: {rmse}
            MAE: {mae}
            R2: {r2}"""
        )

if __name__ == "__main__":
    #pylint: disable=no-value-for-parameter
    main()
