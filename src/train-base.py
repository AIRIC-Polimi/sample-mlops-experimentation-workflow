"""Just train a model, without any fancy stuff.
(code adapted from https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_diabetes/linux/train_diabetes.py)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import datasets

# Evaluate metrics
def eval_metrics(actual, pred):
    """Evaluate metrics by comparing actual and predicted values.
    """

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    """main function
    """
    # Set random seed
    np.random.seed(42)

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
    model = ElasticNet(alpha=0.05, l1_ratio=0.05, random_state=42)
    model.fit(train_x, train_y)
    predicted_qualities = model.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    print(f"""Elasticnet model (alpha=0.05, l1_ratio=0.05):
        RMSE: {rmse}
        MAE: {mae}
        R2: {r2}"""
    )


if __name__ == "__main__":
    main()
