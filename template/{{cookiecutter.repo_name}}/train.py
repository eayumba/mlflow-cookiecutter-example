# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, {{cookiecutter.eval_met3}}_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    {{cookiecutter.eval_met1}} = np.sqrt(mean_squared_error(actual, pred))
    {{cookiecutter.eval_met2}} = mean_absolute_error(actual, pred)
    {{cookiecutter.eval_met3}} = {{cookiecutter.eval_met3}}_score(actual, pred)
    return {{cookiecutter.eval_met1}}, {{cookiecutter.eval_met2}}, {{cookiecutter.eval_met3}}



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    {{cookiecutter.hp1}} = float(sys.argv[1]) if len(sys.argv) > 1 else {{cookiecutter.hp1_default}}
    {{cookiecutter.hp2}} = float(sys.argv[2]) if len(sys.argv) > 2 else {{cookiecutter.hp2_default}}

    with mlflow.start_run():
        lr = ElasticNet({{cookiecutter.hp1}}={{cookiecutter.hp1}}, {{cookiecutter.hp2}}={{cookiecutter.hp2}}, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        ({{cookiecutter.eval_met1}}, {{cookiecutter.eval_met2}}, {{cookiecutter.eval_met3}}) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model ({{cookiecutter.hp1}}=%f, {{cookiecutter.hp2}}=%f):" % ({{cookiecutter.hp1}}, {{cookiecutter.hp2}}))
        print("  RMSE: %s" % {{cookiecutter.eval_met1}})
        print("  MAE: %s" % {{cookiecutter.eval_met2}})
        print("  R2: %s" % {{cookiecutter.eval_met3}})

        mlflow.log_param("{{cookiecutter.hp1}}", {{cookiecutter.hp1}})
        mlflow.log_param("{{cookiecutter.hp2}}", {{cookiecutter.hp2}})
        mlflow.log_metric("{{cookiecutter.eval_met1}}", {{cookiecutter.eval_met1}})
        mlflow.log_metric("{{cookiecutter.eval_met3}}", {{cookiecutter.eval_met3}})
        mlflow.log_metric("{{cookiecutter.eval_met2}}", {{cookiecutter.eval_met2}})

        # If, for example, you wrote some training output besides the model into a file, use log_artifact
        # mlflow.log_artifact("artifact", artifact_file)

        mlflow.sklearn.log_model(lr, "model")
