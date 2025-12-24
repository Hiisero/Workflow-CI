import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn


def main():
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart Disease Classification - Tuning V2")
    
    df = pd.read_csv("heart_preprocessing.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model & Hyperparameter Tuning
    base_model = LogisticRegression(max_iter=1000)

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="f1"
    )

    with mlflow.start_run():
        # Train
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predict
        y_pred = best_model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Manual logging
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_param("best_C", grid_search.best_params_["C"])
        mlflow.log_param("best_solver", grid_search.best_params_["solver"])
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("model_type", "LogisticRegression")

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

        print("Best Parameters :", grid_search.best_params_)
        print("Accuracy        :", accuracy)
        print("Precision       :", precision)
        print("Recall          :", recall)
        print("F1-score        :", f1)


if __name__ == "__main__":
    main()

