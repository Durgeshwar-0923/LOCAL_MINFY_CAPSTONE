# src/utils/mlflow_logger.py
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.metrics import mean_absolute_scaled_error
import matplotlib.pyplot as plt
import os
import joblib


def log_model_run(model, X_train, X_valid, y_train, y_valid, model_name, trial_params=None):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, y_pred)
        mse = mean_squared_error(y_valid, y_pred)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        r2 = r2_score(y_valid, y_pred)
        mase = mean_absolute_scaled_error(y_valid, y_pred, y_train_mean=np.mean(np.abs(np.diff(y_train))))

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MASE", mase)

        if trial_params:
            mlflow.log_params(trial_params)

        # Save model artifact
        model_path = f"outputs/models/{model_name}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Residual Plot
        plt.figure()
        residuals = y_valid - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residual Plot")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plot_path = f"outputs/plots/{model_name}_residuals.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        return {
            "model": model,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mase": mase
        }
