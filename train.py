import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import joblib


if __name__ == "__main__":

    print("training model...")
    
    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False) # We won't log models right away

    # Import dataset
    df = pd.read_csv("data/Salary_Data.csv")

    # X, y split 
    X = df.loc[:, ["YearsExperience"]]
    y = df.loc[:, ["Salary"]]

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Log experiment to MLFlow
    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)

        # Log model seperately to have more flexibility on setup 
        mlflow.sklearn.log_model(sk_model=model, artifact_path="salary_estimator")

    # If you want to persist model locally
    #joblib.dump(model, "model.joblib")

    print("...Done!")
    print(f"---Total training time: {time.time()-start_time}")