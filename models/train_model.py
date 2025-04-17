# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Function to train a classifier using synthetic data
def train_model(data_path="data/synthetic_data.csv"):
    # Load dataset
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Print evaluation metrics
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the trained model
    joblib.dump(clf, "models/classifier.pkl")
    print("Model saved to models/classifier.pkl")

# Run model training when this script is executed
if __name__ == "__main__":
    train_model()
