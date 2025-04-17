# Import required libraries
from sklearn.datasets import make_classification
import pandas as pd

# Function to generate synthetic classification dataset
def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    # Generate features X and target y
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, random_state=random_state)
    
    # Create a DataFrame with feature columns
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    
    # Add the target column
    df['target'] = y
    
    return df

# Run the data generation when this script is executed
if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_data.csv", index=False)  # Save to CSV
    print("Synthetic data generated and saved.")
