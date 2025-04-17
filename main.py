# Import functions from other scripts
from data.generate_data import generate_synthetic_data
from models.train_model import train_model

# Main function to run the full pipeline
if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_data.csv", index=False)

    print("Training machine learning model...")
    train_model()
