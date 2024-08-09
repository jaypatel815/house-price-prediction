import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_loader import load_data
from preprocess import preprocess_data
from model import build_model, train_model
from evaluate import evaluate_model
from visualize import visualize_results

def main():
    # 1. Load the data
    print("Loading data...")
    df = load_data()

    # 2. Preprocess the data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()

    # 3. Build and train the model
    print("Building and training the model...")
    model = build_model()
    model = train_model(model, X_train, y_train)

    # 4. Evaluate the model
    print("Evaluating the model...")
    mse, rscore, predictions = evaluate_model(model, X_test, y_test)
    print(mse, rscore, predictions)

    # 5. Visualize the results
    print("Visualizing the results...")
    visualize_results(y_test, predictions)
    
    print("Pipeline execution complete!")

if __name__ == "__main__":
    main()