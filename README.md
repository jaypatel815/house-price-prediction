# House Price Prediction

A machine learning project to predict house prices using linear regression.

This project aims to predict house prices using the California Housing Prices dataset, which is available through the Scikit-learn library. The project uses a machine learning model to predict the median house prices in California based on various features such as median income, housing age, and location.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Further Development](#further-development)

## Project Description

The House Price Prediction project is a machine learning pipeline that uses linear regression to estimate house prices. It loads, preprocesses, visualizes, and models the data from the California Housing Prices dataset. The pipeline is modular, with separate scripts for data loading, preprocessing, modeling, evaluation, and visualization, making it easy to maintain and expand.

## Dataset

The project uses the California Housing Prices dataset, which is available via Scikit-learn. The dataset contains the following features:

- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average number of household members
- **Latitude**: Block group’s latitude
- **Longitude**: Block group’s longitude

The target variable is the **median house value** for each block group.

## Installation

### Step 1: Set up the Virtual Environment

To ensure your development environment is clean and isolated, create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Activate the virtual environment (MacOS/Linux)
source venv/bin/activate
```

### Step 2: Install Required Libraries
Once your virtual environment is activated, install the necessary libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```
This will install all the dependencies required for the project, including Scikit-learn, Pandas, NumPy, and Matplotlib.

## Usage

### Running the Full Pipeline

To run the entire machine learning pipeline, execute the main.py script. This script will load the data, preprocess it, build and train the model, and evaluate its performance:

```bash
python main.py
```

### Example Functions

The project consists of multiple modular components, each encapsulated in different scripts within the src/ directory.

1. Data Loading (data_loader.py)

    ```bash
    from src.data_loader import load_data

    df = load_data()
    ```

2. Data Preprocessing (preprocess.py)

    ```bash 
    from src.preprocess import preprocess_data

    X_train, X_test, y_train, y_test = preprocess_data()
    ```

3. Model Training (model.py)
    ```bash
    from src.model import build_model, train_model

    model = build_model()
    model = train_model(model, X_train, y_train)
    ```

4. Evaluation (evaluate.py)
    ```bash
    from src.evaluate import evaluate_model

    evaluate_model(model, X_test, y_test)
    ```

5. Visualization (visualize.py)
    ```bash
    visualize_results(y_test, predictions)
    ```

## Model Evaluation

After building and training the model, it is evaluated using two key metrics:

- Root Mean Squared Error (RMSE): Measures the standard deviation of the residuals (prediction errors). The RMSE gives an indication of how well the model predicts the house prices.
    > ![screenshot](RMSE.png) 
 
- R-squared (R²) Score: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better model performance.

## Results

After training the linear regression model, the evaluation metrics showed the following performance:

- RMSE: X
- R² Score: Y

We also visualized the performance of the model using a scatter plot that compares the actual vs predicted house prices.

## Further Development
This project could be extended in several ways:

1. **Feature Engineering:** Adding more features or creating new features from the existing data could potentially improve the model’s performance.

2. **Model Tuning:** Hyperparameter tuning or using more advanced models like Decision Trees or Random Forests could lead to better accuracy.

3. **Exploratory Data Analysis (EDA):** More in-depth analysis and visualization of the data might uncover patterns that could be useful in improving the model.

4. **Cross-Validation:** Implementing cross-validation for more reliable performance metrics.

---

This project is a solid foundation for learning and building more complex machine learning models. Feel free to fork the repository, try out new ideas, and contribute to the project!

### Additional Notes:
- Be sure to update the RMSE and R² values in the "Results" section after you've trained your model and obtained these metrics.
- Include any key visualizations or plots generated from `visualize.py` to enhance the README if applicable.

Let me know if you'd like to adjust any parts or need further clarification!