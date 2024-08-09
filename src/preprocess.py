from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_data


def preprocess_data():
    # Load the data
    df = load_data()

    # Handle missing values
    df.dropna(inplace=True)

    # Split the data into features (X) and target variable (y)
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling (normalization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test