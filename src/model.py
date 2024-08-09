from sklearn.linear_model import LinearRegression

def build_model():
    return LinearRegression()

def train_model(model, X, y):
    model.fit(X, y)
    return model