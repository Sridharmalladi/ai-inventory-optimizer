from xgboost import XGBRegressor

def train_xgboost_model(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=5):
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_demand(model, X):
    return model.predict(X)