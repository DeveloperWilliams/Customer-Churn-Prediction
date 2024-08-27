# churn_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ChurnPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def preprocess_data(self):
        self.data['Churn'] = self.data['Churn'].map({'Yes': 1, 'No': 0})
        self.X = self.data.drop(['Churn'], axis=1)
        self.y = self.data['Churn']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

# Example usage
if __name__ == "__main__":
    churn_predictor = ChurnPredictor("customer_data.csv")
    churn_predictor.preprocess_data()
    churn_predictor.train_model()
