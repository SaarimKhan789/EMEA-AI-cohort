import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Sample Insurance Purchase Dataset
data = {
    'age':[25,35,45,20,30,50,23,40,60,48,32,28,55,38],
    'income':[30000,50000,80000,20000,40000,90000,25000,65000,100000,85000,55000,35000,95000,60000],
    'purchased':[0,0,1,0,0,1,0,1,1,1,0,0,1,1]
}

df = pd.DataFrame(data)

X = df[['age','income']]
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")