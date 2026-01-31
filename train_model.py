import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    "role": [0,1,2,1,0,2,2,1,0,2],
    "screen": [3,2,3,1,0,3,2,1,0,3],
    "exercise": [0,1,0,2,3,0,1,2,3,0],
    "lifestyle": [2,1,2,0,0,2,1,0,0,2],
    "stress": [2,1,2,0,0,2,1,0,0,2]
}

df = pd.DataFrame(data)
X = df.drop("stress", axis=1)
y = df["stress"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "stress_model.pkl")
print("Model trained and saved!")
