import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('FULL_DATA_SET_CODED_out_2022.csv')
print(df.columns.tolist())
# Separate features and labels
y = df['outcome']
X = df.drop(columns=['outcome'])  # Drop non-feature columns

# Train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Save the trained model
joblib.dump(clf, 'NLP_RFtrained.joblib')

with open("rf_features.txt", "w") as f:
    for col in X.columns:
        f.write(f"{col}\n")

print("Model trained and saved as NLP_RFtrained.joblib")
