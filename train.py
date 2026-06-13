import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
df = pd.read_csv('data/crop_recommendation.csv')

# 2. Preprocessing
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df.drop('label', axis=1)
y = df['label']

# 3. Scaling EVERYTHING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Define the 3 Required Algorithms
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=10, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print(f"{'Algorithm':<20} | {'Accuracy Score':<15}")
print("-" * 40)

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train) # Training on SCALED data
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<20} | {acc:.4%}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

# 5. Save EVERYTHING
with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"\nModel saved successfully! Best: {type(best_model).__name__}")