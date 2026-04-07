import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# =============================================
# STEP 1: Load Dataset
# =============================================
print("📂 Dataset load ho raha hai...")
df = pd.read_csv('Crop_recommendation.csv')

print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"📊 Crops: {df['label'].unique()}")
print(f"\nDataset Info:\n{df.describe()}")

# =============================================
# STEP 2: Features & Target
# =============================================
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

print(f"\n✅ Features: {list(X.columns)}")
print(f"✅ Total crops to predict: {y.nunique()}")

# =============================================
# STEP 3: Train-Test Split
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")

# =============================================
# STEP 4: Feature Scaling
# =============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# STEP 5: Train Random Forest Model
# =============================================
print("\n🤖 Model train ho raha hai...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# =============================================
# STEP 6: Evaluate Model
# =============================================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print(f"\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =============================================
# STEP 7: Feature Importance
# =============================================
feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
importances = model.feature_importances_
print("\n📊 Feature Importance:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"   {name}: {imp:.4f}")

# =============================================
# STEP 8: Save Model & Scaler
# =============================================
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ model.pkl saved!")
print("✅ scaler.pkl saved!")

# =============================================
# STEP 9: Test Prediction
# =============================================
print("\n🌾 Test Prediction:")
sample = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])  # Rice ke liye
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled).max() * 100

print(f"   Input: N=90, P=42, K=43, Temp=20.87, Humidity=82, pH=6.5, Rainfall=202.93")
print(f"   Predicted Crop: {prediction[0].upper()}")
print(f"   Confidence: {probability:.2f}%")

print("\n🎉 ML Model ready hai! Ab Flask backend banana shuru karo.")