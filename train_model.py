import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report


DATASET = "data/dataset.csv"
OUTPUT_MODEL = "models/fitted_model_new.pkl"
OUTPUT_FEATURES = "data/list_of_symptoms_new.pickle"


print("Loading dataset...")
raw = pd.read_csv(DATASET)

# ---------------------------------------------------------
# 1. Reproduce original symptom extraction
# ---------------------------------------------------------

symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

symptoms = np.concatenate(
    [raw[col].dropna().unique() for col in symptom_columns]
)

# Strip whitespace exactly before creating feature names
symptoms = [str(s).strip() for s in symptoms]

# Remove empty/invalid values
symptoms_unique = sorted(set(s for s in symptoms if s))

print(f"Unique symptoms found: {len(symptoms_unique)}")

# ---------------------------------------------------------
# 2. Convert symptom lists → binary feature matrix
# ---------------------------------------------------------

rows = []

for _, row in raw.iterrows():
    present = {
        str(row[col]).strip()
        for col in symptom_columns
        if pd.notna(row[col]) and str(row[col]).strip()
    }

    rows.append([1 if symptom in present else 0
                 for symptom in symptoms_unique])

X = np.asarray(rows, dtype=np.int8)
y = raw["Disease"].astype(str).str.strip().to_numpy()

print("Original rows:", len(X))
print("Original features:", X.shape[1])

# ---------------------------------------------------------
# 3. Remove exact duplicate disease + symptom combinations
# ---------------------------------------------------------

clean_df = pd.DataFrame(X, columns=symptoms_unique)
clean_df["Disease"] = y

before = len(clean_df)
clean_df = clean_df.drop_duplicates()
after = len(clean_df)

print(f"Duplicate rows removed: {before - after}")
print(f"Unique examples: {after}")

X = clean_df.drop(columns=["Disease"]).to_numpy(dtype=np.int8)
y = clean_df["Disease"].to_numpy()

print("Final X shape:", X.shape)
print("Number of diseases:", len(np.unique(y)))

# ---------------------------------------------------------
# 4. Leakage-safe stratified split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# ---------------------------------------------------------
# 5. Original model architecture
# ---------------------------------------------------------

level0 = [
    (
        "lr",
        LogisticRegression(
            solver="liblinear",
            C=0.03,
            max_iter=2000
        )
    ),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=6,
            metric="cosine"
        )
    ),
    (
        "dctree",
        DecisionTreeClassifier(
            splitter="random",
            max_depth=34,
            random_state=42
        )
    ),
    (
        "svm",
        SVC(C=0.1)
    )
]

level1 = LogisticRegression(max_iter=2000)

model = StackingClassifier(
    estimators=level0,
    final_estimator=level1,
    cv=5
)

print("\nTraining stacking ensemble...")
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Honest hold-out evaluation
# ---------------------------------------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\n==============================")
print("HOLD-OUT TEST RESULT")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)

# ---------------------------------------------------------
# 7. Retrain final model on all UNIQUE examples
# ---------------------------------------------------------

print("\nTraining final model on all unique examples...")

final_model = StackingClassifier(
    estimators=level0,
    final_estimator=level1,
    cv=5
)

final_model.fit(X, y)

# ---------------------------------------------------------
# 8. Save model + exact feature order
# ---------------------------------------------------------

with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump(final_model, f)

with open(OUTPUT_FEATURES, "wb") as f:
    pickle.dump(symptoms_unique, f)

print("\n==============================")
print("DONE")
print("==============================")
print("Model saved:", OUTPUT_MODEL)
print("Features saved:", OUTPUT_FEATURES)
print("Feature count:", len(symptoms_unique))
