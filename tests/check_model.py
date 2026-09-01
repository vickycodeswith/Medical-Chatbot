import pickle
import pandas as pd
import numpy as np

# ============================================================
# LOAD MODEL
# ============================================================

with open("models/fitted_model_new.pkl", "rb") as f:
    model = pickle.load(f)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/dataset.csv")

# IMPORTANT:
# Remove duplicate rows AND reset pandas index
df = df.drop_duplicates().reset_index(drop=True)


# ============================================================
# GET ALL UNIQUE SYMPTOMS
# ============================================================

symptoms = np.unique(
    np.concatenate([
        df[f"Symptom_{i}"].dropna().unique()
        for i in range(1, 18)
    ])
)

# Same cleaning as original notebook
symptoms = sorted(
    [
        str(s).replace(" ", "")
        for s in symptoms
    ]
)


print("Unique rows:", len(df))
print("Unique symptoms:", len(symptoms))


# ============================================================
# CREATE BINARY FEATURE MATRIX
# ============================================================

X = np.zeros(
    (len(df), len(symptoms)),
    dtype=np.int8
)

symptom_to_index = {
    symptom: index
    for index, symptom in enumerate(symptoms)
}


for row_index, row in df.iterrows():

    row_symptoms = set()

    for column in range(1, 18):

        value = row.iloc[column]

        if pd.notna(value):

            value = str(value).replace(
                " ",
                ""
            )

            row_symptoms.add(value)

    for symptom in row_symptoms:

        if symptom in symptom_to_index:

            feature_index = symptom_to_index[
                symptom
            ]

            X[
                row_index,
                feature_index
            ] = 1


# ============================================================
# LABELS
# ============================================================

y = df["Disease"].values


# ============================================================
# BASIC CHECKS
# ============================================================

print()
print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "Expected X shape: (304, 131)"
)

print(
    "Expected y shape: (304,)"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print()
print(
    "Model:",
    type(model).__name__
)

print(
    "Model features:",
    model.n_features_in_
)

print(
    "Model classes:",
    len(model.classes_)
)


# ============================================================
# TRAINING ACCURACY
# ============================================================

accuracy = model.score(
    X,
    y
)

print()
print(
    "Training accuracy:",
    accuracy
)

print(
    "Training accuracy (%):",
    round(
        accuracy * 100,
        2
    )
)


# ============================================================
# FEATURE COUNT CHECK
# ============================================================

print()
print(
    "Feature count matches:",
    model.n_features_in_ == X.shape[1]
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

predictions = model.predict(X)

print()
print(
    "First 10 actual → predicted:"
)

for actual, predicted in zip(
    y[:10],
    predictions[:10]
):

    print(
        actual,
        "→",
        predicted
    )
