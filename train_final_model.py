import pickle
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/dataset.csv")

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print("Unique rows:", len(df))


# ============================================================
# BUILD SYMPTOM LIST
# ============================================================

symptoms = np.unique(
    np.concatenate([
        df[f"Symptom_{i}"].dropna().unique()
        for i in range(1, 18)
    ])
)

# Same preprocessing used by original notebook
symptoms = sorted(
    [str(s).replace(" ", "") for s in symptoms]
)

print("Unique symptoms:", len(symptoms))


# ============================================================
# SAVE SYMPTOM LIST
# ============================================================

with open(
    "data/list_of_symptoms_new.pickle",
    "wb"
) as f:

    pickle.dump(
        symptoms,
        f
    )


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

            symptom = str(value).replace(
                " ",
                ""
            )

            row_symptoms.add(symptom)

    for symptom in row_symptoms:

        if symptom in symptom_to_index:

            X[
                row_index,
                symptom_to_index[symptom]
            ] = 1


# ============================================================
# LABELS
# ============================================================

y = df["Disease"].values


print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# FINAL KNN MODEL
# ============================================================

model = KNeighborsClassifier(
    n_neighbors=6,
    metric="cosine"
)


print()
print("Training KNN...")


model.fit(
    X,
    y
)


# ============================================================
# TRAINING ACCURACY
# ============================================================

accuracy = model.score(
    X,
    y
)


print(
    "Training accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    "models/fitted_model_new.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


print()
print("Model saved:")
print("models/fitted_model_new.pkl")

print()
print("Symptom list saved:")
print("data/list_of_symptoms_new.pickle")

print()
print("Model:")
print(model)
