import pickle
import numpy as np
import pandas as pd
from collections import Counter

# Load model + symptom list
with open("models/fitted_model_new.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/list_of_symptoms_new.pickle", "rb") as f:
    symptoms = pickle.load(f)

# Load unique dataset
df = pd.read_csv("data/dataset.csv")
df = df.drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------------
# Query
# ------------------------------------------------------------

user_symptoms = [
    "chest_pain",
    "high_fever",
    "cough"
]

X = np.array([
    [
        1 if symptom in user_symptoms else 0
        for symptom in symptoms
    ]
], dtype=np.int8)


# ------------------------------------------------------------
# Get nearest 20 examples
# ------------------------------------------------------------

distances, indices = model.kneighbors(
    X,
    n_neighbors=20
)


print("=" * 70)
print("USER SYMPTOMS")
print("=" * 70)
print(user_symptoms)


# ------------------------------------------------------------
# Show nearest examples
# ------------------------------------------------------------

print()
print("=" * 70)
print("20 NEAREST DATASET EXAMPLES")
print("=" * 70)

for rank, (distance, index) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):

    row = df.iloc[index]

    row_symptoms = [
        str(value).strip()
        for value in row.iloc[1:]
        if pd.notna(value)
    ]

    print(
        f"{rank:2d}. "
        f"distance={distance:.4f} | "
        f"{row['Disease']}"
    )

    print(
        "    ",
        ", ".join(row_symptoms)
    )


# ------------------------------------------------------------
# Disease vote among nearest neighbors
# ------------------------------------------------------------

print()
print("=" * 70)
print("DISEASE VOTES")
print("=" * 70)

nearest_diseases = [
    df.iloc[index]["Disease"]
    for index in indices[0]
]

votes = Counter(nearest_diseases)

for disease, count in votes.most_common():

    print(
        f"{disease:<40} {count} neighbors"
    )


# ------------------------------------------------------------
# Top 3 diseases
# ------------------------------------------------------------

top3 = [
    disease
    for disease, count in votes.most_common(3)
]

print()
print("=" * 70)
print("TOP 3 CANDIDATES")
print("=" * 70)

for rank, disease in enumerate(top3, start=1):

    print(
        f"{rank}. {disease} "
        f"({votes[disease]} / 20 neighbors)"
    )


# ------------------------------------------------------------
# Compare symptoms of top candidates
# ------------------------------------------------------------

print()
print("=" * 70)
print("SYMPTOMS OF TOP CANDIDATES")
print("=" * 70)

for disease in top3:

    disease_rows = df[
        df["Disease"] == disease
    ]

    disease_symptoms = set()

    for _, row in disease_rows.iterrows():

        for value in row.iloc[1:]:

            if pd.notna(value):

                disease_symptoms.add(
                    str(value).strip()
                )

    print()
    print(disease)

    print(
        "Symptoms:",
        ", ".join(sorted(disease_symptoms))
    )

    # Symptoms not yet provided by user
    missing = sorted(
        disease_symptoms -
        set(user_symptoms)
    )

    print(
        "Not provided:",
        ", ".join(missing)
    )
