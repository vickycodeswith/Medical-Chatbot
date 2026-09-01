import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/dataset.csv")

# Remove duplicates exactly like original notebook
df = df.drop_duplicates().reset_index(drop=True)


# Get unique symptoms
symptoms = np.unique(
    np.concatenate([
        df[f"Symptom_{i}"].dropna().unique()
        for i in range(1, 18)
    ])
)

symptoms = sorted(
    [str(s).replace(" ", "") for s in symptoms]
)

symptom_to_index = {
    symptom: index
    for index, symptom in enumerate(symptoms)
}


# Create binary feature matrix
X = np.zeros(
    (len(df), len(symptoms)),
    dtype=np.int8
)


for row_index, row in df.iterrows():

    for column in range(1, 18):

        value = row.iloc[column]

        if pd.notna(value):

            symptom = str(value).replace(
                " ",
                ""
            )

            if symptom in symptom_to_index:

                X[
                    row_index,
                    symptom_to_index[symptom]
                ] = 1


y = df["Disease"].values


# Same train/test split for every k
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Dataset:", X.shape)
print("Train:", X_train.shape)
print("Test:", X_test.shape)

print()
print("=" * 50)
print("KNN K-VALUE COMPARISON")
print("=" * 50)


results = []


for k in range(1, 21):

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine"
    )

    model.fit(
        X_train,
        y_train
    )

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    test_accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    results.append(
        (k, train_accuracy, test_accuracy)
    )

    print(
        f"k={k:2d} | "
        f"Train={train_accuracy * 100:6.2f}% | "
        f"Test={test_accuracy * 100:6.2f}%"
    )


# Best test accuracy
best = max(
    results,
    key=lambda x: x[2]
)

print()
print("=" * 50)
print("BEST RESULT")
print("=" * 50)

print(
    "Best k:",
    best[0]
)

print(
    "Training accuracy:",
    round(best[1] * 100, 2),
    "%"
)

print(
    "Test accuracy:",
    round(best[2] * 100, 2),
    "%"
)
