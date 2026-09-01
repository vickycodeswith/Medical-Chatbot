import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/dataset.csv")

df = df.drop_duplicates().reset_index(drop=True)


# ============================================================
# BUILD 131 SYMPTOM FEATURES
# ============================================================

symptoms = np.unique(
    np.concatenate([
        df[f"Symptom_{i}"].dropna().unique()
        for i in range(1, 18)
    ])
)

symptoms = sorted(
    [
        str(s).replace(" ", "")
        for s in symptoms
    ]
)

symptom_to_index = {
    symptom: index
    for index, symptom in enumerate(symptoms)
}


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


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Dataset:", X.shape)
print("Training:", X_train.shape)
print("Testing:", X_test.shape)
print("Classes:", len(np.unique(y)))

print()
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            solver="liblinear",
            C=0.03,
            max_iter=2000
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=6,
            metric="cosine"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            splitter="random",
            max_depth=34,
            random_state=42
        ),

    "SVM":
        SVC(
            C=0.1
        )
}


# ============================================================
# TRAIN + EVALUATE INDIVIDUAL MODELS
# ============================================================

results = []


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    train_prediction = model.predict(
        X_train
    )

    test_prediction = model.predict(
        X_test
    )

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    results.append(
        (
            name,
            train_accuracy,
            test_accuracy
        )
    )

    print()
    print(name)

    print(
        "Training accuracy:",
        round(train_accuracy * 100, 2),
        "%"
    )

    print(
        "Test accuracy:",
        round(test_accuracy * 100, 2),
        "%"
    )


# ============================================================
# STACKING CLASSIFIER
# ============================================================

print()
print("Stacking Classifier")


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
        SVC(
            C=0.1
        )
    )
]


stacking_model = StackingClassifier(
    estimators=level0,
    final_estimator=LogisticRegression(
        max_iter=2000
    ),
    cv=5
)


stacking_model.fit(
    X_train,
    y_train
)


stack_train_prediction = stacking_model.predict(
    X_train
)

stack_test_prediction = stacking_model.predict(
    X_test
)


stack_train_accuracy = accuracy_score(
    y_train,
    stack_train_prediction
)

stack_test_accuracy = accuracy_score(
    y_test,
    stack_test_prediction
)


print(
    "Training accuracy:",
    round(
        stack_train_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Test accuracy:",
    round(
        stack_test_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)

for name, train_acc, test_acc in results:

    print(
        f"{name:<22}"
        f" Train: {train_acc * 100:6.2f}%"
        f" | Test: {test_acc * 100:6.2f}%"
    )


print(
    f"{'Stacking':<22}"
    f" Train: {stack_train_accuracy * 100:6.2f}%"
    f" | Test: {stack_test_accuracy * 100:6.2f}%"
)
