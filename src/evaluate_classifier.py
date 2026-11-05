
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

DATA_PATH = Path("data/telugu_news_labeled.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

LABELS = [
    "environment",
    "culture",
    "education",
    "agriculture",
    "water"
]

# Load the same dataset and split as your original experiment
df = pd.read_csv(DATA_PATH)

X = df["headline"]
y = df[LABELS]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    )
])

model.fit(X_train, y_train)

# Obtain prediction probabilities
probabilities = model.predict_proba(X_test)

# Compare thresholds
results = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results.append({
        "threshold": threshold,
        "micro_precision": precision_score(
            y_test, predictions,
            average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_test, predictions,
            average="micro", zero_division=0
        ),
        "micro_f1": f1_score(
            y_test, predictions,
            average="micro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, predictions,
            average="macro", zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False
)

# Generate detailed error report at baseline threshold
baseline_predictions = (
    probabilities >= 0.50
).astype(int)

error_records = []

for position, original_index in enumerate(X_test.index):

    actual = y_test.iloc[position].to_numpy()
    predicted = baseline_predictions[position]

    if not (actual == predicted).all():

        error_records.append({
            "headline": X_test.loc[original_index],
            "actual_labels": ", ".join(
                label for label, value
                in zip(LABELS, actual) if value == 1
            ) or "Other",
            "predicted_labels": ", ".join(
                label for label, value
                in zip(LABELS, predicted) if value == 1
            ) or "Other",
            "probabilities": ", ".join(
                f"{label}={score:.3f}"
                for label, score
                in zip(LABELS, probabilities[position])
            )
        })

errors_df = pd.DataFrame(
    error_records,
    columns=[
        "headline",
        "actual_labels",
        "predicted_labels",
        "probabilities"
    ]
)

errors_df.to_csv(
    OUTPUT_DIR / "classification_errors.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nMisclassified headlines:", len(errors_df))
print("\nReports saved in:", OUTPUT_DIR)