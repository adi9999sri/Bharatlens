
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score


# -----------------------------------
# 1. Define file paths
# -----------------------------------

DATA_PATH = Path("data/telugu_news_labeled.csv")
MODEL_PATH = Path("models/telugu_news_classifier.joblib")

LABELS = [
    "environment",
    "culture",
    "education",
    "agriculture",
    "water"
]


# -----------------------------------
# 2. Load and validate dataset
# -----------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = ["headline"] + LABELS

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

if df["headline"].isna().any():
    raise ValueError("Dataset contains missing headlines")

if not df["headline"].map(
    lambda x: isinstance(x, str) and bool(x.strip())
).all():
    raise ValueError("Dataset contains invalid headlines")

for label in LABELS:
    if not df[label].isin([0, 1]).all():
        raise ValueError(
            f"Invalid label values in: {label}"
        )

if df["headline"].duplicated().any():
    raise ValueError("Dataset contains duplicate headlines")

X = df["headline"]
y = df[LABELS]

print("Dataset loaded successfully")
print("Total articles:", len(df))
print("\nCategory distribution:")
print(y.sum())


# -----------------------------------
# 3. Split train and test data
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))


# -----------------------------------
# 4. Build the NLP pipeline
# -----------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
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


# -----------------------------------
# 5. Train the model
# -----------------------------------

print("\nTraining Telugu NLP classifier...")

model.fit(X_train, y_train)

print("Training completed!")


# -----------------------------------
# 6. Evaluate model
# -----------------------------------

predictions = model.predict(X_test)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=LABELS,
        zero_division=0
    )
)

print(
    "Micro F1:",
    round(
        f1_score(
            y_test,
            predictions,
            average="micro",
            zero_division=0
        ),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        4
    )
)


# -----------------------------------
# 7. Test on a new Telugu headline
# -----------------------------------

sample_headline = "రైతులకు సాగునీటి సౌకర్యాల మెరుగుదల"

predicted_labels = model.predict(
    [sample_headline]
)[0]

probabilities = model.predict_proba(
    [sample_headline]
)[0]

print("\nSample Telugu headline:", sample_headline)

print("\nPrediction probabilities:")

for label, probability in zip(LABELS, probabilities):
    print(f"{label}: {probability:.3f}")

print("\nPredicted categories:")

selected = [
    label
    for label, prediction in zip(
        LABELS,
        predicted_labels
    )
    if prediction == 1
]

print(selected if selected else ["Other"])


# -----------------------------------
# 8. Save trained model
# -----------------------------------

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
