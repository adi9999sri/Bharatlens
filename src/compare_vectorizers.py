
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import clone


# File locations
DATA_PATH = Path("data/telugu_news_labeled.csv")
OUTPUT_PATH = Path("reports/vectorizer_comparison.csv")

LABELS = [
    "environment",
    "culture",
    "education",
    "agriculture",
    "water"
]

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df["headline"]
y = df[LABELS]

# Same initial split as our previous experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Configurations to compare
configurations = {
    "char_2_3": (2, 3),
    "char_2_5": (2, 5),
    "char_3_6": (3, 6)
}

# Cross-validation using training data only
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "micro_f1": make_scorer(
        f1_score,
        average="micro",
        zero_division=0
    ),
    "macro_f1": make_scorer(
        f1_score,
        average="macro",
        zero_division=0
    )
}

results = []

for name, ngram_range in configurations.items():

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=ngram_range,
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

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        error_score="raise"
    )

    results.append({
        "configuration": name,
        "ngram_range": str(ngram_range),
        "micro_f1_mean": scores["test_micro_f1"].mean(),
        "micro_f1_std": scores["test_micro_f1"].std(),
        "macro_f1_mean": scores["test_macro_f1"].mean()
    })

# Save comparison
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "micro_f1_mean",
    ascending=False
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nVECTORIZER COMPARISON")
print(results_df.to_string(index=False))

# Identify strongest configuration
best_name = results_df.iloc[0]["configuration"]
best_range = configurations[best_name]

print("\nBest configuration:", best_name)
print("Best n-gram range:", best_range)
print("\nTest set remains unused for model selection.")