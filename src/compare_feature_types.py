
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer, f1_score


# -----------------------------------
# 1. File paths and labels
# -----------------------------------

DATA_PATH = Path("data/telugu_news_labeled.csv")
OUTPUT_PATH = Path("reports/feature_type_comparison.csv")

LABELS = [
    "environment",
    "culture",
    "education",
    "agriculture",
    "water"
]


# -----------------------------------
# 2. Load dataset
# -----------------------------------

df = pd.read_csv(DATA_PATH)

X = df["headline"].fillna("")
y = df[LABELS]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Total articles:", len(df))
print("Training articles:", len(X_train))
print("Reserved test articles:", len(X_test))


# -----------------------------------
# 3. Define feature configurations
# -----------------------------------

feature_configurations = {

    "character_tfidf": TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 3),
        sublinear_tf=True
    ),

    "word_tfidf": TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True
    ),

    "combined_tfidf": FeatureUnion([
        (
            "character_features",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 3),
                sublinear_tf=True
            )
        ),
        (
            "word_features",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        )
    ])
}


# -----------------------------------
# 4. Configure cross-validation
# -----------------------------------

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


# -----------------------------------
# 5. Train and evaluate models
# -----------------------------------

results = []

for name, vectorizer in feature_configurations.items():

    print(f"\nEvaluating: {name}")

    model = Pipeline([
        ("features", vectorizer),

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
        "model": name,
        "mean_micro_f1": scores["test_micro_f1"].mean(),
        "std_micro_f1": scores["test_micro_f1"].std(),
        "mean_macro_f1": scores["test_macro_f1"].mean()
    })


# -----------------------------------
# 6. Save comparison results
# -----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="mean_micro_f1",
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

print("\nFEATURE TYPE COMPARISON\n")
print(results_df.to_string(index=False))


# -----------------------------------
# 7. Identify best configuration
# -----------------------------------

best_model = results_df.iloc[0]["model"]

print("\nBest feature configuration:", best_model)

print(
    "\nNote: Best means highest observed "
    "cross-validation Micro F1, not proven superiority."
)

print("\nTest set remains unused for model selection.")
print("Results saved to:", OUTPUT_PATH)
