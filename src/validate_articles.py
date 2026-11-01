
import json

# Required fields for each news article
required_fields = ["article_id", "headline", "publisher", "language"]


def validate_article(article):
    errors = []

    # Check that the article is a dictionary
    if not isinstance(article, dict):
        return ["Article must be a JSON object"]

    # Check missing, empty, and invalid fields
    for field in required_fields:

        if field not in article:
            errors.append(f"Missing field: {field}")
            continue

        value = article[field]

        if not isinstance(value, str):
            errors.append(f"Invalid data type for {field}")
        elif not value.strip():
            errors.append(f"Empty field: {field}")

    # Check language
    if (
        isinstance(article.get("language"), str)
        and article["language"].strip()
        and article["language"] != "te"
    ):
        errors.append("Invalid language: expected 'te'")

    return errors


def validate_dataset(articles):
    seen_ids = set()
    valid_count = 0
    invalid_count = 0

    for index, article in enumerate(articles, start=1):

        errors = validate_article(article)

        # Check for duplicate IDs
        if isinstance(article, dict):
            article_id = article.get("article_id")

            if isinstance(article_id, str) and article_id.strip():
                if article_id in seen_ids:
                    errors.append("Duplicate article ID")
                else:
                    seen_ids.add(article_id)

        if errors:
            invalid_count += 1
            print(f"\nArticle {index}: INVALID")

            for error in errors:
                print(f"- {error}")

        else:
            valid_count += 1
            print(f"\nArticle {index}: VALID")

    print("\nValidation Complete")
    print(f"Valid Articles: {valid_count}")
    print(f"Invalid Articles: {invalid_count}")


# Load JSON dataset
try:
    with open("data/articles.json", "r", encoding="utf-8") as file:
        articles = json.load(file)

    if not isinstance(articles, list):
        print("Error: JSON must contain a list of articles")
    else:
        validate_dataset(articles)

except FileNotFoundError:
    print("Error: articles.json file not found")

except json.JSONDecodeError:
    print("Error: Invalid JSON format")