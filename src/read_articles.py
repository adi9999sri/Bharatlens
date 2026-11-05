import json
file_path = "data/articles.json"
with open(file_path, "r", encoding="utf-8") as file:
    articles = json.load(file)
for article in articles:
    print("ID:", article["article_id"])
    print("Headline:", article["headline"])
    print("Publisher:", article["publisher"])
    print("--------------------")


