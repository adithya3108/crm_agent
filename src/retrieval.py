import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def build_index(threads_path="data/xbox_threads.jsonl", out_path="data/retrieval_index.pkl"):
    threads = []
    with open(threads_path, encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))

    texts = [t["customer_text"] for t in threads]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "threads": threads}, f)

    print(f"Indexed {len(threads)} threads -> {out_path}")


class Retriever:
    def __init__(self, index_path="data/retrieval_index.pkl"):
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.threads = data["threads"]

    def query(self, text, k=3):
        vec = self.vectorizer.transform([text])
        sims = (self.matrix @ vec.T).toarray().ravel()
        top_idx = sims.argsort()[::-1][:k]
        return [(self.threads[i], float(sims[i])) for i in top_idx if sims[i] > 0]


if __name__ == "__main__":
    build_index()
