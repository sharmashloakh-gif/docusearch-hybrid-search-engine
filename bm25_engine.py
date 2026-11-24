import math
import re
from collections import Counter, defaultdict

class BM25SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = {}
        self.doc_len = {}
        self.inverted_index = defaultdict(dict)
        self.avgdl = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def index_document(self, doc_id: str, text: str):
        tokens = self._tokenize(text)
        self.docs[doc_id] = text
        self.doc_len[doc_id] = len(tokens)
        freqs = Counter(tokens)
        for term, count in freqs.items():
            self.inverted_index[term][doc_id] = count
        self.avgdl = sum(self.doc_len.values()) / max(1, len(self.doc_len))

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores = defaultdict(float)
        N = len(self.docs)
        for term in self._tokenize(query):
            postings = self.inverted_index.get(term, {})
            df = len(postings)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in postings.items():
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * ((tf * (self.k1 + 1)) / denom)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

if __name__ == "__main__":
    engine = BM25SearchEngine()
    engine.index_document("doc1", "Django ORM optimization and MySQL composite indexing")
    engine.index_document("doc2", "Python asynchronous programming and distributed systems")
    engine.index_document("doc3", "MySQL query tuning and database sharding patterns")
    print(engine.search("mysql indexing"))
