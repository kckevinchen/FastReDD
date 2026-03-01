from typing import List

from .vectorizer import DocVectorizer
from .clusterer import ClustererGPT


class DocumentClustering:
    def __init__(self, documents: List[str], n_clusters: int, vectorizer: DocVectorizer, clusterer: ClustererGPT):
        self.documents = documents
        self.vectorizer = vectorizer if vectorizer else DocVectorizer()
        self.clusterer = clusterer if clusterer else ClustererGPT(n_clusters)
        self.vectors = []
        self.cluster_ids = []

    def cluster(self):
        self.vectors = self.vectorizer.fit_transform(self.documents)
        self.cluster_ids = self.clusterer.fit_predict(self.vectors)
        return self.cluster_ids

    def get_clustered_documents(self):
        clustered_documents = {}
        for cluster_id, document in zip(self.cluster_ids, self.documents):
            if cluster_id not in clustered_documents:
                clustered_documents[cluster_id] = []
            clustered_documents[cluster_id].append(document)
        return clustered_documents

