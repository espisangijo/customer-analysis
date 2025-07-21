import re
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class RecommenderSystem:
    """
    Loads pre-built graphs and provides recommendation and analysis functions.
    """

    def __init__(self, concept_graph, emotion_graph, cross_sell_graph):
        self.G = concept_graph
        self.B = emotion_graph
        self.X = cross_sell_graph
        self.all_graph_nodes = list(self.G.nodes())
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return []
        text = re.sub(r"[^a-z\s]", "", text.lower())
        tokens = word_tokenize(text)
        return [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]

    def get_recommendations(self, current_concepts, top_n=3):
        scores = defaultdict(float)
        for concept in current_concepts:
            if concept in self.G:
                for neighbor, attr in self.G[concept].items():
                    scores[neighbor] += attr.get("weight", 1.0)
        for concept in current_concepts:
            if concept in scores:
                del scores[concept]
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

    def detect_emotion_from_text(self, text):
        tokens = self._preprocess_text(text)
        emotion_scores = defaultdict(int)
        for token in tokens:
            if token in self.B:
                strongest_emotion, max_weight = None, 0
                for neighbor in self.B.neighbors(token):
                    if self.B.nodes[neighbor]["bipartite"] == 1:
                        weight = self.B[token][neighbor].get("weight", 1.0)
                        if weight > max_weight:
                            max_weight, strongest_emotion = weight, neighbor
                if strongest_emotion:
                    emotion_scores[strongest_emotion] += max_weight
        return (
            max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"  # type: ignore
        )

    def extract_concepts_from_text(self, text):
        found_concepts = []
        text_lower = text.lower()
        for node in sorted(self.all_graph_nodes, key=len, reverse=True):
            node_text = node.replace("pref:", "").lower()
            if node_text in text_lower:
                found_concepts.append(node)
                text_lower = text_lower.replace(node_text, "")
        return found_concepts

    def get_cross_sell_recommendation(self, current_products):
        """
        Finds the best cross-sell opportunity for the given products.
        """
        if not current_products:
            return None

        best_cross_sell = None
        max_weight = 0

        for product in current_products:
            if self.X.has_node(product):
                for _, cross_sell_opp, data in self.X.out_edges(product, data=True):
                    weight = data.get("weight", 0)
                    if weight > max_weight:
                        max_weight = weight
                        best_cross_sell = cross_sell_opp

        return best_cross_sell
