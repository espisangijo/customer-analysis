import re
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import networkx as nx
import torch
import torch.nn.functional as F
from transformers import pipeline

from src.train_gat import GATLinkPredictor


class RecommenderSystem:
    """
    Loads pre-built graphs and provides recommendation and analysis functions.
    """

    def __init__(
        self,
        concept_graph,
        emotion_graph,
        cross_sell_graph,
        gat_model_path=None,
        node_mapping=None,
    ):
        self.G = concept_graph
        self.B = emotion_graph
        self.X = cross_sell_graph
        self.all_graph_nodes = list(self.G.nodes())
        self.lemmatizer = WordNetLemmatizer()
        self.approach = "link"

        self.stop_words = set(stopwords.words("english"))

        # link prediction approach
        self._emotion_nodes = {
            n for n, d in self.B.nodes(data=True) if d.get("bipartite") == 1
        }
        self._product_nodes = {
            n
            for n in self.G.nodes()
            if not n.startswith("pref:")
            and n not in self._emotion_nodes
            and "question" not in n
        }
        self._preference_nodes = {n for n in self.G.nodes() if n.startswith("pref:")}
        self._question_nodes = (
            set(self.G.nodes())
            - self._product_nodes
            - self._emotion_nodes
            - self._preference_nodes
        )

        # GAT approach
        if node_mapping and gat_model_path:
            self.node_mapping = node_mapping
            self.gat_model = None
            self.node_embeddings = None
            try:
                self.gat_model = GATLinkPredictor(len(node_mapping), 128, 64)
                self.gat_model.load_state_dict(torch.load(gat_model_path))
                self.gat_model.eval()
                print(f"Successfully loaded trained GAT model from '{gat_model_path}'.")

                with torch.no_grad():
                    G_relabeled = nx.relabel_nodes(self.G, self.node_mapping)
                    x = F.one_hot(
                        torch.arange(len(self.node_mapping)),
                        num_classes=len(self.node_mapping),
                    ).float()
                    edge_index = torch.tensor(list(G_relabeled.edges)).t().contiguous()
                    self.node_embeddings = self.gat_model.encode(x, edge_index)
                print("Generated node embeddings using GAT model.")

                self.approach = "gat"
            except FileNotFoundError:
                print(
                    f"Warning: GAT model file not found at '{gat_model_path}'. GAT recommendations will be unavailable."
                )
            except Exception as e:
                print(f"An error occurred while loading the GAT model: {e}")

        self.emotion_pipeline = pipeline(
            "text-classification", model="j-hartmann/emotion-english-distilroberta-base"
        )

        print("Sentiment model loaded.")

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

    def get_link_prediction_recommendations(self, current_concepts, top_n=3):
        """
        Advanced recommender using link prediction (Jaccard Coefficient)
        to find non-obvious connections.
        """
        if self.approach == "gat":
            return self.get_gat_recommendations(current_concepts, top_n)

        if not current_concepts:
            return []

        affinity_scores = defaultdict(float)

        candidate_products = self._product_nodes - set(current_concepts)

        for candidate in candidate_products:
            total_score = 0
            for concept in current_concepts:
                for _, _, score in nx.jaccard_coefficient(
                    self.G, [(candidate, concept)]
                ):
                    total_score += score
            affinity_scores[candidate] = total_score

        return sorted(affinity_scores.items(), key=lambda item: item[1], reverse=True)[
            :top_n
        ]

    def get_gat_recommendations(self, current_concepts, top_n=3):
        """
        Most advanced recommender using the trained GAT model's node embeddings.
        """
        if self.node_embeddings is None or not current_concepts:
            return []

        if not self.node_mapping or not self.gat_model:
            return self.get_link_prediction_recommendations(current_concepts, top_n)

        affinity_scores = defaultdict(float)

        context_indices = [
            self.node_mapping[c] for c in current_concepts if c in self.node_mapping
        ]
        if not context_indices:
            return []

        context_embeddings = self.node_embeddings[context_indices]

        candidate_products_str = self._product_nodes - set(current_concepts)

        for candidate_str in candidate_products_str:
            if candidate_str in self.node_mapping:
                candidate_idx = self.node_mapping[candidate_str]
                candidate_embedding = self.node_embeddings[candidate_idx]

                score = (context_embeddings @ candidate_embedding).mean().item()
                affinity_scores[candidate_str] = score

        return sorted(affinity_scores.items(), key=lambda item: item[1], reverse=True)[
            :top_n
        ]

    def detect_emotion_from_text(self, text):
        if self.approach == "gat":
            return self.detect_emotion_from_text_transformer(text)

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

    def detect_emotion_from_text_transformer(self, text):
        """
        Limited set of emotion but better accuracy
        """
        if not text or not text.strip():
            return "neutral"

        try:
            result = self.emotion_pipeline(text)[0]
            label = result["label"]

            emotion_map = {
                "joy": "eager",
                "sadness": "concerned",
                "fear": "anxious",
                "anger": "frustrated",
                "surprise": "curious",
                "disgust": "frustrated",
                "neutral": "neutral",
            }
            return emotion_map.get(label, "neutral")
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return "neutral"

    def extract_concepts_from_text(self, text):
        found_concepts = []
        text_lower = text.lower()
        for node in sorted(self.all_graph_nodes, key=len, reverse=True):
            node_text = node.replace("pref:", "").lower()
            if node_text in text_lower:
                found_concepts.append(node)
                text_lower = text_lower.replace(node_text, "")
        return found_concepts

    def get_cross_sell_recommendation(self, current_concepts):
        """
        Proactively finds the best cross-sell opportunity based on the entire conversation context.
        """
        if not current_concepts:
            return None

        # 1. Identify all products that are known to lead to a cross-sell
        source_products = {
            n for n, d in self.X.nodes(data=True) if d.get("type") == "product"
        }
        if not source_products:
            return None

        # 2. Score each source product based on its relevance to the current context
        affinity_scores = defaultdict(float)
        for product in source_products:
            score = 0
            for concept in current_concepts:
                # Check for a connection in the holistic graph
                if self.G.has_edge(product, concept):
                    # Add the strength of the connection to the score
                    score += self.G[product][concept].get("weight", 1.0)
            affinity_scores[product] = score

        # 3. Find the product that is most relevant to the context
        if not any(affinity_scores.values()):
            return None

        best_primary_product = max(affinity_scores, key=affinity_scores.get)

        # 4. Find the strongest cross-sell opportunity for that best product
        best_cross_sell = None
        max_weight = 0
        if self.X.has_node(best_primary_product):
            for _, cross_sell_opp, data in self.X.out_edges(
                best_primary_product, data=True
            ):
                weight = data.get("weight", 0)
                if weight > max_weight:
                    max_weight = weight
                    best_cross_sell = cross_sell_opp

        return best_cross_sell
