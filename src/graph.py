import re
import pickle
import pandas as pd
import networkx as nx
import nltk
from itertools import product, combinations
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class GraphBuilder:
    """
    Handles loading raw data, processing it, and building the necessary graphs.
    The graphs are saved to disk to avoid rebuilding them on every run.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self._download_nltk_data()

    def _download_nltk_data(self):
        try:
            stopwords.words("english")
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("wordnet", quiet=True)
            print("Downloads complete.")

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

    def _extract_customer_text(self, conversation):
        if not isinstance(conversation, str):
            return ""
        lines = conversation.split("\n")
        customer_dialogue = [
            line.strip()[2:].strip()
            for line in lines
            if line.strip().upper().startswith("C:")
        ]
        return " ".join(customer_dialogue)

    def _parse_comma_separated_string(self, s):
        if not isinstance(s, str):
            return []
        return [emotion.strip() for emotion in s.split(",")]

    def load_and_clean_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Successfully loaded '{self.data_path}'.")
        except FileNotFoundError:
            print(f"Error: '{self.data_path}' not found. Cannot build graphs.")
            return False
        return True

    def build_holistic_concept_graph(self):
        G = nx.Graph()
        df_g = self.df.copy()  # type: ignore[]
        pref_columns = [col for col in df_g.columns if col.startswith("pref_")]
        df_g[pref_columns] = df_g[pref_columns].fillna("not_specified")
        df_g["product_interest"].fillna("not_specified", inplace=True)
        df_g["emotions_list"] = df_g["emotions"].apply(
            self._parse_comma_separated_string
        )
        df_g["question_types_list"] = df_g["question_types"].apply(
            self._parse_comma_separated_string
        )

        for _, row in df_g.iterrows():
            concepts = []
            if row["product_interest"] != "not_specified":
                concepts.append(row["product_interest"])
            for col in pref_columns:
                if row[col] != "not_specified":
                    concepts.append(f"pref:{row[col]}")
            concepts.extend(row["emotions_list"])
            concepts.extend(row["question_types_list"])
            for edge in combinations(set(concepts), 2):
                if G.has_edge(*edge):
                    G[edge[0]][edge[1]]["weight"] += 1
                else:
                    G.add_edge(edge[0], edge[1], weight=1)

        print(f"Holistic Concept Graph (G) created with {G.number_of_nodes()} nodes.")
        return G

    def build_emotion_word_graph(self):
        B = nx.Graph()
        df_b = self.df[self.df["emotions"].notna()].copy()  # type: ignore[]
        df_b["customer_text"] = df_b["conversation_text"].apply(  # type: ignore[]
            self._extract_customer_text
        )
        df_b["cleaned_tokens"] = df_b["customer_text"].apply(self._preprocess_text)  # type: ignore[]
        df_b["emotions_list"] = df_b["emotions"].apply(  # type: ignore[]
            self._parse_comma_separated_string
        )

        words = set(token for tokens in df_b["cleaned_tokens"] for token in tokens)
        emotions = set(
            emotion for emotions in df_b["emotions_list"] for emotion in emotions
        )
        B.add_nodes_from(words, bipartite=0)
        B.add_nodes_from(emotions, bipartite=1)

        for _, row in df_b.iterrows():
            for word, emotion in product(row["cleaned_tokens"], row["emotions_list"]):
                if B.has_edge(word, emotion):
                    B[word][emotion]["weight"] += 1
                else:
                    B.add_edge(word, emotion, weight=1)

        print(f"Emotion-Word Graph (B) created with {B.number_of_nodes()} nodes.")
        return B

    def build_cross_sell_graph(self):
        """
        Builds a directed graph from product interest to cross-sell opportunities.
        """
        X = nx.DiGraph()
        df_x = self.df[self.df["cross_sell_opportunity"].notna()].copy()  # type: ignore[]

        df_x["product_interest_cleaned"] = df_x["product_interest"].str.strip()  # type: ignore[]

        for _, row in df_x.iterrows():
            product = row["product_interest_cleaned"]
            cross_sell_opp = row["cross_sell_opportunity"]

            if not X.has_node(product):
                X.add_node(product, type="product")
            if not X.has_node(cross_sell_opp):
                X.add_node(cross_sell_opp, type="cross_sell")

            if X.has_edge(product, cross_sell_opp):
                X[product][cross_sell_opp]["weight"] += 1
            else:
                X.add_edge(product, cross_sell_opp, weight=1)

        print(f"Cross-Sell Graph (X) created with {X.number_of_nodes()} nodes.")

        # graph enrichment
        X_undirected = X.to_undirected()

        product_nodes = {n for n, d in X.nodes(data=True) if d.get("type") == "product"}

        node_pairs_to_predict = combinations(product_nodes, 2)

        ebunch = [
            pair
            for pair in node_pairs_to_predict
            if not X.has_edge(pair[0], pair[1]) and not X.has_edge(pair[1], pair[0])
        ]

        predictions = nx.jaccard_coefficient(X_undirected, ebunch)

        sorted_predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

        num_links_to_add = 5
        new_edges_added = 0

        for u, v, score in sorted_predictions:
            if new_edges_added >= num_links_to_add or score == 0:
                break

            if X.has_node(u):
                potential_targets = set()
                for neighbor in nx.common_neighbors(X_undirected, u, v):
                    if X.has_node(neighbor):
                        potential_targets.update(X.successors(neighbor))

                if potential_targets:
                    target_opp = max(
                        potential_targets,
                        key=lambda t: sum(
                            X[neighbor][t]["weight"]
                            for neighbor in nx.common_neighbors(X_undirected, u, v)
                            if X.has_edge(neighbor, t)
                        ),
                    )

                    if not X.has_edge(v, target_opp):
                        X.add_edge(v, target_opp, weight=score, type="predicted")
                        new_edges_added += 1

        print(
            f"Graph Enrichment Complete: Added {new_edges_added} new predicted cross-sell edges."
        )
        return X

    def build_and_save_graphs(self, g_path, b_path, x_path, mapping_path):
        """
        Orchestrates building and saving all three graphs.
        """
        if not self.load_and_clean_data():
            return

        print(f"Saved Node Mapping to {mapping_path}")
        G = self.build_holistic_concept_graph()
        with open(g_path, "wb") as f:
            pickle.dump(G, f)
        print(f"Saved Concept Graph to {g_path}")

        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        with open(mapping_path, "wb") as f:
            pickle.dump(node_mapping, f)
        print(f"Saved Node Mapping to {mapping_path}")

        B = self.build_emotion_word_graph()
        with open(b_path, "wb") as f:
            pickle.dump(B, f)
        print(f"Saved Emotion-Word Graph to {b_path}")

        X = self.build_cross_sell_graph()
        with open(x_path, "wb") as f:
            pickle.dump(X, f)
        print(f"Saved Cross-Sell Graph to {x_path}")
