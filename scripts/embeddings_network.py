from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import load_dataset


def tokenize(value: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", str(value).lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", default="gobierno")
    parser.add_argument("--network-demo", action="store_true")
    args = parser.parse_args()
    df = load_dataset()
    sentences = [tokenize(value) for value in df[settings.text_column].dropna()]
    model = Word2Vec(sentences, vector_size=60, window=5, min_count=1, workers=1, seed=42, epochs=30)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    model.save(str(ROOT / "models/word2vec.model"))
    tokens = [token for sentence in sentences for token in sentence]
    coverage = sum(token in model.wv for token in tokens) / max(len(tokens), 1)
    print("Vocabulario:", len(model.wv), "Cobertura:", round(coverage, 3))
    if args.word in model.wv:
        print("Vecinos:", model.wv.most_similar(args.word, topn=min(10, len(model.wv) - 1)))
    else:
        print(f"La palabra '{args.word}' no está en el vocabulario")
    if not args.network_demo:
        return
    graph = nx.karate_club_graph()
    metrics = pd.DataFrame({
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph),
        "pagerank": nx.pagerank(graph),
    })
    metrics.index.name = "node"
    metrics.to_csv(ROOT / "reports/centralities.csv")
    communities = list(nx.community.greedy_modularity_communities(graph))
    community = {node: index for index, group in enumerate(communities) for node in group}
    print("Nodos:", graph.number_of_nodes(), "Aristas:", graph.number_of_edges())
    print("Comunidades:", len(communities), "Modularidad:", round(nx.community.modularity(graph, communities), 3))
    position = nx.spring_layout(graph, seed=42)
    nx.draw(graph, position, node_color=[community[node] for node in graph], cmap="tab10", with_labels=True, font_size=7)
    plt.title("Comunidades estructurales · red de demostración")
    plt.savefig(ROOT / "reports/network.png", dpi=180, bbox_inches="tight")
    nx.write_graphml(graph, ROOT / "reports/social_network.graphml")


if __name__ == "__main__":
    main()
