import networkx as nx

from scripts.embeddings_network import tokenize


def test_tokenize_normalizes_and_removes_one_character_tokens():
    assert tokenize("¡Hola A TODOS!") == ["hola", "todos"]


def test_demo_network_has_valid_pagerank():
    graph = nx.karate_club_graph()
    values = nx.pagerank(graph)
    assert graph.number_of_nodes() == 34
    assert all(0 <= value <= 1 for value in values.values())
    assert abs(sum(values.values()) - 1) < 1e-9
