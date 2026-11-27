"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, attributes in graph.edges(data=True):
        weight = attributes.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    
    LOUVAIN_SEED = 0
    groups = nx.community.louvain_communities(undirected, weight="weight", seed=LOUVAIN_SEED)
    ordered_groups = sorted((sorted(group) for group in groups), key=lambda group: group[0])
    return [Community(community_id, nodes) for community_id, nodes in enumerate(ordered_groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    nodes = community.nodes
    subgraph = graph.subgraph(nodes)

    entities = [
        f"{graph.nodes[node]['type']}: {graph.nodes[node]['name']}"
        for node in nodes
    ]

    relations = []
    relation_counts = {}

    for u, v, data in subgraph.edges(data=True):
        relations.append(
            f"{graph.nodes[u]['name']} -{data['type']}-> {graph.nodes[v]['name']} "
            f"(вес: {data['weight']})"
        )
        relation_counts[data["type"]] = relation_counts.get(data["type"], 0) + data["weight"]

    total_weight = sum(relation_counts.values())

    messages = [
    {
        "role": "system",
        "content": (
            "Составь краткую сводку сообщества графа знаний не более 150 слов. "
            "Опиши входящие сущности, их взаимосвязи и наиболее часто встречающиеся "
            "типы связей. Используй только предоставленные данные. "
            "Не пересчитывай числа, используй готовые статистические данные."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Сущности:\n{entities}\n\n"
            f"Связи:\n{relations}\n\n"
            f"Количество связей по типам (сумма весов):\n{relation_counts}\n\n"
            f"Общий вес связей: {total_weight}"
        ),
    },
]

    response = llm.chat(messages)
    return " ".join(response.content.split()[:150])
