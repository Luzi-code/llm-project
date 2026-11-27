"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена."""
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))

    for u, v, attrs in sorted(graph.edges(data=True)):
        weight = attrs.get("weight", 1)

        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += weight
        else:
            undirected.add_edge(u, v, weight=weight)

    if undirected.size(weight="weight") == 0:
        groups = [{node} for node in undirected.nodes]
    else:
        groups = nx.community.louvain_communities(
            undirected,
            weight="weight",
            seed=SEED,
        )

    return [Community(i, nodes) for i, nodes in enumerate(sorted(sorted(group) for group in groups))]


def summarize_community(
    community: Community,
    graph: nx.DiGraph,
    llm: LLMClient,
) -> str:
    """Создаёт сводку сообщества одним вызовом модели."""
    subgraph = graph.subgraph(community.nodes)

    entities = []
    facts = []
    relation_counts = Counter()
    failure_counts = Counter()

    for node, attrs in sorted(subgraph.nodes(data=True)):
        entities.append(f"{attrs.get('type', '')}: {attrs.get('name', node)}")

    for u, v, attrs in sorted(subgraph.edges(data=True)):
        relation = attrs["type"]
        weight = attrs.get("weight", 1)

        source = graph.nodes[u].get("name", u)
        target = graph.nodes[v].get("name", v)

        facts.append(f"{source} -{relation}-> {target}; вес: {weight}")

        relation_counts[relation] += weight

        if relation == "FAILED":
            failure_counts[target] += weight

    prompt = "\n".join(
        [
            f"Число сущностей: {subgraph.number_of_nodes()}",
            f"Число связей: {subgraph.number_of_edges()}",
            "Сущности:",
            *entities,
            "Связи и их веса (число чанков):",
            *facts,
            "Суммы весов по типам связей:",
            *(f"{name}: {count}" for name, count in relation_counts.most_common()),
            "Суммы весов FAILED по компонентам:",
            *(f"{name}: {count}" for name, count in failure_counts.most_common()),
        ]
    )

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши сводку сообщества графа на русском языке, "
                    "не более 150 слов. "
                    "Опиши сущности, связи и наиболее частые явления. "
                    "Используй только переданные факты и готовые числа. "
                    "Не выдумывай данные и не пересчитывай статистику."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    return " ".join(response.content.split()[:150])
