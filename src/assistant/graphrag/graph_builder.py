"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен
    graph = nx.DiGraph()
    conflicts = 0

    def add_entity(entity: Entity, chunk_id: str) -> None:
        node_id = entity.node_id
        if not graph.has_node(node_id):
            graph.add_node(node_id, type=entity.type, name=entity.name, sources=[])
        sources = graph.nodes[node_id]["sources"]
        if chunk_id not in sources:
            sources.append(chunk_id)

    # Узлы
    for result in results:
        for entity in result.entities:
            add_entity(entity, result.chunk_id)

    # Рёбра
    for result in results:
        for relation in result.relations:
            if not relation.matches_schema():
                continue

            add_entity(relation.subject, result.chunk_id)
            add_entity(relation.object, result.chunk_id)
            u = relation.subject.node_id
            v = relation.object.node_id

            if graph.has_edge(u, v) and graph[u][v]["type"] == relation.type:
                if result.chunk_id not in graph[u][v]["sources"]:
                    graph[u][v]["sources"].append(result.chunk_id)
                    graph[u][v]["weight"] += 1
            else:
                graph.add_edge(u, v, type=relation.type,
                            weight=1, sources=[result.chunk_id])

    # Встречные REPLACED_BY
    for u, v in list(graph.edges()):
        if not graph.has_edge(u, v):
            continue

        if graph[u][v]["type"] != "REPLACED_BY":
            continue

        if not graph.has_edge(v, u) or graph[v][u]["type"] != "REPLACED_BY":
            continue

        weight_uv = graph[u][v]["weight"]
        weight_vu = graph[v][u]["weight"]

        if weight_uv > weight_vu:
            graph.remove_edge(v, u)
        elif weight_vu > weight_uv:
            graph.remove_edge(u, v)
        else:
            graph.remove_edge(u, v)
            graph.remove_edge(v, u)

        conflicts += 1

    graph.graph["conflicts"] = conflicts
    return graph