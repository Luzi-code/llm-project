"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "В базе знаний не найдено достаточно данных для ответа."


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отбирает факты по путям между опорными сущностями или их окрестности."""
    graph = kb.graph
    anchors = {f"{code_type(code)}:{code}" for code in find_codes(question)}
    if not anchors.issubset(graph.nodes):
        return Answer(REFUSAL, refused=True)
    anchors.update(f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary))
    anchors.intersection_update(graph.nodes)
    if not anchors:
        return Answer(REFUSAL, refused=True)

    undirected = graph.to_undirected(as_view=True)
    edges: set[tuple[str, str]] = set()

    def add_edge(u: str, v: str) -> None:
        if graph.has_edge(u, v):
            edges.add((u, v))
        if graph.has_edge(v, u):
            edges.add((v, u))

    if len(anchors) == 1:
        distances = nx.single_source_shortest_path_length(undirected, next(iter(anchors)), cutoff=2)
        edges.update(graph.subgraph(distances).edges())
    else:
        for start, end in combinations(sorted(anchors), 2):
            for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
                for u, v in zip(path[:-1], path[1:], strict=True):
                    add_edge(u, v)

        pending = sorted({node for edge in edges for node in edge if graph.nodes[node]["type"] == "Part"})
        visited = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            for target, attrs in graph[node].items():
                if attrs["type"] == "REPLACED_BY":
                    edges.add((node, target))
                    pending.append(target)

    facts = []
    sources = set()
    lines = []
    size = 0
    for u, v in sorted(edges):
        attrs = graph[u][v]
        fact = f"{graph.nodes[u]['name']} -{attrs['type']}-> {graph.nodes[v]['name']}"
        line = f"{len(facts) + 1}. {fact}"
        if size + len(line) + 1 > 8000:
            break
        size += len(line) + 1
        lines.append(line)
        facts.append(fact)
        sources.update(source.split("#", 1)[0] for source in attrs.get("sources", []))
    if not facts:
        return Answer(REFUSAL, refused=True)

    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Отвечай только по приведённым фактам. Не выдумывай коды и связи. "
                    "При наличии цепочки REPLACED_BY назови последнюю деталь цепочки. "
                    "Если данных недостаточно, напиши: в базе знаний не найдено."
                ),
            },
            {"role": "user", "content": "Факты:\n" + "\n".join(lines) + f"\n\nВопрос: {question}"},
        ]
    )
    return Answer(
        reply.content or REFUSAL,
        sorted(sources),
        facts,
        refused=not reply.content.strip() or "не найдено" in reply.content.lower(),
    )


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Объединяет полезные частичные ответы по сводкам сообществ."""
    partials = []
    sources = set()
    for community in kb.communities:
        reply = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь только по сводке. Оцени полезность ответа от 0 до 10. "
                        "Если данных для ответа нет, поставь 0."
                    ),
                },
                {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            data = json.loads(reply.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        answer, score = data.get("answer"), data.get("score")
        if not isinstance(answer, str) or not answer.strip() or type(score) is not int or not 0 < score <= 10:
            continue
        partials.append(f"Сообщество {community.community_id}, оценка {score}: {answer}")
        subgraph = kb.graph.subgraph(community.nodes)
        for _, attrs in subgraph.nodes(data=True):
            sources.update(source.split("#", 1)[0] for source in attrs.get("sources", []))
        for _, _, attrs in subgraph.edges(data=True):
            sources.update(source.split("#", 1)[0] for source in attrs.get("sources", []))
    if not partials:
        return Answer(REFUSAL, refused=True)

    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы без выдуманных фактов и чисел. "
                    "Если данных недостаточно, напиши: в базе знаний не найдено."
                ),
            },
            {"role": "user", "content": "\n".join(partials) + f"\n\nВопрос: {question}"},
        ]
    )
    return Answer(
        reply.content or REFUSAL,
        sorted(sources),
        refused=not reply.content.strip() or "не найдено" in reply.content.lower(),
    )
