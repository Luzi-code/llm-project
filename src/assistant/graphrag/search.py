"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from assistant.infra.llm import LLMClient
import networkx as nx
from assistant.infra.grounding import find_codes, find_names, code_type
import json
from assistant.infra.schemas import Answer, KnowledgeBase, MAP_SCHEMA


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # TODO: опорные сущности, отбор фактов, запрос к модели, источники
    graph = kb.graph

    codes = find_codes(question)
    names = find_names(question, kb.vocabulary)

    anchors = set()

    for code in codes:
        node_id = f"{code_type(code)}:{code}"
        if graph.has_node(node_id):
            anchors.add(node_id)

    for entity_type, name in names:
        node_id = f"{entity_type}:{name}"
        if graph.has_node(node_id):
            anchors.add(node_id)

    if codes and any(
        not graph.has_node(f"{code_type(code)}:{code}")
        for code in codes
    ):
        return Answer(text="", sources=[], facts=[], refused=True)
    undirected = graph.to_undirected()
    selected_edges = set()

    if len(anchors) >= 2:
        anchor_list = sorted(anchors)

        for i, source in enumerate(anchor_list):
            for target in anchor_list[i + 1:]:
                for path in nx.all_simple_paths(
                    undirected, source, target, cutoff=3
                ):
                    for u, v in zip(path, path[1:]):
                        selected_edges.add(frozenset((u, v)))
    elif len(anchors) == 1:
        anchor = next(iter(anchors))
        neighborhood = nx.ego_graph(undirected, anchor, radius=2)

        for u, v in neighborhood.edges():
            selected_edges.add(frozenset((u, v)))

    path_nodes = set()
    for edge in selected_edges:
        path_nodes.update(edge)

    parts = [
        node for node in path_nodes
        if graph.nodes[node]["type"] == "Part"
    ]

    for part in parts:
        current = [part]
        visited = {part}

        while current:
            node = current.pop()

            for neighbor in graph.successors(node):
                if graph[node][neighbor]["type"] != "REPLACED_BY":
                    continue

                selected_edges.add(frozenset((node, neighbor)))

                if neighbor not in visited:
                    visited.add(neighbor)
                    current.append(neighbor)
    facts = []

    for u, v, data in graph.edges(data=True):
        if frozenset((u, v)) not in selected_edges:
            continue

        facts.append(
            f"{graph.nodes[u]['name']} -{data['type']}-> "
            f"{graph.nodes[v]['name']}"
        )

    facts = list(dict.fromkeys(facts))

    numbered_facts = []
    total_length = 0

    for fact in facts:
        number = len(numbered_facts) + 1
        formatted = f"{number}. {fact}"

        if total_length + len(formatted) + 1 > 8000:
            break

        numbered_facts.append(formatted)
        total_length += len(formatted) + 1

    fact_text = "\n".join(numbered_facts)

    if not fact_text:
        return Answer(text="", sources=[], facts=[], refused=True)

    messages = [
        {
            "role": "system",
            "content": (
                "Ответь на вопрос исключительно по предоставленным фактам. "
                "Не добавляй неподтверждённую информацию. "
                "Если присутствует цепочка REPLACED_BY, укажи последнюю деталь цепочки. "
                "Ссылайся на номера фактов, подтверждающих ответ."
            ),
        },
        {
            "role": "user",
            "content": f"Вопрос: {question}\n\nФакты:\n{fact_text}",
        },
    ]

    response = llm.chat(messages)
    sources = set()

    for u, v, data in graph.edges(data=True):
        if frozenset((u, v)) not in selected_edges:
            continue

        for source in data.get("sources", []):
            sources.add(source.split("#", 1)[0])

    return Answer(
        text=response.content.strip(),
        sources=sorted(sources),
        facts=numbered_facts,
        refused=False,
    )


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    partial_answers = []
    sources = set()
    for community in kb.communities:
        if not community.summary.strip():
            continue

        messages = [
            {
                "role": "system",
                "content": (
                    "Ответь на вопрос пользователя, используя только сводку сообщества. "
                    "Если сводка не содержит полезной информации для ответа, верни пустой answer "
                    "и score 0. Оцени полезность ответа от 0 до 10. "
                    "Верни JSON с полями answer и score."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    f"Сводка сообщества:\n{community.summary}"
                ),
            },
        ]

        try:
            response = llm.chat(messages, json_schema=MAP_SCHEMA)
            data = json.loads(response.content)

            answer = data["answer"]
            score = data["score"]

            if not isinstance(answer, str) or not isinstance(score, int):
                continue

            if score > 0 and answer.strip():
                partial_answers.append((answer, score))

                for node in community.nodes:
                    if kb.graph.has_node(node):
                        sources.update(
                            source.split("#", 1)[0]
                            for source in kb.graph.nodes[node].get("sources", [])
                        )

        except Exception:
            continue


    if not partial_answers:
        return Answer(text="", sources=[], facts=[], refused=True)

    partial_text = "\n\n".join(
        f"{i}. Оценка полезности: {score}/10\nОтвет: {answer}"
        for i, (answer, score) in enumerate(partial_answers, 1)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Объедини частичные ответы в единый связный ответ на вопрос. "
                "Используй только информацию из частичных ответов. "
                "Не добавляй неподтверждённых фактов и не повторяй одно и то же."
            ),
        },
        {
            "role": "user",
            "content": f"Вопрос: {question}\n\nЧастичные ответы:\n{partial_text}",
        },
    ]

    try:
        response = llm.chat(messages)
    except Exception:
        return Answer(
            text="\n\n".join(answer for answer, _ in partial_answers),
            sources=sorted(sources),
            facts=[answer for answer, _ in partial_answers],
            refused=False,
        )

    return Answer(
        text=response.content.strip(),
        sources=sorted(sources),
        facts=[answer for answer, _ in partial_answers],
        refused=False,
    )
