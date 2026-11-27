"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, ExtractionResult, EXTRACTION_SCHEMA, Entity, Relation
from assistant.infra.grounding import find_codes, canonical_name, load_vocabulary, find_names
import json



def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    result = ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)

    vocabulary = load_vocabulary('data/vocabulary.json')

    codes = set(find_codes(chunk.text))
    names = set(find_names(chunk.text, vocabulary))

    messages = [
            {
                "role": "system",
                "content": (
                    "Извлеки сущности и направленные связи из текста. "
                    "Используй только факты, прямо подтверждённые текстом. "
                    "Не придумывай сущности и связи. "
                    "Соблюдай типы и направление связей согласно схеме. "
                    "Пример: модель AR-210 содержит деталь P-1042: "
                    '{"entities":[{"type":"Model","name":"AR-210"},'
                    '{"type":"Part","name":"P-1042"}],'
                    '"relations":[{"subject_type":"Model","subject":"AR-210",'
                    '"type":"HAS_PART","object_type":"Part","object":"P-1042"}]}'
                )
            },
            {"role": "user", "content": chunk.text},
        ]
    
    for _ in range(2):
        try:
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            data = json.loads(response.content)
            if not isinstance(data, dict) or not isinstance(data.get("entities"), list) or not isinstance(data.get("relations"), list):
                raise ValueError("Invalid extraction response")
            break
        except Exception:
            pass
    else:
        return result

    entities = {}
    dropped_entities = set()
    entity_candidates = list(data["entities"])

    for item in data["relations"]:
        entity_candidates.append({
            "type": item["subject_type"],
            "name": item["subject"],
        })
        entity_candidates.append({
            "type": item["object_type"],
            "name": item["object"],
        })
    code_types = ("Model", "Part", "ErrorCode", "Procedure")

    for item in entity_candidates:
        name = canonical_name(item["type"], item["name"], vocabulary)

        if name is None or (item["type"] in code_types and name not in codes) or (
            item["type"] not in code_types and (item["type"], name) not in names):
            dropped_name = name if name is not None else item["name"].strip().casefold()
            dropped_entities.add((item["type"], dropped_name))
            continue

        entity = Entity(item["type"], name)
        entities[entity.node_id] = entity

    result.dropped += len(dropped_entities)
    result.entities = list(entities.values())

    for item in data["relations"]:
        subject_name = canonical_name(item["subject_type"], item["subject"], vocabulary)
        object_name = canonical_name(item["object_type"], item["object"], vocabulary)

        subject = entities.get(f"{item['subject_type']}:{subject_name}")
        object_ = entities.get(f"{item['object_type']}:{object_name}")

        if subject and object_:
            relation = Relation(subject, item["type"], object_)
            if relation.matches_schema():
                result.relations.append(relation)
                continue

        result.dropped += 1

    return result
