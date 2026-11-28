"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, Entity, ExtractionResult, EXTRACTION_SCHEMA, RELATION_SIGNATURES, Relation, ENTITY_TYPES
from assistant.infra.grounding import find_codes, canonical_name, load_vocabulary, find_names, KINDS, code_type
import json

ex_1="""миксер AR-210 содержит датчик температуры P-1099:
                    Ответ модели:  {
                        "entities": [
                            {"type": "Part", "name": "P-1099"}, {"type":"Model","name":"AR-210"},
                            {"type": "Component", "name": "датчик температуры"}
                        ],
                        "relations": [
                            {"subject_type":"Model","subject":"AR-210",
                            "type":"HAS_PART","object_type":"Part","object":"P-1099"},
                            {"subject_type": "Part", "subject": "P-1099", "type": "IS_A",
                            "object_type": "Component", "object": "датчик температуры"},
                        ],
                        }
                        """

ex_2 = '''#Регламент SP-108: Замена вентилятора испарителя
                        Регламент SP-108 применяется при ошибке E-24. Ошибка E-24 указывает на отказ узла «вентилятор испарителя». Норма времени – 55 минут."
                    Ответ модели:  {
                        "entities": [
                            {"type": "ErrorCode", "name": "E-24"},
                            {"type": "Component", "name": "вентилятор испарителя"},
                            {"type": "Procedure", "name": "SP-108"},
                        ],
                        "relations": [
                            {"subject_type": "ErrorCode", "subject": "E-24",
                                "type": "INDICATES",
                                "object_type": "Component", "object": "вентилятор испарителя"},
                            {"subject_type": "ErrorCode", "subject": "E-24",
                                "type": "FIXED_BY",
                                "object_type": "Procedure", "object": "SP-108"},
                        ]
                    }'''

ex_3 = '''Отчёт о выезде R-2025-026

                        Дата выезда: 27.07.2025. Город: Екатеринбург. Объект: супермаркет «Семейный».

                        Оборудование: витрина ar-320. На панели горел код Е-14, обрыв цепи датчика.

                        Диагностика показала отказ узла: датчик температуры. Узел заменён, оборудование запущено, температура вышла на режим за 50 минут.
            Ответ модели:{
                        "entities": [
                            {"type": "ErrorCode", "name": "Е-14"},
                            {"type": "Component", "name": "датчик температуры"},
                            {"type": "Model", "name": "AR-320"},
                        ],
                        "relations": [
                            {"subject_type": "ErrorCode", "subject": "Е-14",
                                "type": "INDICATES",
                                "object_type": "Component", "object": "датчик температуры"},
                            {
                            "subject_type": "Model", "subject": "AR-320",
                            "type": "FAILED",
                            "object_type": "Component", "object": "датчик температуры"
                        }
                        ],

                    }

'''

ex_4 = ''' Текст: # Сервисный бюллетень SB-2025-07
P-3126 (реле давления) снята с производства. Вместо неё устанавливается P-3126B.
Поставщик детали P-3126B – «АвтоматикаСевер». Детали взаимозаменяемы по посадочным местам и разъёмам, доработка оборудования не требуется.
Новые заказы оформляются только на P-3126B.
Ответ модели:{
                "entities": [
                    {"type": "Part", "name": "P-3126"},
                    {"type": "Part", "name": "P-3126B"},
                    {"type": "Component", "name": "реле давления"},
                    {"type": "Supplier", "name": "АвтоматикаСевер"},
                ],
                "relations": [
                    {"subject_type": "Part", "subject": "P-3126",
                        "type": "IS_A",
                        "object_type": "Component", "object": "реле давления"},
                    {"subject_type": "Part", "subject": "P-3126B",
                        "type": "IS_A",
                        "object_type": "Component", "object": "реле давления"},
                    {"subject_type": "Part", "subject": "P-3126B",
                        "type": "SUPPLIED_BY",
                        "object_type": "Supplier", "object": "АвтоматикаСевер"},
                    {"subject_type": "Part", "subject": "P-3126",
                        "type": "REPLACED_BY",
                        "object_type": "Part", "object": "P-3126B"},
                ]
            }

'''

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
    vocabulary_text = json.dumps(vocabulary, ensure_ascii=False)

    codes = set(find_codes(chunk.text))
    names = set(find_names(chunk.text, vocabulary))


    messages = [
            {
                "role": "system",
                "content": (
                    "Извлеки максимально полный набор сущностей и связей из текста. "
                    "При извлечении стремись к полноте: не пропускай факты, "
                    "даже если они выражены косвенно. Один факт может подразумевать "
                    "несколько связей разных типов. "
                    f"Если в тексте встречается последовательность из букв и цифр описанная в {KINDS}, это явный указатель на узел, его необходимо добавить. Помимо текста анализируй заголовок "
                    "Проверь текст на наличие каждого допустимого типа связи. Возможные виды сущностей:"
                    f"{ENTITY_TYPES}\n"
                    "Возможные виды связей:"
                    f"{RELATION_SIGNATURES}\n"
                    "Семантика связей:\n"
                    "BELONGS_TO — модель относится к продуктовой линейке.\n"
                    "HAS_PART — модель содержит деталь.\n"
                    "IS_A — деталь является экземпляром или представителем компонента.\n"
                    "INDICATES — код ошибки указывает на неисправность компонента.\n"
                    "FIXED_BY — ошибка устраняется процедурой/регламентом.\n"
                    "REPLACED_BY — деталь заменяется другой деталью.\n"
                    "SUPPLIED_BY — деталь поставляется поставщиком.\n"
                    "FAILED — у модели отказал компонент.\n"
                    f"Допустимые названия сущностей: {vocabulary_text}\n"
                    f'''Пример 1: {ex_1}
                        Пример 2: {ex_2}
                        Пример 3: {ex_3}
                        Пример 4: {ex_4}
'''
                    

                )
            },
            {
                "role": "user",
                "content": (
                    f"Текст документа:\n{chunk.text}\n\n"
                    f"Найденные коды: {sorted(codes)}\n"
                    f"Найденные названия (тип, имя): {sorted(names)}\n"
                    "Используй найденные кандидаты при извлечении, но не создавай связи "
                    "только на основании совместного присутствия сущностей."
                ),
            },
        ]
    
    for _ in range(2):
        try:
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            data = json.loads(response.content)

            if (
                not isinstance(data, dict)
                or not isinstance(data.get("entities"), list)
                or not isinstance(data.get("relations"), list)
            ):
                raise ValueError("Ожидались массивы entities и relations")

            break

        except Exception as exc:
            messages.append({
                "role": "user",
                "content": f"Исправь предыдущий ответ. Ошибка: {exc}",
            })
    else:
        return result

    entities = {}
    for item in data["relations"]:
        expected = RELATION_SIGNATURES.get(item["type"])
        actual = (item["subject_type"], item["object_type"])

        if expected != actual and expected == actual[::-1]:
            item["subject_type"], item["object_type"] = (
                item["object_type"], item["subject_type"]
            )
            item["subject"], item["object"] = item["object"], item["subject"]
            
    dropped_entities = set()
    entity_candidates = list(data["entities"])

    entity_candidates.extend(
        {"type": code_type(code), "name": code}
        for code in codes
    )

    entity_candidates.extend(
        {"type": kind, "name": name}
        for kind, name in names
    )

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

    relations = set()

    for item in data["relations"]:
        subject_name = canonical_name(item["subject_type"], item["subject"], vocabulary)
        object_name = canonical_name(item["object_type"], item["object"], vocabulary)

        subject_valid = (
            subject_name is not None
            and (
                (item["subject_type"] in code_types and subject_name in codes)
                or (item["subject_type"] not in code_types and (item["subject_type"], subject_name) in names)
            )
        )

        object_valid = (
            object_name is not None
            and (
                (item["object_type"] in code_types and object_name in codes)
                or (item["object_type"] not in code_types and (item["object_type"], object_name) in names)
            )
        )

        subject = Entity(item["subject_type"], subject_name) if subject_valid else None
        object_ = Entity(item["object_type"], object_name) if object_valid else None

        if subject and object_:
            # relation = Relation(subject, item["type"], object_)

            # expected = RELATION_SIGNATURES.get(item["type"])
            # if expected and (subject.type, object_.type) != expected:
            #     if (object_.type, subject.type) == expected and subject.type != object_.type:
            #         subject, object_ = object_, subject

            relation = Relation(subject, item["type"], object_)

            if relation.matches_schema():
                entities[subject.node_id] = subject
                entities[object_.node_id] = object_
                result.entities = list(entities.values())
                relations.add(relation)
                continue

        if not subject or not object_ or not relation.matches_schema():
            result.dropped += 1
            continue
    result.relations = sorted(
        relations,
        key=lambda r: (r.subject.node_id, r.type, r.object.node_id)
    )

    return result
