"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


_PART = {"type": "string", "pattern": "^P-[0-9]{4}[A-Z]?$"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
_QUESTION = {"type": "string", "minLength": 1}

TOOL_SCHEMAS: list[dict] = [
    _schema("search_knowledge", "Найти конкретные факты в графе знаний", {"question": _QUESTION}),
    _schema("summarize_topic", "Обобщить сведения по теме", {"question": _QUESTION}),
    _schema("check_stock", "Проверить остаток детали на складе", {"part_id": _PART, "warehouse": _WAREHOUSE}),
    _schema("get_price", "Получить цену детали в рублях", {"part_id": _PART}),
    _schema(
        "create_service_order",
        "Создать сервисный заказ после подтверждения пользователя",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "minLength": 1},
        },
    ),
]


def _valid_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_part(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"P-[0-9]{4}[A-Z]?", value) is not None


def _erp_call(method: Callable, **kwargs) -> dict:
    try:
        return method(**kwargs)
    except ErpNotFound as exc:
        return {"error": f"Данных в ERP нет: {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Возвращает пять инструментов с проверкой аргументов."""

    def search_knowledge(question: str) -> dict:
        if not _valid_text(question):
            return {"error": "Вопрос должен быть непустой строкой"}
        result = local_search(question, kb, llm)
        return {"answer": result.text, "facts": result.facts, "sources": result.sources, "refused": result.refused}

    def summarize_topic(question: str) -> dict:
        if not _valid_text(question):
            return {"error": "Вопрос должен быть непустой строкой"}
        result = global_search(question, kb, llm)
        return {"answer": result.text, "sources": result.sources, "refused": result.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        if not _valid_part(part_id):
            return {"error": "Некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад"}
        return _erp_call(erp.get_stock, part_id=part_id, warehouse=warehouse)

    def get_price(part_id: str) -> dict:
        if not _valid_part(part_id):
            return {"error": "Некорректный номер детали"}
        return _erp_call(erp.get_price, part_id=part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        if not _valid_part(part_id):
            return {"error": "Некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад"}
        if type(qty) is not int or qty <= 0:
            return {"error": "Количество должно быть положительным целым числом"}
        if not _valid_text(model):
            return {"error": "Не указана модель оборудования"}
        return _erp_call(erp.create_order, part_id=part_id, warehouse=warehouse, qty=qty, model=model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
