"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from src.assistant.graphrag.search import local_search, global_search
import re
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Поиск фактов в графе знаний по конкретному вопросу.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Поиск и обобщение информации по теме через сообщества графа.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверка остатка детали на складе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                    "warehouse": {
                        "type": "string",
                        "enum": WAREHOUSES,
                    },
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получение цены детали в рублях.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                },
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создание сервисного заказа на деталь.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                    "warehouse": {
                        "type": "string",
                        "enum": WAREHOUSES,
                    },
                    "qty": {"type": "integer"},
                    "model": {"type": "string"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


_PART_PATTERN = re.compile(r"^P-\d+[A-Z]?$")



def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    def validate_part(part_id: str) -> None:
        if not isinstance(part_id, str) or not _PART_PATTERN.fullmatch(part_id):
            raise ValueError("Некорректный номер детали.")

    def validate_warehouse(warehouse: str) -> None:
        if warehouse not in WAREHOUSES:
            raise ValueError("Неизвестный склад.")

    def handle_erp_errors(func: Callable[..., dict]) -> Callable[..., dict]:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ErpNotFound, ErpRejected) as exc:
                return {"error": str(exc)}
        return wrapper

    @handle_erp_errors
    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Пустой вопрос."}

        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    @handle_erp_errors
    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Пустой вопрос."}

        answer = global_search(question, kb, llm)
        return {
            "answer": answer.text,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    @handle_erp_errors
    def check_stock(part_id: str, warehouse: str) -> dict:
        validate_part(part_id)
        validate_warehouse(warehouse)
        return erp.get_stock(part_id, warehouse)

    @handle_erp_errors
    def get_price(part_id: str) -> dict:
        validate_part(part_id)
        return erp.get_price(part_id)

    @handle_erp_errors
    def create_service_order(
        part_id: str, warehouse: str, qty: int, model: str
    ) -> dict:
        validate_part(part_id)
        validate_warehouse(warehouse)

        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            raise ValueError("Количество должно быть целым числом больше нуля.")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("Не указана модель оборудования.")

        result = erp.create_order(part_id, warehouse, qty, model)
        return {"order_id": result["order_id"]}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
