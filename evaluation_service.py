import json
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


LM_STUDIO_CHAT_BASE_URL = os.getenv("LM_STUDIO_CHAT_BASE_URL", "http://192.168.1.42:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "qwen/qwen3.5-4b")
EVALUATOR_TEMPERATURE = float(os.getenv("EVALUATOR_TEMPERATURE", "0.1"))
EVALUATOR_MAX_TOKENS = int(os.getenv("EVALUATOR_MAX_TOKENS", "4096"))
EVALUATOR_TIMEOUT = float(os.getenv("EVALUATOR_TIMEOUT", "120"))
TRUST_MODEL_RAG_USAGE = os.getenv("TRUST_MODEL_RAG_USAGE", "false").lower() in {
    "1",
    "true",
    "yes",
}

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("EVALUATION_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
ALLOW_CREDENTIALS = os.getenv("EVALUATION_ALLOW_CREDENTIALS", "false").lower() in {
    "1",
    "true",
    "yes",
}

client = OpenAI(
    base_url=LM_STUDIO_CHAT_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    timeout=EVALUATOR_TIMEOUT,
)

app = FastAPI(
    title="hack-rag evaluation service",
    description="Evaluates procurement trainer selections using LM Studio and hack-rag MCP context.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: Any = Field(None, description="Trainer task title/object or legacy task object.")
    selected_items: list[dict[str, Any]] | None = Field(None, description="Legacy selected products.")
    levels: list[dict[str, Any]] | None = Field(None, description="Frontend level results with chosen items.")
    microTasks: list[dict[str, Any]] | None = Field(None, description="Optional micro task answers.")
    rubricHint: str | None = Field(None, description="Frontend rubric hint.")
    user_answer: dict[str, Any] | None = Field(None, description="Optional extra frontend state or totals.")
    locale: str = Field("ru", description="Response language hint.")


MASTER_PROMPT = """
Ты строгий эксперт по российским госзакупкам и проверяющий учебного тренажера для джунов.

Твоя задача: оценить итоговую попытку студента в тренажере закупок.

В новом формате фронтенд передает:
- `levels[]`: каждый уровень, тема, требуемый товар, количество, источник/норма, подсказка-ловушка `watch`,
  выбранный товар `chosen`;
- `microTasks[]`: дополнительные вопросы по НМЦК/44-ФЗ;
- `rubricHint`: явная рубрика проверки.

Оцени подбор относительно каждого уровня: соответствие предмету закупки, критериям, количеству,
единицам измерения, срочности, наличию, стране происхождения, нацрежиму, совместимости, РНП,
минимальной партии, НДС, покупке/аренде и НМЦК/цене.

Для `microTasks[]` сравнивай `candidateAnswer` с правильным вариантом из `options` по смыслу.
Не считай ответ правильным только потому, что студент выбрал один из вариантов.
Если в microTask нет отдельного поля `correctAnswer`/`expectedAnswer`, используй первый элемент `options`
как эталонный правильный ответ.

У тебя в окружении должен быть доступен MCP server `hack-rag` с инструментами:
- `search_rag` — поиск по локальной базе документов 44-ФЗ, 135-ФЗ, КоАП, УК РФ, обзорам ВС РФ и материалам тренажера;
- `rag_stats` — статистика RAG индекса.

Перед финальной оценкой используй `search_rag` самостоятельно, когда нужно проверить норму закона,
ограничение, национальный режим, описание объекта закупки, сроки исполнения, требования к характеристикам
или иное юридически значимое правило. Не выдумывай нормы права: опирайся на найденные источники.

Если инструмент недоступен в твоем окружении, честно отрази это в поле `rag_status`, но все равно оцени
по данным задания и выбранных товаров.

Строго запрещено выдумывать использование RAG. `rag_status` может быть `used` только если ты реально вызвал
инструмент `search_rag` или `rag_stats` и получил результат инструмента в текущем контексте. Если реального
ответа инструмента нет, поставь `rag_status: "not_available"` и `rag_evidence: []`. Не придумывай `source_path`,
номер chunk или law_reference из RAG без реального tool output.

Если `rag_status` не равен `used`, поле `law_reference` в нарушениях можно заполнять только ссылками,
которые явно пришли во входном JSON в `task.legal_hints` или других полях задания. Если явной ссылки нет,
используй пустую строку.

Верни только валидный JSON. Никакого Markdown, пояснений вне JSON, code fences или обычного текста.

Схема ответа строго такая:
{
  "is_correct": boolean,
  "score": integer от 0 до 10,
  "readiness_score": integer от 0 до 100,
  "verdict": "correct" | "partially_correct" | "incorrect",
  "recommendation": "recommended" | "needs_review" | "not_recommended",
  "rag_status": "used" | "not_available" | "not_needed",
  "summary": "короткий итог проверки",
  "comment": "комментарий для карточки кандидата",
  "competencies": [
    {
      "name": "название компетенции",
      "score": integer от 0 до 100,
      "comment": "пояснение"
    }
  ],
  "level_results": [
    {
      "level": "l1",
      "theme": "тема уровня",
      "expected_item": "что надо было подобрать",
      "chosen_item": "что выбрал студент",
      "passed": boolean,
      "score": integer от 0 до 10,
      "comment": "почему уровень засчитан или нет",
      "issues": ["ошибки уровня"]
    }
  ],
  "microtask_results": [
    {
      "question": "вопрос",
      "candidate_answer": "ответ кандидата",
      "passed": boolean,
      "correct_answer": "правильный ответ",
      "comment": "пояснение"
    }
  ],
  "criteria_check": [
    {
      "criterion": "название критерия",
      "passed": boolean,
      "comment": "почему да или нет"
    }
  ],
  "violations": [
    {
      "type": "law" | "task" | "quality" | "price" | "quantity" | "delivery" | "availability" | "other",
      "severity": "low" | "medium" | "high" | "critical",
      "description": "что нарушено",
      "law_reference": "ссылка на норму или источник, если есть",
      "affected_items": ["наименования товаров"],
      "fix": "как исправить"
    }
  ],
  "item_feedback": [
    {
      "item_name": "наименование товара",
      "accepted": boolean,
      "comment": "почему товар подходит или нет"
    }
  ],
  "rag_evidence": [
    {
      "query": "какой запрос был использован",
      "source_path": "источник из RAG",
      "chunk": "номер/описание фрагмента, если есть",
      "evidence": "краткое содержание найденного правила"
    }
  ],
  "recommended_action": "что студенту сделать дальше"
}

Правила оценки:
- 10: подбор полностью соответствует заданию и закону.
- 7-9: есть мелкие недочеты без существенного нарушения задания.
- 4-6: часть критериев выполнена, но есть существенные ошибки.
- 1-3: подбор в основном неверный или незаконный.
- 0: ответ пустой, очевидно не по заданию или содержит критическое нарушение.

Перевод в `readiness_score`: примерно `score * 10`, но можно скорректировать на скорость, stress/fatigue,
microTasks и системность ошибок.

Рекомендация:
- `recommended`: readiness_score >= 80 и нет критических нарушений.
- `needs_review`: readiness_score 50-79 или есть существенные, но исправимые ошибки.
- `not_recommended`: readiness_score < 50 либо много неверных уровней.

Обязательные компетенции для поля `competencies`:
- "Знание 44-ФЗ и структуры"
- "Подбор позиций по критериям"
- "Расчёт НМЦК"
- "Внимательность к ловушкам"
- "Устойчивость к нагрузке"

Если пришел `levels[]`, обязательно верни по одному объекту `level_results` на каждый уровень.
Если пришел старый формат `selected_items`, оцени его как один набор.
""".strip()

EVALUATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "readiness_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "recommendation": {
            "type": "string",
            "enum": ["recommended", "needs_review", "not_recommended"],
        },
        "rag_status": {"type": "string", "enum": ["used", "not_available", "not_needed"]},
        "summary": {"type": "string"},
        "comment": {"type": "string"},
        "competencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "comment": {"type": "string"},
                },
                "required": ["name", "score", "comment"],
            },
        },
        "level_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "string"},
                    "theme": {"type": "string"},
                    "expected_item": {"type": "string"},
                    "chosen_item": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 10},
                    "comment": {"type": "string"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "level",
                    "theme",
                    "expected_item",
                    "chosen_item",
                    "passed",
                    "score",
                    "comment",
                    "issues",
                ],
            },
        },
        "microtask_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "candidate_answer": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "correct_answer": {"type": "string"},
                    "comment": {"type": "string"},
                },
                "required": [
                    "question",
                    "candidate_answer",
                    "passed",
                    "correct_answer",
                    "comment",
                ],
            },
        },
        "criteria_check": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "comment": {"type": "string"},
                },
                "required": ["criterion", "passed", "comment"],
            },
        },
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "law",
                            "task",
                            "quality",
                            "price",
                            "quantity",
                            "delivery",
                            "availability",
                            "other",
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "description": {"type": "string"},
                    "law_reference": {"type": "string"},
                    "affected_items": {"type": "array", "items": {"type": "string"}},
                    "fix": {"type": "string"},
                },
                "required": [
                    "type",
                    "severity",
                    "description",
                    "law_reference",
                    "affected_items",
                    "fix",
                ],
            },
        },
        "item_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "accepted": {"type": "boolean"},
                    "comment": {"type": "string"},
                },
                "required": ["item_name", "accepted", "comment"],
            },
        },
        "rag_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source_path": {"type": "string"},
                    "chunk": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["query", "source_path", "chunk", "evidence"],
            },
        },
        "recommended_action": {"type": "string"},
    },
    "required": [
        "is_correct",
        "score",
        "readiness_score",
        "verdict",
        "recommendation",
        "rag_status",
        "summary",
        "comment",
        "competencies",
        "level_results",
        "microtask_results",
        "criteria_check",
        "violations",
        "item_feedback",
        "rag_evidence",
        "recommended_action",
    ],
}


def build_user_prompt(request: EvaluationRequest) -> str:
    payload = request.model_dump(mode="json", exclude_none=True)
    levels = payload.get("levels") or []
    level_summary = []
    if isinstance(levels, list):
        for level in levels:
            if not isinstance(level, dict):
                continue
            chosen = level.get("chosen") or {}
            if not isinstance(chosen, dict):
                chosen = {}
            level_summary.append(
                {
                    "level": level.get("level"),
                    "theme": level.get("theme"),
                    "expected_item": level.get("item"),
                    "qty_needed": level.get("qtyNeeded"),
                    "unit_needed": level.get("unitNeeded"),
                    "source": level.get("source"),
                    "trap_or_watch": level.get("watch"),
                    "chosen_item": chosen.get("name"),
                    "chosen_attrs": chosen.get("attrs"),
                    "chosen_country": chosen.get("country"),
                    "chosen_qty": chosen.get("chosenQty"),
                    "chosen_units_in_pieces": chosen.get("chosenUnitsInPieces"),
                    "chosen_delivery_days": chosen.get("srok"),
                    "chosen_availability": chosen.get("avail"),
                    "chosen_supplier": chosen.get("supplier"),
                    "chosen_rnp": chosen.get("rnp"),
                    "chosen_nds": chosen.get("nds"),
                    "chosen_min_batch": chosen.get("minBatch"),
                    "chosen_unit_price": chosen.get("unitPrice"),
                    "chosen_cost": chosen.get("cost"),
                }
            )

    microtasks = payload.get("microTasks") or []
    microtask_summary = []
    if isinstance(microtasks, list):
        for task in microtasks:
            if not isinstance(task, dict):
                continue
            options = task.get("options") or []
            expected_answer = (
                task.get("correctAnswer")
                or task.get("expectedAnswer")
                or task.get("answer")
                or (options[0] if isinstance(options, list) and options else None)
            )
            microtask_summary.append(
                {
                    "question": task.get("question"),
                    "candidate_answer": task.get("candidateAnswer"),
                    "expected_answer": expected_answer,
                    "options": options,
                }
            )

    return (
        "Проверь результат подбора в тренажере закупок.\n"
        "Входные данные ниже переданы backend'ом фронтенда в JSON.\n"
        "Основной формат — `levels[]`, где в каждом уровне есть `watch` с ловушкой и `chosen` с выбором кандидата.\n"
        "Сначала проверь каждый уровень по `watch`, `source`, количеству и выбранному товару.\n"
        "Затем оцени microTasks по `expected_answer` и общий профиль компетенций.\n"
        "Если в твоем окружении доступны tools MCP hack-rag, используй их для юридических проверок.\n"
        "Верни финальную оценку строго JSON по схеме из system prompt.\n\n"
        "Краткая выжимка уровней для проверки:\n"
        f"{json.dumps(level_summary, ensure_ascii=False, indent=2)}\n\n"
        "Краткая выжимка microTasks для проверки:\n"
        f"{json.dumps(microtask_summary, ensure_ascii=False, indent=2)}\n\n"
        "Полный payload фронтенда:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def extract_message_content(message: Any) -> str:
    content = message.content or getattr(message, "reasoning_content", None)
    if not content:
        raise RuntimeError("LM Studio returned an empty response.")
    return content


def call_json_schema_completion(messages: list[dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        temperature=EVALUATOR_TEMPERATURE,
        max_tokens=EVALUATOR_MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "procurement_selection_evaluation",
                "schema": EVALUATION_RESPONSE_SCHEMA,
            },
        },
        messages=messages,
    )
    return extract_message_content(response.choices[0].message)


def repair_model_json(raw_response: str, parse_error: str) -> dict[str, Any]:
    repair_prompt = """
Ты исправляешь ответ модели в валидный JSON.
Верни только один JSON object по заданной схеме. Никакого Markdown и текста вне JSON.
Не добавляй новые выводы и не меняй смысл оценки, только исправь синтаксис и пропущенные обязательные поля.
Если значение невозможно восстановить, используй безопасное краткое значение на русском языке.
""".strip()
    repaired = call_json_schema_completion(
        [
            {"role": "system", "content": repair_prompt},
            {
                "role": "user",
                "content": (
                    "Ниже почти-JSON, который не удалось распарсить.\n"
                    f"Ошибка парсинга: {parse_error}\n\n"
                    "Исправь его в валидный JSON по схеме:\n"
                    f"{raw_response}"
                ),
            },
        ]
    )
    return extract_json_object(repaired)


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    if not TRUST_MODEL_RAG_USAGE and result.get("rag_status") == "used":
        result["rag_status"] = "not_available"
        result["rag_evidence"] = []
    return result


def call_evaluator(request: EvaluationRequest) -> dict[str, Any]:
    content = call_json_schema_completion(
        [
            {"role": "system", "content": MASTER_PROMPT},
            {"role": "user", "content": build_user_prompt(request)},
        ]
    )
    try:
        return normalize_result(extract_json_object(content))
    except (json.JSONDecodeError, ValueError) as exc:
        return normalize_result(repair_model_json(content, str(exc)))


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "hack-rag evaluation service",
        "routes": ["/health", "/evaluate-selection"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "lm_studio_chat_base_url": LM_STUDIO_CHAT_BASE_URL,
        "model": EVALUATOR_MODEL,
    }


@app.post("/evaluate-selection")
def evaluate_selection(request: EvaluationRequest) -> JSONResponse:
    try:
        result = call_evaluator(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(content=result)


@app.post("/evaluate")
def evaluate_alias(request: EvaluationRequest) -> JSONResponse:
    return evaluate_selection(request)
