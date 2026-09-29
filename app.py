import os
import re
import json
import hmac
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sales_copilot.sqlite3"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:3b"
)

AMO_CHANNEL_SECRET = os.getenv(
    "AMO_CHANNEL_SECRET",
    ""
)


app = FastAPI(
    title="AI Sales Copilot",
    description="AI assistant for CRM managers",
    version="1.0.0"
)


# ==================================================
# DATABASE
# ==================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            keywords TEXT NOT NULL,
            answer TEXT NOT NULL,
            upsell TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_message TEXT NOT NULL,
            client_reply TEXT NOT NULL,
            manager_upsell TEXT NOT NULL,
            used_kb TEXT,
            ai_mode TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_knowledge_active
        ON knowledge(active);

        CREATE INDEX IF NOT EXISTS idx_interactions_created
        ON interactions(created_at);
        """
    )

    count = conn.execute(
        "SELECT COUNT(*) FROM knowledge"
    ).fetchone()[0]

    if count == 0:
        demo = [
            (
                "Доставка",
                "доставка, доставить, курьер, срок, привезти, быстрее, срочно",
                "По демонстрационным условиям стандартная доставка занимает 1–3 рабочих дня.",
                "Если клиенту важна скорость, уточните срочность и предложите экспресс-доставку, если такая услуга доступна."
            ),
            (
                "Оплата",
                "оплата, оплатить, карта, счет, счёт, безнал, перевод",
                "По демонстрационным условиям оплатить заказ можно банковской картой или по счёту.",
                "Если клиент представляет компанию, уточните необходимость счёта и закрывающих документов."
            ),
            (
                "Гарантия",
                "гарантия, гарантийный, сломался, неисправность, возврат",
                "По демонстрационным условиям на услугу действует гарантийная поддержка. Точный срок зависит от выбранной услуги.",
                "Уточните, насколько критична бесперебойная работа, и при необходимости предложите расширенное обслуживание."
            ),
            (
                "Базовая услуга",
                "подключить, настройка, настроить, установка, установить, услуга",
                "Мы можем помочь с подключением и первоначальной настройкой услуги.",
                "После основной услуги можно предложить сопровождение или расширенную настройку, если это действительно решает задачу клиента."
            )
        ]

        conn.executemany(
            """
            INSERT INTO knowledge
            (title, keywords, answer, upsell)
            VALUES (?, ?, ?, ?)
            """,
            demo
        )

    conn.commit()
    conn.close()


init_db()


# ==================================================
# MODELS
# ==================================================

class AnalyzeRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    contact_name: Optional[str] = None


class KBRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    keywords: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=5000)
    upsell: str = Field(min_length=1, max_length=5000)


# ==================================================
# KNOWLEDGE SEARCH
# ==================================================

def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().replace("ё", "е")
    ).strip()


def tokens(text: str):
    return set(
        re.findall(
            r"[a-zа-я0-9]{3,}",
            normalize(text),
            flags=re.IGNORECASE
        )
    )


def search_knowledge(message: str, limit: int = 3):
    conn = db()

    rows = conn.execute(
        """
        SELECT *
        FROM knowledge
        WHERE active = 1
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    query = normalize(message)
    query_tokens = tokens(message)

    ranked = []

    for row in rows:
        keywords = [
            normalize(x)
            for x in re.split(r"[,;]", row["keywords"])
            if x.strip()
        ]

        blob = normalize(
            f'{row["title"]} '
            f'{row["keywords"]} '
            f'{row["answer"]}'
        )

        score = 0

        # Keywords are weighted higher.
        for keyword in keywords:
            if keyword and keyword in query:
                score += 7

        blob_tokens = tokens(blob)
        score += len(query_tokens & blob_tokens) * 2

        # Partial word matching for short demo KB.
        for token in query_tokens:
            if len(token) >= 5 and token in blob:
                score += 1

        if score > 0:
            ranked.append(
                (
                    score,
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "keywords": row["keywords"],
                        "answer": row["answer"],
                        "upsell": row["upsell"],
                        "score": score
                    }
                )
            )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [item[1] for item in ranked[:limit]]


# ==================================================
# AI
# ==================================================

def kb_to_context(items):
    if not items:
        return """
Подходящих сведений в базе знаний не найдено.
Нельзя придумывать цены, сроки, гарантии или условия.
Если данных недостаточно — нужно вежливо уточнить детали.
""".strip()

    blocks = []

    for i, item in enumerate(items, 1):
        blocks.append(
            f"""
Запись #{i}
Название: {item["title"]}
Ответ из базы: {item["answer"]}
Рекомендация по допродаже: {item["upsell"]}
""".strip()
        )

    return "\n\n".join(blocks)


async def call_ollama(message: str, knowledge):
    context = kb_to_context(knowledge)

    system_prompt = """
Ты — AI Copilot менеджера по продажам.

Твоя задача:
1. Подготовить короткий, естественный и вежливый ответ клиенту.
2. Отдельно дать менеджеру рекомендацию по уместной допродаже.

КРИТИЧЕСКИЕ ПРАВИЛА:

- Используй факты только из предоставленной базы знаний.
- Никогда не придумывай цены, скидки, сроки, гарантии и условия.
- Если информации недостаточно — задай клиенту уточняющий вопрос.
- Не дави на клиента.
- Допродажа должна быть связана с исходной потребностью.
- Не выдавай внутреннюю рекомендацию менеджеру клиенту.
- Пиши по-русски.
- Ответ должен быть компактным.

Верни ТОЛЬКО JSON:

{
  "client_reply": "...",
  "manager_upsell": "..."
}
""".strip()

    user_prompt = f"""
ОБРАЩЕНИЕ КЛИЕНТА:

{message}

БАЗА ЗНАНИЙ:

{context}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "options": {
            "temperature": 0.2
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload
        )

        response.raise_for_status()

        data = response.json()

    raw = data["message"]["content"].strip()

    parsed = json.loads(raw)

    reply = str(
        parsed.get("client_reply", "")
    ).strip()

    upsell = str(
        parsed.get("manager_upsell", "")
    ).strip()

    if not reply or not upsell:
        raise ValueError("AI returned incomplete JSON")

    return {
        "client_reply": reply,
        "manager_upsell": upsell
    }


def deterministic_fallback(knowledge):
    if knowledge:
        top = knowledge[0]

        return {
            "client_reply": top["answer"],
            "manager_upsell": top["upsell"]
        }

    return {
        "client_reply":
            "Спасибо за обращение. Чтобы дать точный ответ, уточните, пожалуйста, немного подробнее, какая услуга или вариант вас интересует.",

        "manager_upsell":
            "Сначала уточните потребность клиента. Не предлагайте дополнительную услугу, пока не станет понятно, какую задачу она должна решить."
    }


async def analyze_message(message: str):
    knowledge = search_knowledge(message)

    ai_mode = "ollama"

    try:
        result = await call_ollama(
            message,
            knowledge
        )

    except Exception as error:
        print(
            "Ollama fallback:",
            repr(error)
        )

        result = deterministic_fallback(
            knowledge
        )

        ai_mode = "fallback"

    used_titles = [
        item["title"]
        for item in knowledge
    ]

    conn = db()

    conn.execute(
        """
        INSERT INTO interactions
        (
            customer_message,
            client_reply,
            manager_upsell,
            used_kb,
            ai_mode
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            message,
            result["client_reply"],
            result["manager_upsell"],
            json.dumps(
                used_titles,
                ensure_ascii=False
            ),
            ai_mode
        )
    )

    conn.commit()
    conn.close()

    return {
        **result,
        "ai_mode": ai_mode,
        "knowledge": used_titles
    }


# ==================================================
# API
# ==================================================

@app.get("/api/status")
async def status():
    ollama = False

    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(
                f"{OLLAMA_URL}/api/tags"
            )

            ollama = response.status_code == 200

    except Exception:
        pass

    conn = db()

    kb_count = conn.execute(
        "SELECT COUNT(*) FROM knowledge WHERE active=1"
    ).fetchone()[0]

    interactions = conn.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]

    conn.close()

    return {
        "status": "ok",
        "database": "SQLite",
        "knowledge_items": kb_count,
        "interactions": interactions,
        "ollama": ollama,
        "model": OLLAMA_MODEL,
        "amocrm_signature_validation":
            bool(AMO_CHANNEL_SECRET)
    }


@app.post("/api/analyze")
async def analyze(body: AnalyzeRequest):
    return await analyze_message(
        body.message.strip()
    )


@app.get("/api/kb")
def get_kb():
    conn = db()

    rows = conn.execute(
        """
        SELECT *
        FROM knowledge
        WHERE active = 1
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


@app.post("/api/kb")
def add_kb(body: KBRequest):
    conn = db()

    cursor = conn.execute(
        """
        INSERT INTO knowledge
        (title, keywords, answer, upsell)
        VALUES (?, ?, ?, ?)
        """,
        (
            body.title.strip(),
            body.keywords.strip(),
            body.answer.strip(),
            body.upsell.strip()
        )
    )

    conn.commit()

    item_id = cursor.lastrowid

    conn.close()

    return {
        "ok": True,
        "id": item_id
    }


@app.put("/api/kb/{item_id}")
def update_kb(
    item_id: int,
    body: KBRequest
):
    conn = db()

    cursor = conn.execute(
        """
        UPDATE knowledge
        SET
            title = ?,
            keywords = ?,
            answer = ?,
            upsell = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        AND active = 1
        """,
        (
            body.title.strip(),
            body.keywords.strip(),
            body.answer.strip(),
            body.upsell.strip(),
            item_id
        )
    )

    conn.commit()

    changed = cursor.rowcount

    conn.close()

    if not changed:
        raise HTTPException(
            status_code=404,
            detail="Knowledge item not found"
        )

    return {
        "ok": True
    }


@app.delete("/api/kb/{item_id}")
def delete_kb(item_id: int):
    conn = db()

    conn.execute(
        """
        UPDATE knowledge
        SET active = 0,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (item_id,)
    )

    conn.commit()
    conn.close()

    return {
        "ok": True
    }


# Generic endpoint for CRM middleware / automation.
@app.post("/api/integration/message")
async def integration_message(
    body: AnalyzeRequest
):
    return await analyze_message(
        body.message.strip()
    )


# --------------------------------------------------
# amoCRM Chat API adapter
#
# This is intentionally isolated from the AI core.
# When AMO_CHANNEL_SECRET is configured, X-Signature
# verification becomes mandatory.
# --------------------------------------------------

@app.post("/api/amocrm/webhook/{scope_id}")
async def amocrm_webhook(
    scope_id: str,
    request: Request
):
    raw_body = await request.body()

    signature_checked = False

    if AMO_CHANNEL_SECRET:
        received_signature = request.headers.get(
            "X-Signature",
            ""
        )

        calculated_signature = hmac.new(
            AMO_CHANNEL_SECRET.encode(),
            raw_body,
            hashlib.sha1
        ).hexdigest()

        if not hmac.compare_digest(
            received_signature,
            calculated_signature
        ):
            raise HTTPException(
                status_code=403,
                detail="Invalid X-Signature"
            )

        signature_checked = True

    try:
        payload = json.loads(raw_body)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Expected JSON"
        )

    message = payload.get(
        "message",
        {}
    )

    message_data = message.get(
        "message",
        {}
    )

    text = message_data.get(
        "text",
        ""
    )

    if not text:
        return {
            "ok": True,
            "ignored": True,
            "scope_id": scope_id,
            "signature_checked": signature_checked
        }

    result = await analyze_message(text)

    return {
        "ok": True,
        "scope_id": scope_id,
        "signature_checked": signature_checked,
        "analysis": result
    }


# ==================================================
# UI
# ==================================================

HTML = r"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Sales Copilot</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at 20% 0%,
            #182554 0,
            transparent 30%
        ),
        #080b14;

    color: #eef2ff;
}

button,
input,
textarea {
    font: inherit;
}

header {
    padding: 20px 26px;
    border-bottom: 1px solid #20263b;
    background: rgba(8,11,20,.8);
    backdrop-filter: blur(18px);
    position: sticky;
    top: 0;
    z-index: 10;
}

.header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.brand {
    font-weight: 800;
    font-size: 21px;
}

.subtitle {
    color: #8f9ab9;
    margin-top: 5px;
    font-size: 13px;
}

.statuses {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.badge {
    padding: 7px 10px;
    border-radius: 999px;
    background: #151b2d;
    border: 1px solid #29324e;
    color: #aeb9d9;
    font-size: 12px;
}

.badge.good {
    color: #86efac;
    border-color: #1d6841;
}

.badge.warn {
    color: #fcd34d;
    border-color: #6a5520;
}

main {
    width: min(1450px, 96%);
    margin: 24px auto 60px;
}

.grid {
    display: grid;
    grid-template-columns:
        minmax(350px, .9fr)
        minmax(450px, 1.1fr);
    gap: 18px;
}

.card {
    background: rgba(16,20,34,.94);
    border: 1px solid #242c44;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,.18);
}

.card-head {
    padding: 16px 18px;
    border-bottom: 1px solid #242c44;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card-head h2 {
    font-size: 15px;
    margin: 0;
}

.muted {
    color: #8f9ab9;
    font-size: 12px;
}

.chat {
    height: 410px;
    overflow-y: auto;
    padding: 18px;
}

.message {
    margin-bottom: 13px;
    max-width: 88%;
}

.message.customer {
    margin-right: auto;
}

.message.manager {
    margin-left: auto;
}

.author {
    color: #818dab;
    font-size: 11px;
    margin-bottom: 5px;
}

.bubble {
    border-radius: 14px;
    padding: 11px 13px;
    background: #1b2236;
    border: 1px solid #29324c;
    line-height: 1.5;
    white-space: pre-wrap;
}

.manager .bubble {
    background: #172e28;
    border-color: #245446;
}

.composer {
    border-top: 1px solid #242c44;
    padding: 14px;
}

textarea,
input {
    width: 100%;
    border: 1px solid #303956;
    color: #edf2ff;
    background: #0e1322;
    border-radius: 10px;
    padding: 11px;
    outline: none;
}

textarea:focus,
input:focus {
    border-color: #6378ff;
}

textarea {
    resize: vertical;
    min-height: 82px;
}

.actions {
    margin-top: 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

button {
    border: 0;
    cursor: pointer;
    border-radius: 9px;
    padding: 9px 13px;
    background: #6378ff;
    color: white;
    font-weight: 700;
}

button.secondary {
    background: #1b2236;
    color: #cbd5ed;
    border: 1px solid #303956;
}

button.danger {
    background: #461e29;
    color: #fecdd3;
}

button:disabled {
    opacity: .55;
    cursor: progress;
}

.results {
    padding: 18px;
    display: grid;
    gap: 14px;
}

.result {
    padding: 17px;
    border: 1px solid #28314c;
    border-radius: 14px;
    background: #0f1423;
}

.result.client {
    border-left: 4px solid #6378ff;
}

.result.upsell {
    border-left: 4px solid #3fb981;
}

.result h3 {
    font-size: 13px;
    margin: 0 0 12px;
}

.result-text {
    white-space: pre-wrap;
    min-height: 70px;
    line-height: 1.6;
    color: #dde4f7;
}

.result-footer {
    margin-top: 13px;
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: center;
}

.kb {
    margin-top: 18px;
}

.kb-grid {
    display: grid;
    grid-template-columns:
        minmax(320px, .7fr)
        minmax(500px, 1.3fr);
}

.kb-form {
    padding: 18px;
    border-right: 1px solid #242c44;
}

.field {
    margin-bottom: 11px;
}

.field label {
    font-size: 12px;
    color: #9ca8c7;
    display: block;
    margin-bottom: 5px;
}

.kb-list {
    max-height: 520px;
    overflow-y: auto;
}

.kb-row {
    padding: 15px 18px;
    border-bottom: 1px solid #242c44;
}

.kb-row:last-child {
    border-bottom: 0;
}

.kb-title {
    font-weight: 750;
}

.kb-keywords {
    margin-top: 4px;
    color: #7e89a8;
    font-size: 12px;
}

.kb-answer,
.kb-upsell {
    margin-top: 8px;
    color: #c3cce3;
    line-height: 1.45;
    font-size: 13px;
}

.kb-buttons {
    margin-top: 10px;
    display: flex;
    gap: 7px;
}

.demo-warning {
    background: #292310;
    border: 1px solid #60531f;
    color: #fde68a;
    padding: 10px 13px;
    border-radius: 10px;
    margin-bottom: 15px;
    font-size: 12px;
}

code {
    color: #9eb0ff;
}

@media (max-width: 900px) {
    .grid,
    .kb-grid {
        grid-template-columns: 1fr;
    }

    .kb-form {
        border-right: 0;
        border-bottom: 1px solid #242c44;
    }

    .header-row {
        align-items: flex-start;
        flex-direction: column;
    }
}

</style>
</head>

<body>

<header>
<div class="header-row">

<div>
    <div class="brand">
        AI Sales Copilot
    </div>

    <div class="subtitle">
        CRM assistant • SQLite RAG • Local AI
    </div>
</div>

<div class="statuses">
    <span id="dbStatus" class="badge">
        SQLite…
    </span>

    <span id="aiStatus" class="badge">
        AI…
    </span>

    <span class="badge">
        amoCRM adapter
    </span>
</div>

</div>
</header>

<main>

<div class="grid">

<section class="card">

<div class="card-head">
    <h2>Диалог с клиентом</h2>
    <span class="muted">CRM prototype</span>
</div>

<div id="chat" class="chat">

<div class="message customer">
    <div class="author">Клиент</div>
    <div class="bubble">
Здравствуйте! Сколько занимает доставка и можно ли получить заказ быстрее?
    </div>
</div>

</div>

<div class="composer">

<textarea
    id="message"
    placeholder="Введите обращение клиента..."
>Здравствуйте! Сколько занимает доставка и можно ли получить заказ быстрее?</textarea>

<div class="actions">

<button
    id="analyzeButton"
    onclick="analyzeMessage()"
>
    Анализировать
</button>

<button
    class="secondary"
    onclick="exampleGuarantee()"
>
    Пример: гарантия
</button>

<button
    class="secondary"
    onclick="exampleUnknown()"
>
    Неизвестный вопрос
</button>

</div>

</div>

</section>


<section class="card">

<div class="card-head">
    <h2>AI Copilot</h2>
    <span id="mode" class="muted">
        ожидание
    </span>
</div>

<div class="results">

<div class="result client">

<h3>Ответ клиенту</h3>

<div
    id="clientReply"
    class="result-text"
>
После анализа здесь появится готовый текст менеджеру.
</div>

<div class="result-footer">

<span
    id="knowledgeUsed"
    class="muted"
></span>

<button
    class="secondary"
    onclick="copyText('clientReply')"
>
    Копировать
</button>

</div>

</div>


<div class="result upsell">

<h3>Подсказка по допродаже менеджеру</h3>

<div
    id="upsell"
    class="result-text"
>
AI подскажет уместную дополнительную услугу или предложит сначала уточнить потребность.
</div>

<div class="result-footer">

<span class="muted">
    Видит только менеджер
</span>

<button
    class="secondary"
    onclick="copyText('upsell')"
>
    Копировать
</button>

</div>

</div>

</div>

</section>

</div>


<section class="card kb">

<div class="card-head">
    <h2>База знаний</h2>
    <span class="muted">
        SQLite
    </span>
</div>

<div class="kb-grid">

<div class="kb-form">

<div class="demo-warning">
Данные ниже демонстрационные. На реальном проекте сюда загружаются реальные условия компании.
</div>

<input
    type="hidden"
    id="kbId"
/>

<div class="field">
<label>Название</label>
<input
    id="kbTitle"
    placeholder="Например: Доставка"
/>
</div>

<div class="field">
<label>Ключевые слова</label>
<input
    id="kbKeywords"
    placeholder="доставка, курьер, срок"
/>
</div>

<div class="field">
<label>Что отвечать клиенту</label>
<textarea
    id="kbAnswer"
    placeholder="Факты из базы знаний"
></textarea>
</div>

<div class="field">
<label>Подсказка по допродаже</label>
<textarea
    id="kbUpsell"
    placeholder="Что можно предложить менеджеру"
></textarea>
</div>

<div class="actions">

<button onclick="saveKnowledge()">
    Сохранить
</button>

<button
    class="secondary"
    onclick="resetKnowledgeForm()"
>
    Очистить
</button>

</div>

</div>

<div
    id="kbList"
    class="kb-list"
></div>

</div>

</section>

</main>


<script>

let knowledge = [];


async function loadStatus() {

    try {

        const response = await fetch('/api/status');
        const data = await response.json();

        const db = document.getElementById('dbStatus');
        const ai = document.getElementById('aiStatus');

        db.textContent =
            `SQLite: ${data.knowledge_items} записей`;

        db.className =
            'badge good';

        if (data.ollama) {

            ai.textContent =
                `AI: ${data.model}`;

            ai.className =
                'badge good';

        } else {

            ai.textContent =
                'AI: fallback';

            ai.className =
                'badge warn';
        }

    } catch (error) {

        console.error(error);
    }
}


function addChatMessage(
    author,
    text,
    manager = false
) {

    const chat =
        document.getElementById('chat');

    const root =
        document.createElement('div');

    root.className =
        'message ' +
        (manager ? 'manager' : 'customer');

    const authorNode =
        document.createElement('div');

    authorNode.className =
        'author';

    authorNode.textContent =
        author;

    const bubble =
        document.createElement('div');

    bubble.className =
        'bubble';

    bubble.textContent =
        text;

    root.appendChild(authorNode);
    root.appendChild(bubble);

    chat.appendChild(root);

    chat.scrollTop =
        chat.scrollHeight;
}


async function analyzeMessage() {

    const textarea =
        document.getElementById('message');

    const button =
        document.getElementById('analyzeButton');

    const message =
        textarea.value.trim();

    if (!message) {
        return;
    }

    button.disabled = true;
    button.textContent = 'Анализ…';

    addChatMessage(
        'Клиент',
        message
    );

    document.getElementById(
        'clientReply'
    ).textContent = 'Генерация ответа…';

    document.getElementById(
        'upsell'
    ).textContent = 'Анализ возможной допродажи…';

    try {

        const response =
            await fetch(
                '/api/analyze',
                {
                    method: 'POST',
                    headers: {
                        'Content-Type':
                            'application/json'
                    },
                    body: JSON.stringify({
                        message
                    })
                }
            );

        if (!response.ok) {
            throw new Error(
                'HTTP ' + response.status
            );
        }

        const data =
            await response.json();

        document.getElementById(
            'clientReply'
        ).textContent =
            data.client_reply;

        document.getElementById(
            'upsell'
        ).textContent =
            data.manager_upsell;

        document.getElementById(
            'mode'
        ).textContent =
            data.ai_mode === 'ollama'
            ? 'AI generation'
            : 'fallback mode';

        document.getElementById(
            'knowledgeUsed'
        ).textContent =
            data.knowledge.length
            ? 'База: ' + data.knowledge.join(', ')
            : 'Совпадений в базе нет';

        addChatMessage(
            'AI Copilot',
            'Черновик ответа и рекомендация подготовлены.',
            true
        );

        loadStatus();

    } catch (error) {

        document.getElementById(
            'clientReply'
        ).textContent =
            'Ошибка: ' + error.message;

        document.getElementById(
            'upsell'
        ).textContent =
            'Проверьте backend.';

    } finally {

        button.disabled = false;
        button.textContent =
            'Анализировать';
    }
}


async function loadKnowledge() {

    const response =
        await fetch('/api/kb');

    knowledge =
        await response.json();

    const list =
        document.getElementById(
            'kbList'
        );

    list.replaceChildren();

    knowledge.forEach(item => {

        const row =
            document.createElement('div');

        row.className =
            'kb-row';

        const title =
            document.createElement('div');

        title.className =
            'kb-title';

        title.textContent =
            item.title;

        const keywords =
            document.createElement('div');

        keywords.className =
            'kb-keywords';

        keywords.textContent =
            item.keywords;

        const answer =
            document.createElement('div');

        answer.className =
            'kb-answer';

        answer.textContent =
            'Ответ: ' + item.answer;

        const upsell =
            document.createElement('div');

        upsell.className =
            'kb-upsell';

        upsell.textContent =
            'Допродажа: ' + item.upsell;

        const buttons =
            document.createElement('div');

        buttons.className =
            'kb-buttons';

        const edit =
            document.createElement('button');

        edit.className =
            'secondary';

        edit.textContent =
            'Изменить';

        edit.onclick =
            () => editKnowledge(item.id);

        const remove =
            document.createElement('button');

        remove.className =
            'danger';

        remove.textContent =
            'Удалить';

        remove.onclick =
            () => deleteKnowledge(item.id);

        buttons.appendChild(edit);
        buttons.appendChild(remove);

        row.appendChild(title);
        row.appendChild(keywords);
        row.appendChild(answer);
        row.appendChild(upsell);
        row.appendChild(buttons);

        list.appendChild(row);
    });
}


function editKnowledge(id) {

    const item =
        knowledge.find(
            x => x.id === id
        );

    if (!item) return;

    document.getElementById(
        'kbId'
    ).value =
        item.id;

    document.getElementById(
        'kbTitle'
    ).value =
        item.title;

    document.getElementById(
        'kbKeywords'
    ).value =
        item.keywords;

    document.getElementById(
        'kbAnswer'
    ).value =
        item.answer;

    document.getElementById(
        'kbUpsell'
    ).value =
        item.upsell;
}


function resetKnowledgeForm() {

    [
        'kbId',
        'kbTitle',
        'kbKeywords',
        'kbAnswer',
        'kbUpsell'
    ].forEach(id => {

        document.getElementById(
            id
        ).value = '';
    });
}


async function saveKnowledge() {

    const id =
        document.getElementById(
            'kbId'
        ).value;

    const body = {

        title:
            document.getElementById(
                'kbTitle'
            ).value.trim(),

        keywords:
            document.getElementById(
                'kbKeywords'
            ).value.trim(),

        answer:
            document.getElementById(
                'kbAnswer'
            ).value.trim(),

        upsell:
            document.getElementById(
                'kbUpsell'
            ).value.trim()
    };

    if (
        !body.title ||
        !body.keywords ||
        !body.answer ||
        !body.upsell
    ) {

        alert(
            'Заполните все поля'
        );

        return;
    }

    const url =
        id
        ? '/api/kb/' + id
        : '/api/kb';

    const method =
        id
        ? 'PUT'
        : 'POST';

    const response =
        await fetch(
            url,
            {
                method,
                headers: {
                    'Content-Type':
                        'application/json'
                },
                body:
                    JSON.stringify(body)
            }
        );

    if (!response.ok) {

        alert(
            'Не удалось сохранить запись'
        );

        return;
    }

    resetKnowledgeForm();
    await loadKnowledge();
    await loadStatus();
}


async function deleteKnowledge(id) {

    if (
        !confirm(
            'Удалить запись из базы знаний?'
        )
    ) {
        return;
    }

    await fetch(
        '/api/kb/' + id,
        {
            method: 'DELETE'
        }
    );

    await loadKnowledge();
    await loadStatus();
}


async function copyText(id) {

    const text =
        document.getElementById(
            id
        ).textContent;

    await navigator.clipboard.writeText(
        text
    );
}


function exampleGuarantee() {

    document.getElementById(
        'message'
    ).value =
        'А если после покупки что-нибудь сломается, какая у вас гарантия?';
}


function exampleUnknown() {

    document.getElementById(
        'message'
    ).value =
        'А у вас есть филиал в Казани и какая там сейчас скидка?';
}


loadStatus();
loadKnowledge();

</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8787
    )
