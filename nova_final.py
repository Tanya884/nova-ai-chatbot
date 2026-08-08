import streamlit as st
import ollama
import io
import json
import os
import sqlite3
import hashlib
import re
import html
import threading
import uuid
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Nova",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEMPERATURE = 0.5
WINDOW_MEMORY = 20


def get_secret(name):
    """Read a deployment secret without ever putting it in source code."""
    key = os.getenv(name, "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
CLOUD_MODE = bool(GEMINI_API_KEY or OPENROUTER_API_KEY)

# A public Streamlit deployment must not share one SQLite file between users.
# Cloud chats are isolated per browser session; local Ollama keeps the existing
# persistent chat_history.db behaviour.
if CLOUD_MODE:
    if "nova_session_id" not in st.session_state:
        st.session_state.nova_session_id = uuid.uuid4().hex
    DB_FILE = os.path.join(
        "/tmp", f"nova_chat_{st.session_state.nova_session_id}.db"
    )
else:
    DB_FILE = "chat_history.db"

# =========================================================
# CHATGPT-LIKE CSS
# =========================================================
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
    font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif !important;
}

* {
    box-shadow:none !important;
}

[data-testid="stAppViewContainer"] {
    background:#ffffff;
}

/* Keep the existing conversation fully visible while Streamlit reruns. */
[data-stale="true"],
.stale {
    opacity:1 !important;
    filter:none !important;
    transition:none !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"] {
    filter:none !important;
}

[data-testid="stSidebar"] {
    background:#f7f7f8;
    border-right:1px solid #e5e5e5;
    overflow-x:hidden !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top:1rem;
    overflow-x:hidden !important;
}

/* Prevent an unavailable Material Symbols font from printing
   "keyboard_double_arrow..." across the top of the sidebar. */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="collapsedControl"] span,
button[kind="headerNoPadding"] span {
    font-size:0 !important;
    overflow:hidden !important;
    width:18px !important;
}

[data-testid="stSidebarCollapseButton"] span::after,
[data-testid="collapsedControl"] span::after,
button[kind="headerNoPadding"] span::after {
    content:"‹";
    font-family:Arial, sans-serif !important;
    font-size:25px !important;
    line-height:1 !important;
}

.block-container {
    max-width:1000px;
    padding-top:2rem;
    padding-bottom:7rem;
}

[data-testid="stChatMessage"] {
    border:none;
    padding-top:1rem;
    padding-bottom:1rem;
}

/* Streamlit sometimes exposes unloaded Material icon names such as
   "face" and "smart_toy" as text. Nova does not need chat avatars. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"],
[data-testid="stChatMessage"] [data-testid*="Avatar"] {
    display:none !important;
}

[data-testid="stChatMessageContent"] {
    width:100% !important;
    max-width:100% !important;
}

[data-testid="stChatInput"] textarea {
    border-radius:24px !important;
    border:1px solid #d9d9e3 !important;
    background:#fff !important;
    padding-left:18px !important;
}

.st-key-bottom_composer {
    width:min(1050px,92vw) !important;
    margin-left:auto !important;
    margin-right:auto !important;
    background:#ffffff !important;
}

.st-key-bottom_composer [data-testid="stHorizontalBlock"] {
    align-items:flex-end !important;
}

.st-key-bottom_composer [data-baseweb="select"] > div {
    border:none !important;
    background:transparent !important;
    box-shadow:none !important;
    font-size:13px !important;
}

@media (max-width:900px) {
    .st-key-bottom_composer {
        width:96vw !important;
    }
}

.chat-title {
    font-size:26px;
    font-weight:700;
    text-align:center;
    margin-top:90px;
}

.chat-subtitle {
    text-align:center;
    color:#777;
}

.file-chip {
    display:block;
    padding:7px 9px;
    margin:4px 0;
    background:#eeeeee;
    border-radius:8px;
    font-size:12px;
    overflow:hidden;
    white-space:nowrap;
    text-overflow:ellipsis;
}

.small-muted {
    color:#777;
    font-size:12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SQLITE MEMORY
# =========================================================
def db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def create_chat(title="New chat"):
    conn = db()
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO chats(title, created_at, updated_at) VALUES(?,?,?)",
        (title, now, now),
    )
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()
    return chat_id


def update_chat_title(chat_id, title):
    conn = db()
    conn.execute(
        "UPDATE chats SET title=?, updated_at=? WHERE id=?",
        (title[:60], datetime.now().isoformat(timespec="seconds"), chat_id),
    )
    conn.commit()
    conn.close()


def save_message(chat_id, role, content):
    conn = db()
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO messages(chat_id, role, content, created_at) VALUES(?,?,?,?)",
        (chat_id, role, content, now),
    )
    conn.execute(
        "UPDATE chats SET updated_at=? WHERE id=?",
        (now, chat_id),
    )
    conn.commit()
    conn.close()


def get_chats():
    conn = db()
    rows = conn.execute(
        "SELECT id,title FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return rows


def get_messages(chat_id):
    conn = db()
    rows = conn.execute(
        "SELECT role,content FROM messages WHERE chat_id=? ORDER BY id",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]


def delete_chat(chat_id):
    conn = db()
    conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()


# =========================================================
# SESSION STATE
# =========================================================
if "chat_id" not in st.session_state:
    chats = get_chats()
    st.session_state.chat_id = chats[0][0] if chats else create_chat()

if "messages" not in st.session_state:
    st.session_state.messages = get_messages(st.session_state.chat_id)

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "file_signature" not in st.session_state:
    st.session_state.file_signature = ""

if "generated_pdf" not in st.session_state:
    st.session_state.generated_pdf = None

if "generated_pdf_name" not in st.session_state:
    st.session_state.generated_pdf_name = "Nova_Chat.pdf"

if "generated_pdf_chat_id" not in st.session_state:
    st.session_state.generated_pdf_chat_id = None


# =========================================================
# PDF EXPORT
# =========================================================
def get_pdf_request_mode(query):
    """Return 'content', 'export', or None for PDF-related requests."""
    q = query.lower()

    # A follow-up asking for the missing link should export the conversation.
    if re.search(r"\b(download\s*(link|button)|where.*download)\b", q):
        return "export"

    if "pdf" not in q:
        return None

    content_terms = [
        "question", "questions", "answer", "answers", "q&a", "qna",
        "interview", "job description", " jd ", "resume", "report",
        "guide", "notes", "summary", "relevant", "sare", "saare",
    ]
    request_terms = [
        "make", "create", "generate", "prepare", "want", "need",
        "chahiye", "bana", "banao", "banado", "de do", "dede",
    ]

    if any(term in f" {q} " for term in content_terms) and any(
        term in q for term in request_terms
    ):
        return "content"

    export_terms = [
        "make", "create", "generate", "prepare", "export", "convert",
        "download", "bana", "banao", "banado", "chahiye",
    ]
    if any(term in q for term in export_terms):
        return "export"

    return None


def wants_pdf_generation(query):
    return get_pdf_request_mode(query) is not None


def build_chat_pdf(messages, title="Nova - Chat Export"):
    # ReportLab is imported only when a PDF is requested, keeping startup fast.
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError as exc:
        raise RuntimeError(
            "PDF support is not installed. Run: pip install reportlab"
        ) from exc

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Nova",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChatTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#202123"),
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )
    user_style = ParagraphStyle(
        "UserLabel",
        parent=styles["Heading3"],
        textColor=colors.HexColor("#C2414D"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    assistant_style = ParagraphStyle(
        "AssistantLabel",
        parent=styles["Heading3"],
        textColor=colors.HexColor("#C88912"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        "ChatBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#303030"),
        spaceAfter=2.5 * mm,
    )

    story = [
        Paragraph(html.escape(title), title_style),
        Paragraph(
            "Generated on " + datetime.now().strftime("%d %B %Y, %I:%M %p"),
            styles["Normal"],
        ),
        Spacer(1, 5 * mm),
    ]

    for message in messages:
        role = message.get("role", "assistant")
        label = "You" if role == "user" else "Nova"
        label_style = user_style if role == "user" else assistant_style
        story.append(Paragraph(label, label_style))

        content = str(message.get("content", "")).strip()
        blocks = re.split(r"\n\s*\n", content) if content else [""]
        for block in blocks:
            safe = html.escape(block).replace("\n", "<br/>")
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe or " ", body_style))

        story.append(Spacer(1, 2 * mm))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawCentredString(
            A4[0] / 2,
            9 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    output.seek(0)
    return output.getvalue()


# =========================================================
# FILE READERS
# =========================================================
def read_pdf(file):
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file.getvalue()))
        return "\n".join(
            f"--- PDF Page {i+1} ---\n{page.extract_text() or ''}"
            for i, page in enumerate(reader.pages)
        )
    except Exception as e:
        return f"Could not read PDF: {e}"


def read_excel(file):
    try:
        import pandas as pd

        raw = file.getvalue()
        book = pd.ExcelFile(io.BytesIO(raw))
        parts = []
        for sheet in book.sheet_names:
            df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet)
            parts.append(f"--- Excel Sheet: {sheet} ---")
            parts.append(df.to_string(index=False))
        return "\n".join(parts)
    except Exception as e:
        return f"Could not read Excel: {e}"


def read_csv(file):
    try:
        import pandas as pd

        raw = file.getvalue()
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                return pd.read_csv(
                    io.BytesIO(raw), encoding=enc
                ).to_string(index=False)
            except Exception:
                pass
        return "Could not decode CSV."
    except Exception as e:
        return f"Could not read CSV: {e}"


def read_docx(file):
    try:
        from docx import Document

        doc = Document(io.BytesIO(file.getvalue()))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    except Exception as e:
        return f"Could not read DOCX: {e}"


def read_text(file):
    raw = file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return "Could not decode text."


def read_json(file):
    try:
        return json.dumps(
            json.loads(file.getvalue().decode("utf-8")),
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Could not read JSON: {e}"


def read_binary(file):
    try:
        return file.getvalue().decode("utf-8")
    except Exception:
        return (
            "Binary file uploaded. Raw binary content cannot be directly "
            "read as text by this version."
        )


def extract_file(file):
    ext = os.path.splitext(file.name.lower())[1]

    if ext == ".pdf":
        return read_pdf(file)
    if ext in [".xlsx", ".xls", ".xlsm"]:
        return read_excel(file)
    if ext == ".csv":
        return read_csv(file)
    if ext == ".docx":
        return read_docx(file)
    if ext == ".json":
        return read_json(file)
    if ext in [
        ".txt", ".md", ".py", ".sql", ".sas", ".r",
        ".html", ".css", ".js", ".xml", ".yaml",
        ".yml", ".log"
    ]:
        return read_text(file)

    return read_binary(file)


def make_signature(files):
    return "|".join(
        f"{f.name}:{len(f.getvalue())}:"
        f"{hashlib.md5(f.getvalue()).hexdigest()}"
        for f in files
    )


def build_file_context(files):
    parts = []

    for f in files:
        content = extract_file(f)

        # Avoid sending enormous files directly into the model context.
        if len(content) > 60000:
            content = (
                content[:60000]
                + "\n[File content truncated for model context.]"
            )

        parts.append(
            f"================ FILE: {f.name} ================\n"
            f"{content}"
        )

    return "\n\n".join(parts)


def should_use_file_context(query):
    """Attach uploaded content only when the question refers to that content."""
    q = query.lower()
    file_terms = [
        "file", "uploaded", "upload", "attached", "attachment",
        "pdf", "excel", "csv", "dataset", "document", "docx",
        "job description", " jd ", "this data", "this sheet",
        "this document", "isme", "iske", "iss file", "is file",
    ]
    padded = f" {q} "
    return any(term in padded for term in file_terms)


# =========================================================
# FAST WEB ROUTER + DOWNLOAD TOOL
# =========================================================
def needs_web_search(query):
    q = query.lower()

    # Any year after the model's knowledge cutoff should automatically use
    # live search (for example: 2024, 2025, 2026...).
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
    if any(year >= 2024 for year in years):
        return True

    keywords = [
        "latest", "today", "current", "now", "news", "recent",
        "this year", "this month", "this week", "yesterday", "tomorrow",
        "price", "weather", "stock", "share price", "score", "result",
        "schedule", "election", "president", "prime minister", "ceo",
        "salary", "package", "hike", "vacancy", "job opening", "release",
        "version", "law", "rule", "policy", "rate", "exchange rate",
        "market", "search internet", "search web", "on internet",
        "google", "online", "website", "download", "csv", "xlsx",
        "excel", "dataset", "data file"
    ]
    return any(x in q for x in keywords)


def wants_download(query):
    q = query.lower()
    return any(x in q for x in [
        "download", "save the file", "get the file", "fetch the file",
        "csv", "xlsx", "excel file", "dataset"
    ])


def extract_urls(text):
    return re.findall(r'https?://[^\s<>"\']+', text or "")


def web_search(query):
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=5)
        return "\n\n".join(
            f"Title: {r.get('title','')}\n"
            f"URL: {r.get('href','')}\n"
            f"Summary: {r.get('body','')}"
            for r in results
        ) or "No web results found."
    except Exception as e:
        return f"Web search unavailable: {e}"


def wants_images(query):
    """Recognize direct requests for pictures in English or Hinglish."""
    q = f" {query.lower()} "
    image_terms = [
        " image ", " images ", " pic ", " pics ", " picture ",
        " pictures ", " photo ", " photos ", " wallpaper ",
        " tasveer ", " tasveeren ", " photo dikhao ", " pics dikhao ",
    ]
    return any(term in q for term in image_terms)


def image_search(query, max_results=6):
    """Return real image results instead of asking the LLM to imagine them."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    cleaned_query = re.sub(
        r"\b(images?|pics?|pictures?|photos?|wallpapers?|dikhao|show me)\b",
        " ",
        query,
        flags=re.I,
    )
    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip() or query

    results = DDGS().images(
        cleaned_query,
        region="wt-wt",
        safesearch="moderate",
        max_results=max_results,
    )
    images = []
    for item in results or []:
        image_url = item.get("image")
        if not image_url:
            continue
        images.append({
            "image": image_url,
            "thumbnail": item.get("thumbnail") or image_url,
            "title": item.get("title") or cleaned_query.title(),
            "source": item.get("url") or "",
        })
    return images


IMAGE_RESULT_PREFIX = "[[NOVA_IMAGE_RESULTS]]"


def pack_image_results(query, images):
    return IMAGE_RESULT_PREFIX + json.dumps(
        {"query": query, "images": images}, ensure_ascii=False
    )


def render_chat_content(content):
    """Render normal messages or a persistent grid of image-search results."""
    if not content.startswith(IMAGE_RESULT_PREFIX):
        st.markdown(content)
        return

    try:
        payload = json.loads(content[len(IMAGE_RESULT_PREFIX):])
        images = payload.get("images", [])
        st.markdown(f"Here are pictures for **{payload.get('query', 'your search')}**:")
        for start in range(0, len(images), 3):
            columns = st.columns(3)
            for column, item in zip(columns, images[start:start + 3]):
                with column:
                    st.image(
                        item.get("thumbnail") or item["image"],
                        use_container_width=True,
                    )
                    title = item.get("title", "View image")
                    source = item.get("source", "")
                    if source:
                        st.markdown(f"[{title}]({source})")
                    else:
                        st.caption(title)
    except Exception:
        st.error("These image results could not be displayed.")


def download_file(url, folder="downloads"):
    import requests

    os.makedirs(folder, exist_ok=True)
    url = url.rstrip(".,);]}>")
    name = os.path.basename(urlparse(url).path) or "downloaded_file"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)

    try:
        response = requests.get(
            url,
            timeout=25,
            allow_redirects=True,
            headers={"User-Agent": "Nova/1.0"}
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "." not in name:
            if "csv" in content_type:
                name += ".csv"
            elif "spreadsheet" in content_type or "excel" in content_type:
                name += ".xlsx"
            elif "json" in content_type:
                name += ".json"

        path = os.path.join(folder, name)

        with open(path, "wb") as f:
            f.write(response.content)

        return path, f"Downloaded successfully: {path}"

    except Exception as e:
        return None, f"Download failed: {e}"


def try_download_from_results(search_text):
    urls = extract_urls(search_text)

    preferred = [
        u for u in urls
        if re.search(r'\.(csv|xlsx|xls|json)(\?|$)', u, re.I)
    ]

    candidates = preferred + [u for u in urls if u not in preferred]

    for url in candidates[:8]:
        path, status = download_file(url)
        if path:
            return path, status

    return None, "No directly downloadable public data file was found."


# =========================================================
# HYBRID AI BACKEND: GEMINI CLOUD + OPTIONAL OPENROUTER + LOCAL OLLAMA
# =========================================================
OPENROUTER_MODEL_CANDIDATES = {
    "llama3.2": [
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "openrouter/auto",
    ],
    "deepseek-r1:latest": [
        "~deepseek/deepseek-v4-flash-latest",
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-r1",
        "openrouter/auto",
    ],
    "qwen2.5:latest": [
        "qwen/qwen3.7-flash",
        "qwen/qwen3-30b-a3b-instruct-2507",
        "qwen/qwen-2.5-72b-instruct",
        "openrouter/auto",
    ],
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_openrouter_model_ids():
    """Cache model availability so changed/deprecated IDs do not break Nova."""
    import requests

    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        timeout=12,
    )
    response.raise_for_status()
    return {item["id"] for item in response.json().get("data", [])}


def resolve_openrouter_model(local_model):
    candidates = OPENROUTER_MODEL_CANDIDATES.get(
        local_model, ["openrouter/auto"]
    )
    try:
        available = get_openrouter_model_ids()
        return next((name for name in candidates if name in available), candidates[0])
    except Exception:
        return candidates[0]


def stream_openrouter_chat(local_model, messages, max_tokens):
    """Yield OpenRouter text deltas in the same shape used by the UI."""
    import requests

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "X-Title": "Nova",
        },
        json={
            "model": resolve_openrouter_model(local_model),
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=(15, 180),
        stream=True,
    )
    response.raise_for_status()

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        payload = raw_line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
            piece = data.get("choices", [{}])[0].get("delta", {}).get(
                "content", ""
            )
            if piece:
                yield {"message": {"content": piece}}
        except (json.JSONDecodeError, IndexError, TypeError):
            continue


def stream_gemini_chat(model_name, messages, max_tokens):
    """Stream Gemini output using REST, so no extra Google SDK is required."""
    import requests

    system_text = "\n\n".join(
        message.get("content", "")
        for message in messages
        if message.get("role") == "system"
    )
    contents = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": message.get("content", "")}],
        })

    # Google periodically changes which models receive free-tier quota. Try
    # current models first and retain 2.5 as compatibility fallbacks.
    model_candidates = list(dict.fromkeys([
        model_name,
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]))
    response = None
    errors = []
    for candidate in model_candidates:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{candidate}:streamGenerateContent?alt=sse",
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "systemInstruction": {"parts": [{"text": system_text}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": TEMPERATURE,
                    "maxOutputTokens": max_tokens,
                },
            },
            timeout=(15, 180),
            stream=True,
        )
        if response.ok:
            break
        errors.append(f"{candidate}: HTTP {response.status_code}")
        # Authentication/permission errors cannot be fixed by another model.
        if response.status_code in (401, 403):
            break

    if response is None or not response.ok:
        detail = response.text[:350] if response is not None else "No response"
        raise RuntimeError(
            "Gemini request failed. " + "; ".join(errors) + f". {detail}"
        )

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        try:
            data = json.loads(raw_line[5:].strip())
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            piece = "".join(
                part.get("text", "") for part in parts if "text" in part
            )
            if piece:
                yield {"message": {"content": piece}}
        except (json.JSONDecodeError, IndexError, TypeError):
            continue


def stream_nova_chat(local_model, messages, model_options):
    if GEMINI_API_KEY:
        max_tokens = model_options.get("num_predict", 1100)
        return stream_gemini_chat(local_model, messages, max_tokens)

    if OPENROUTER_API_KEY:
        max_tokens = model_options.get("num_predict", 1100)
        return stream_openrouter_chat(local_model, messages, max_tokens)

    return ollama.chat(
        model=local_model,
        messages=messages,
        stream=True,
        keep_alive="30m",
        options=model_options,
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ✦ Nova")
    st.caption("Your AI workspace")

    if st.button("＋ New chat", use_container_width=True):
        new_id = create_chat()
        st.session_state.chat_id = new_id
        st.session_state.messages = []
        st.session_state.file_context = ""
        st.session_state.uploaded_files = []
        st.session_state.file_signature = ""
        st.session_state.generated_pdf = None
        st.session_state.generated_pdf_chat_id = None
        st.rerun()

    st.divider()
    st.markdown("### 💬 Chats")

    chats = get_chats()

    for cid, title in chats:
        col1, col2 = st.columns([5, 1])

        with col1:
            label = title if title else "New chat"
            if st.button(
                label[:38],
                key=f"chat_{cid}",
                use_container_width=True
            ):
                st.session_state.chat_id = cid
                st.session_state.messages = get_messages(cid)
                st.session_state.file_context = ""
                st.session_state.uploaded_files = []
                st.session_state.file_signature = ""
                st.session_state.generated_pdf = None
                st.session_state.generated_pdf_chat_id = None
                st.rerun()

        with col2:
            if st.button("⋮", key=f"del_{cid}"):
                delete_chat(cid)

                if cid == st.session_state.chat_id:
                    new_id = create_chat()
                    st.session_state.chat_id = new_id
                    st.session_state.messages = []

                st.rerun()

    use_internet = st.toggle("🌐 Internet", value=True)


# =========================================================
# MAIN CHAT
# =========================================================
if not st.session_state.messages:
    st.markdown("""
    <style>
    .chat-title {
        margin-top:105px !important;
        font-size:36px !important;
        font-weight:400 !important;
    }
    .chat-subtitle {display:none !important;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-title">What should we work on?</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.uploaded_files:
        st.info(
            "📂 Ready: "
            + ", ".join(st.session_state.uploaded_files)
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_chat_content(msg["content"])

if (
    st.session_state.generated_pdf
    and st.session_state.generated_pdf_chat_id == st.session_state.chat_id
):
    st.download_button(
        "⬇️ Download PDF",
        data=st.session_state.generated_pdf,
        file_name=st.session_state.generated_pdf_name,
        mime="application/pdf",
        use_container_width=False,
        key=f"pdf_download_{st.session_state.chat_id}",
    )


# ChatGPT-like bottom controls: attachment plus model selector.
if GEMINI_API_KEY:
    MODEL_LABELS = {
        "gemini-3.6-flash": "Gemini 3.6 Flash",
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
        "gemini-2.5-flash": "Gemini 2.5 Flash",
    }
else:
    MODEL_LABELS = {
        "llama3.2": "Llama 3.2  Fast",
        "deepseek-r1:latest": "DeepSeek R1  Reasoning",
        "qwen2.5:latest": "Qwen 2.5  Balanced",
    }

def render_composer(container_key):
    """Render one native composer without any fixed-position CSS hacks."""
    with st.container(key=container_key):
        input_col, model_col = st.columns(
            [7.2, 2.2], vertical_alignment="bottom"
        )
        with input_col:
            submission = st.chat_input(
                "Message Nova...",
                accept_file="multiple",
                file_type=None,
                submit_mode="disable",
                key="nova_chat_input",
            )
        with model_col:
            selected_model = st.selectbox(
                "Model",
                list(MODEL_LABELS),
                format_func=lambda value: MODEL_LABELS[value],
                label_visibility="collapsed",
                key="selected_model",
            )
    return submission, selected_model


# One permanent composer for every state. New chat and existing chats use the
# exact same bottom position, matching ChatGPT behaviour.
with st.bottom:
    submission, model = render_composer("bottom_composer")


def warm_ollama_model(model_name):
    """Load the selected model in the background without blocking the UI."""
    try:
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            keep_alive="30m",
            options={"num_predict": 1, "num_ctx": 2048},
        )
    except Exception:
        pass


if not CLOUD_MODE and st.session_state.get("warming_model") != model:
    st.session_state.warming_model = model
    threading.Thread(
        target=warm_ollama_model,
        args=(model,),
        daemon=True,
    ).start()


prompt = None
if submission:
    prompt = (submission.text or "").strip()
    submitted_files = list(submission.files or [])

    if submitted_files:
        sig = make_signature(submitted_files)
        if sig != st.session_state.file_signature:
            with st.spinner("Reading files..."):
                st.session_state.file_context = build_file_context(
                    submitted_files
                )
            st.session_state.file_signature = sig
            st.session_state.uploaded_files = [
                file.name for file in submitted_files
            ]

        if not prompt:
            prompt = "Analyze the uploaded file(s)."


if prompt:
    pdf_mode = get_pdf_request_mode(prompt)
    pdf_requested = pdf_mode is not None
    # Save user message immediately.
    save_message(st.session_state.chat_id, "user", prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # First user message becomes the sidebar chat title.
    user_messages = [
        m["content"]
        for m in st.session_state.messages
        if m["role"] == "user"
    ]

    if len(user_messages) == 1:
        title = prompt.strip().replace("\n", " ")
        update_chat_title(
            st.session_state.chat_id,
            title[:55] if title else "New chat"
        )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Image requests are a tool action, not a question for the language model.
    # Search and render actual pictures immediately and persist them in history.
    if wants_images(prompt):
        with st.chat_message("assistant"):
            if not use_internet:
                answer = "Please turn on the Internet toggle to search for pictures."
                st.info(answer)
            else:
                try:
                    with st.spinner("Finding pictures..."):
                        found_images = image_search(prompt)
                    if found_images:
                        answer = pack_image_results(prompt, found_images)
                        render_chat_content(answer)
                    else:
                        answer = "I couldn't find usable image results for that search."
                        st.warning(answer)
                except Exception as e:
                    answer = f"Image search is unavailable right now: {e}"
                    st.error(answer)

        save_message(st.session_state.chat_id, "assistant", answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    # Common greetings should respond instantly without waking the full LLM.
    simple_prompt = re.sub(r"[^a-z ]", "", prompt.lower()).strip()
    instant_replies = {
        "hi": "Hi! How can I help you?",
        "hello": "Hello! How can I help you?",
        "hey": "Hey! How can I help you?",
        "hi nova": "Hi! How can I help you?",
        "hello nova": "Hello! How can I help you?",
        "hey nova": "Hey! How can I help you?",
    }
    if simple_prompt in instant_replies:
        answer = instant_replies[simple_prompt]
        with st.chat_message("assistant"):
            st.markdown(answer)
        save_message(st.session_state.chat_id, "assistant", answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    date_prompts = {
        "what is today date", "what is todays date", "today date",
        "aaj ki date", "aaj date kya hai",
    }
    if simple_prompt in date_prompts:
        answer = datetime.now().strftime("Today is %d %B %Y.")
        with st.chat_message("assistant"):
            st.markdown(answer)
        save_message(st.session_state.chat_id, "assistant", answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    # PDF requests are handled directly by the app. Do not ask the LLM to
    # rewrite the document inside the chat before creating the file.
    if pdf_mode == "export":
        with st.chat_message("assistant"):
            try:
                source_messages = st.session_state.messages[:-1]

                if not source_messages:
                    raise ValueError(
                        "There is no previous conversation to export yet."
                    )

                pdf_bytes = build_chat_pdf(
                    source_messages,
                    title="Nova - Questions and Answers",
                )
                st.session_state.generated_pdf = pdf_bytes
                st.session_state.generated_pdf_name = (
                    "Nova_Questions_and_Answers.pdf"
                )
                st.session_state.generated_pdf_chat_id = (
                    st.session_state.chat_id
                )

                ready_message = "✅ Your PDF is ready."
                st.success(ready_message)
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=st.session_state.generated_pdf_name,
                    mime="application/pdf",
                    key=f"new_pdf_{st.session_state.chat_id}",
                )
                save_message(
                    st.session_state.chat_id,
                    "assistant",
                    ready_message,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": ready_message}
                )
            except Exception as pdf_error:
                error_message = f"❌ Could not create PDF: {pdf_error}"
                st.error(error_message)
                save_message(
                    st.session_state.chat_id,
                    "assistant",
                    error_message,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

        st.rerun()

    system_prompt = """
You are Nova, a fast and helpful AI assistant.

The application's current date is {current_date}.

Rules:
- Be accurate, practical and concise.
- Remember the conversation context provided below.
- If a file is uploaded, use the file as the primary source.
- Never invent values that are not present in the uploaded file.
- For Excel/CSV questions, reason from the supplied data.
- For SQL/Python questions, provide correct practical code.
- If web results are supplied, use them for current information.
- WEB SEARCH RESULTS are fresh external evidence and override older model
  knowledge whenever they conflict.
- When WEB SEARCH RESULTS are present, answer the question directly from them.
  Never mention a knowledge-cutoff date and never say that your knowledge only
  goes up to 2023.
- If fresh results are insufficient, say only what could not be verified; do
  not present the model cutoff as a weakness.
- Do not claim to have browsed unless WEB SEARCH RESULTS are actually supplied.
- If DOWNLOADED FILE CONTENT is supplied, the application has already downloaded and read the file. Analyze it directly.
- Do not claim that you cannot download files when DOWNLOAD STATUS is supplied.
- Never invent values that are not present in supplied file data.
- Never write fake links, placeholder links, "insert downloadable link", or
  claim that a PDF/file exists. The application itself creates download files.
""".format(current_date=datetime.now().strftime("%d %B %Y"))

    if st.session_state.file_context and (
        pdf_mode == "content" or should_use_file_context(prompt)
    ):
        file_context_for_prompt = st.session_state.file_context
        if pdf_mode == "content" and len(file_context_for_prompt) > 18000:
            file_context_for_prompt = (
                file_context_for_prompt[:18000]
                + "\n[Source shortened for faster PDF generation.]"
            )
        system_prompt += (
            "\n\nUPLOADED FILE CONTENT:\n"
            + file_context_for_prompt
        )

    if pdf_mode == "content":
        system_prompt += """

PDF DOCUMENT TASK:
- Create the complete document requested by the user using the uploaded file
  and conversation as the source.
- Produce 12-15 high-value questions WITH concise interview-ready answers,
  not a summary. Keep each answer practical and focused.
- Cover every important topic in the source and organize it with clear
  headings and numbered questions.
- Output only the document content. Do not mention links, downloads, PDF-style
  text, limitations, or ask follow-up questions.
The application will privately convert your output into a real PDF.
"""

    web_needed = use_internet and needs_web_search(prompt)

    if web_needed:
        results = web_search(prompt)

        if wants_download(prompt):
            downloaded_path, download_status = try_download_from_results(results)

            if downloaded_path:
                try:
                    import pandas as pd

                    ext = os.path.splitext(downloaded_path.lower())[1]

                    if ext == ".csv":
                        downloaded_content = pd.read_csv(
                            downloaded_path
                        ).to_string(index=False)

                    elif ext in [".xlsx", ".xls", ".xlsm"]:
                        book = pd.ExcelFile(downloaded_path)
                        parts = []
                        for sheet in book.sheet_names:
                            df = pd.read_excel(downloaded_path, sheet_name=sheet)
                            parts.append(f"--- Excel Sheet: {sheet} ---")
                            parts.append(df.to_string(index=False))
                        downloaded_content = "\n".join(parts)

                    elif ext == ".json":
                        downloaded_content = json.dumps(
                            json.load(open(downloaded_path, encoding="utf-8")),
                            indent=2,
                            ensure_ascii=False
                        )

                    else:
                        downloaded_content = Path(downloaded_path).read_text(
                            encoding="utf-8", errors="replace"
                        )

                    if len(downloaded_content) > 60000:
                        downloaded_content = (
                            downloaded_content[:60000]
                            + "\n[Downloaded file content truncated.]"
                        )

                    system_prompt += (
                        "\n\nDOWNLOAD STATUS:\n"
                        + download_status
                        + "\n\nDOWNLOADED FILE CONTENT:\n"
                        + downloaded_content
                    )

                except Exception as e:
                    system_prompt += (
                        "\n\nDOWNLOAD STATUS:\n"
                        + download_status
                        + f"\nCould not parse downloaded file: {e}"
                    )
            else:
                system_prompt += "\n\nDOWNLOAD STATUS:\n" + download_status

        system_prompt += "\n\nWEB SEARCH RESULTS:\n" + results

    # PDF generation needs the uploaded source and current request, not the
    # entire chat history. This greatly reduces prompt-processing time.
    if pdf_mode == "content":
        window_messages = st.session_state.messages[-1:]
    else:
        window_messages = st.session_state.messages[-WINDOW_MEMORY:]

    nova_messages = [
        {"role": "system", "content": system_prompt}
    ] + window_messages

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        placeholder.markdown("Thinking…")

        try:
            model_options = {
                "temperature": TEMPERATURE,
                "num_ctx": 3072 if pdf_mode == "content" else 4096,
            }
            if pdf_mode == "content":
                model_options.update({
                    "num_predict": 900,
                    "num_batch": 512,
                })

            stream = stream_nova_chat(
                local_model=model,
                messages=nova_messages,
                model_options=model_options,
            )

            writing_started = False
            for chunk in stream:
                piece = chunk.get("message", {}).get(
                    "content", ""
                )

                answer += piece
                if pdf_mode == "content" and piece and not writing_started:
                    placeholder.markdown("Writing your PDF…")
                    writing_started = True
                elif pdf_mode != "content":
                    placeholder.markdown(answer + "▌")

            if pdf_mode == "content":
                pdf_bytes = build_chat_pdf(
                    [{"role": "assistant", "content": answer}],
                    title="Nova - Interview Questions and Answers",
                )
                st.session_state.generated_pdf = pdf_bytes
                st.session_state.generated_pdf_name = (
                    "Nova_Interview_Questions_and_Answers.pdf"
                )
                st.session_state.generated_pdf_chat_id = (
                    st.session_state.chat_id
                )

                ready_message = "✅ Your PDF is ready."
                placeholder.success(ready_message)
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=st.session_state.generated_pdf_name,
                    mime="application/pdf",
                    key=f"content_pdf_{st.session_state.chat_id}",
                )
                save_message(
                    st.session_state.chat_id,
                    "assistant",
                    ready_message,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": ready_message}
                )
            else:
                placeholder.markdown(answer)
                save_message(
                    st.session_state.chat_id,
                    "assistant",
                    answer,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

        except Exception as e:
            if GEMINI_API_KEY:
                error_message = (
                    "❌ Nova could not reach Gemini. "
                    + html.escape(str(e))[:700]
                )
            elif OPENROUTER_API_KEY:
                error_message = (
                    "❌ Nova could not reach the cloud model. Check the "
                    "OpenRouter key, balance, and model access in your "
                    "deployment settings."
                )
            else:
                error_message = f"❌ Error: {e}"
            placeholder.error(error_message)
            save_message(
                st.session_state.chat_id,
                "assistant",
                error_message,
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )

    # Re-render once after completion. This removes the temporary new-chat
    # layout and leaves the finished conversation in the compact chat view.
    st.rerun()
