"""
core.py — AutoApply-AI Shared Engine
=====================================
Central module shared by both Telegram and Streamlit frontends.
Provides:
  - SQLite database layer (init, duplicate check, log, history)
  - CV PDF text extraction (pypdf, cached)
  - Gemini 2.5 Flash job analysis with structured JSON output
  - Image-to-text OCR via Gemini vision
  - SMTP email dispatch with CV attachment
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import sqlite3
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
import pypdf
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("autoapply.core")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER: str = os.getenv("SMTP_USER", "")
SMTP_PASS: str = os.getenv("SMTP_PASS", "")
SENDER_NAME: str = os.getenv("SENDER_NAME", "Applicant")
CV_PDF_PATH: str = os.getenv("CV_PDF_PATH", "./my_cv.pdf")
DB_PATH: str = os.getenv("DB_PATH", "./jobs.db")

GEMINI_MODEL: str = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Gemini client (lazy-initialised)
# ---------------------------------------------------------------------------
_gemini_client: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    """Return a cached Gemini API client, initialising it on first call."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialised (model=%s)", GEMINI_MODEL)
    return _gemini_client


# ---------------------------------------------------------------------------
# Database Layer
# ---------------------------------------------------------------------------
_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title   TEXT    NOT NULL,
    company     TEXT    NOT NULL,
    hr_email    TEXT,
    fit_score   INTEGER,
    pitch       TEXT,
    status      TEXT    NOT NULL DEFAULT 'pending',
    source      TEXT,
    submitted_at TEXT   NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_company_title
    ON applications (company, job_title);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create the SQLite database and table schema if they do not exist."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_DB_SCHEMA)
    conn.commit()
    conn.close()
    logger.info("Database ready at %s", db_path)


def is_duplicate(
    company: str,
    job_title: str,
    db_path: str = DB_PATH,
) -> bool:
    """Return True if a non-discarded application for this role already exists."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT 1 FROM applications WHERE company = ? AND job_title = ? AND status != 'discarded' LIMIT 1",
        (company.strip(), job_title.strip()),
    )
    found = cur.fetchone() is not None
    conn.close()
    return found


def log_application(
    job_title: str,
    company: str,
    hr_email: Optional[str],
    fit_score: Optional[int],
    pitch: Optional[str],
    status: str = "pending",
    source: str = "unknown",
    db_path: str = DB_PATH,
) -> int:
    """Insert a new application record and return its row ID."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        """
        INSERT INTO applications (job_title, company, hr_email, fit_score, pitch, status, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (job_title, company, hr_email, fit_score, pitch, status, source),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    logger.info("Logged application id=%s company=%s title=%s status=%s", row_id, company, job_title, status)
    return row_id


def update_application_status(
    app_id: int,
    status: str,
    db_path: str = DB_PATH,
) -> None:
    """Update the status of an existing application record."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE applications SET status = ? WHERE id = ?",
        (status, app_id),
    )
    conn.commit()
    conn.close()
    logger.info("Updated application id=%s status=%s", app_id, status)


def get_history(db_path: str = DB_PATH) -> list[dict]:
    """Return all application records as a list of dicts, newest first."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM applications ORDER BY submitted_at DESC"
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# CV Loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_cv_text(cv_path: str = CV_PDF_PATH) -> str:
    """
    Extract and cache all text from the candidate's PDF CV.
    Returns an empty string (with a warning) if the file is not found.
    """
    path = Path(cv_path)
    if not path.exists():
        logger.warning("CV PDF not found at %s — proceeding without CV context.", cv_path)
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        logger.info("CV loaded: %d pages, %d chars from %s", len(reader.pages), len(text), cv_path)
        return text
    except Exception as exc:
        logger.error("Failed to read CV PDF: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------
JOB_ANALYSIS_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    description="Structured analysis of a job posting for an automated application agent.",
    required=["job_title", "company", "requirements", "fit_score", "pitch", "missing_info"],
    properties={
        "job_title": types.Schema(
            type=types.Type.STRING,
            description="Exact job title from the posting.",
        ),
        "company": types.Schema(
            type=types.Type.STRING,
            description="Name of the hiring company or organisation.",
        ),
        "hr_email": types.Schema(
            type=types.Type.STRING,
            description="HR or recruiter email address. Empty string if not found.",
        ),
        "requirements": types.Schema(
            type=types.Type.ARRAY,
            description="Key skills and requirements extracted from the posting.",
            items=types.Schema(type=types.Type.STRING),
        ),
        "fit_score": types.Schema(
            type=types.Type.INTEGER,
            description="Candidate fit score 0-100 based on CV vs requirements.",
        ),
        "missing_info": types.Schema(
            type=types.Type.ARRAY,
            description="List of important details absent from the posting (e.g. salary, location, deadline).",
            items=types.Schema(type=types.Type.STRING),
        ),
        "pitch": types.Schema(
            type=types.Type.STRING,
            description=(
                "A polished, tailored 3-paragraph cold email pitch. "
                "Paragraph 1: Strong opening hook referencing the specific role. "
                "Paragraph 2: Candidate top 2-3 achievements mapped to the job requirements. "
                "Paragraph 3: Clear call-to-action and professional sign-off."
            ),
        ),
    },
)


# ---------------------------------------------------------------------------
# Gemini Analysis
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are AutoApply-AI, an expert career coach and recruiter with 15 years of experience.
Your task is to:
1. Parse the provided job posting and extract structured details.
2. Review the candidate's CV to evaluate their fit for the role.
3. Write a compelling, personalised 3-paragraph cold email pitch.
4. Identify any missing information in the posting.

Be honest about the fit score — do not inflate it artificially.
The pitch must sound human, confident, and specific to this role — never generic.
"""


def analyze_job(
    job_text: str,
    cv_text: Optional[str] = None,
) -> dict:
    """
    Send job posting text to Gemini 2.5 Flash and return a structured analysis dict.

    Args:
        job_text: Raw text of the job posting (from paste, URL scrape, or OCR).
        cv_text:  Optional candidate CV text for fit scoring and pitch tailoring.
                  Falls back to loading from CV_PDF_PATH if not supplied.

    Returns:
        Dict matching JOB_ANALYSIS_SCHEMA with keys:
        job_title, company, hr_email, requirements, fit_score, missing_info, pitch
    """
    if cv_text is None:
        cv_text = load_cv_text()

    client = get_gemini_client()

    user_message = f"""## JOB POSTING
{job_text.strip()}

## CANDIDATE CV
{cv_text.strip() if cv_text else "(No CV provided — base fit score on job requirements only.)"}
"""

    logger.info("Sending job analysis request to Gemini (%d chars)...", len(user_message))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=JOB_ANALYSIS_SCHEMA,
            temperature=0.4,
        ),
    )

    raw_json = response.text
    result = json.loads(raw_json)
    logger.info(
        "Analysis complete — company=%s title=%s fit=%s%%",
        result.get("company"),
        result.get("job_title"),
        result.get("fit_score"),
    )
    return result


# ---------------------------------------------------------------------------
# Image OCR via Gemini Vision
# ---------------------------------------------------------------------------
def extract_text_from_image(image_path: str) -> str:
    """
    Use Gemini vision capability to extract the full text content of a job
    flyer or screenshot image.

    Args:
        image_path: Absolute or relative path to the image file (JPEG/PNG/WEBP).

    Returns:
        Extracted text string ready to be fed into analyze_job().
    """
    import io

    client = get_gemini_client()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(str(path))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(suffix, "image/jpeg")

    buffer = io.BytesIO()
    fmt = "JPEG" if mime_type == "image/jpeg" else "PNG"
    img.save(buffer, format=fmt)
    image_bytes = buffer.getvalue()

    logger.info("Extracting text from image %s (%d bytes)...", path.name, len(image_bytes))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Extract all the text content from this job posting image. "
            "Return only the raw extracted text, preserving structure where possible.",
        ],
    )
    extracted = response.text.strip()
    logger.info("Image OCR complete — extracted %d chars.", len(extracted))
    return extracted


# ---------------------------------------------------------------------------
# Email Dispatch
# ---------------------------------------------------------------------------
def build_email_html(pitch: str, job_title: str, company: str, sender_name: str) -> str:
    """Build a clean HTML email body from the pitch text."""
    paragraphs = pitch.split("\n\n") if "\n\n" in pitch else pitch.split("\n")
    para_html = "".join(f"<p>{p.strip()}</p>" for p in paragraphs if p.strip())
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 15px; color: #222; line-height: 1.7; max-width: 680px; margin: auto; padding: 32px; }}
    p {{ margin: 0 0 16px 0; }}
    .footer {{ margin-top: 32px; border-top: 1px solid #eee; padding-top: 16px; color: #555; font-size: 13px; }}
  </style>
</head>
<body>
  {para_html}
  <div class="footer">
    This email was prepared by AutoApply-AI for <strong>{sender_name}</strong>.<br>
    Application for: <em>{job_title}</em> @ <em>{company}</em>
  </div>
</body>
</html>"""


def send_application(
    to_email: str,
    job_title: str,
    company: str,
    pitch: str,
    cv_path: str = CV_PDF_PATH,
    smtp_host: str = SMTP_HOST,
    smtp_port: int = SMTP_PORT,
    smtp_user: str = SMTP_USER,
    smtp_pass: str = SMTP_PASS,
    sender_name: str = SENDER_NAME,
) -> bool:
    """
    Dispatch the cold application email with CV attachment via SMTP SSL.

    Args:
        to_email:    Recipient HR/recruiter email address.
        job_title:   Job role title (used in subject line).
        company:     Company name (used in subject line).
        pitch:       3-paragraph pitch text (converted to HTML body).
        cv_path:     Path to the candidate CV PDF to attach.
        smtp_*:      SMTP server credentials (defaults to env vars).
        sender_name: Display name for the From header.

    Returns:
        True if the email was dispatched successfully, False otherwise.
    """
    if not all([smtp_host, smtp_user, smtp_pass]):
        raise EnvironmentError(
            "SMTP configuration is incomplete. Set SMTP_HOST, SMTP_USER, SMTP_PASS in .env."
        )

    subject = f"Application: {job_title} — {sender_name}"
    html_body = build_email_html(pitch, job_title, company, sender_name)

    msg = MIMEMultipart("mixed")
    msg["From"] = f"{sender_name} <{smtp_user}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")

    alt_part = MIMEMultipart("alternative")
    alt_part.attach(MIMEText(pitch, "plain", "utf-8"))
    alt_part.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt_part)

    cv_file = Path(cv_path)
    if cv_file.exists():
        with open(cv_file, "rb") as f:
            pdf_bytes = f.read()
        cv_attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
        cv_attachment.add_header(
            "Content-Disposition",
            "attachment",
            filename=cv_file.name,
        )
        msg.attach(cv_attachment)
        logger.info("CV attached: %s (%d bytes)", cv_file.name, len(pdf_bytes))
    else:
        logger.warning("CV file not found at %s — sending without attachment.", cv_path)

    try:
        logger.info("Connecting to SMTP %s:%s...", smtp_host, smtp_port)
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        logger.info("Email dispatched to %s (subject: %s)", to_email, subject)
        return True
    except smtplib.SMTPException as exc:
        logger.error("SMTP error: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error during email dispatch: %s", exc)
        return False
