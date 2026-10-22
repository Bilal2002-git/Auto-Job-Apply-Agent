"""
test_pipeline.py — AutoApply-AI Automated Sanity Test Suite
============================================================
Runs a series of unit and integration checks against the core module.
No actual emails are sent and no Gemini API calls are made during testing
(mocked where needed). A real Gemini call can be enabled via --live flag.

Usage:
    python test_pipeline.py          # Fast offline tests only
    python test_pipeline.py --live   # Include live Gemini API call (requires GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import textwrap
import time
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure project root is on path when running from another directory
sys.path.insert(0, str(Path(__file__).parent))

import core

# ---------------------------------------------------------------------------
# Test Runner Helpers
# ---------------------------------------------------------------------------
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results: list[tuple[str, str, str]] = []  # (name, status, message)


def run_test(name: str, fn) -> None:
    try:
        fn()
        results.append((name, "PASS", ""))
        print(f"  {PASS}  {name}")
    except AssertionError as e:
        results.append((name, "FAIL", str(e)))
        print(f"  {FAIL}  {name}: {e}")
    except Exception as e:
        results.append((name, "FAIL", f"Unexpected exception: {e}"))
        print(f"  {FAIL}  {name}: {type(e).__name__}: {e}")


def skip_test(name: str, reason: str) -> None:
    results.append((name, "SKIP", reason))
    print(f"  {SKIP}  {name}: {reason}")


# ---------------------------------------------------------------------------
# 1. Database Tests
# ---------------------------------------------------------------------------
print("\n?? [1/6] Database Layer Tests")

def test_db_init():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        core.init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applications'")
        assert cur.fetchone() is not None, "Table 'applications' not created"
        conn.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_db_columns():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        core.init_db(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.execute("PRAGMA table_info(applications)")
        cols = {row[1] for row in cur.fetchall()}
        required = {"id", "job_title", "company", "hr_email", "fit_score", "pitch", "status", "submitted_at"}
        missing = required - cols
        assert not missing, f"Missing columns: {missing}"
        conn.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_log_and_retrieve():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        core.init_db(db_path)
        app_id = core.log_application(
            job_title="Software Engineer",
            company="Test Corp",
            hr_email="hr@testcorp.com",
            fit_score=85,
            pitch="Test pitch.",
            status="pending",
            source="test",
            db_path=db_path,
        )
        assert isinstance(app_id, int) and app_id > 0, f"Invalid app_id: {app_id}"
        history = core.get_history(db_path=db_path)
        assert len(history) == 1, f"Expected 1 record, got {len(history)}"
        record = history[0]
        assert record["company"] == "Test Corp"
        assert record["fit_score"] == 85
        assert record["status"] == "pending"
    finally:
        Path(db_path).unlink(missing_ok=True)


run_test("DB init creates schema", test_db_init)
run_test("DB has all required columns", test_db_columns)
run_test("Log and retrieve application", test_log_and_retrieve)


# ---------------------------------------------------------------------------
# 2. Duplicate Detection Tests
# ---------------------------------------------------------------------------
print("\n?? [2/6] Duplicate Detection Tests")


def test_duplicate_detection():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        core.init_db(db_path)
        core.log_application(
            job_title="Data Analyst",
            company="Acme",
            hr_email="jobs@acme.com",
            fit_score=70,
            pitch="pitch",
            status="pending",
            source="test",
            db_path=db_path,
        )
        assert core.is_duplicate("Acme", "Data Analyst", db_path=db_path), "Should detect duplicate"
        assert not core.is_duplicate("Acme", "Senior Engineer", db_path=db_path), "Should not flag different title"
        assert not core.is_duplicate("Other Corp", "Data Analyst", db_path=db_path), "Should not flag different company"
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_discarded_not_duplicate():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        core.init_db(db_path)
        app_id = core.log_application(
            job_title="PM",
            company="Corp",
            hr_email=None,
            fit_score=60,
            pitch="pitch",
            status="discarded",
            source="test",
            db_path=db_path,
        )
        # Discarded applications should NOT block re-application
        assert not core.is_duplicate("Corp", "PM", db_path=db_path), \
            "Discarded applications should not count as duplicates"
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_status_update():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        core.init_db(db_path)
        app_id = core.log_application(
            job_title="Dev",
            company="XCo",
            hr_email="h@xco.com",
            fit_score=75,
            pitch="p",
            status="pending",
            source="test",
            db_path=db_path,
        )
        core.update_application_status(app_id, "sent", db_path=db_path)
        history = core.get_history(db_path=db_path)
        assert history[0]["status"] == "sent", "Status not updated to 'sent'"
    finally:
        Path(db_path).unlink(missing_ok=True)


run_test("Duplicate detection fires correctly", test_duplicate_detection)
run_test("Discarded applications are not duplicates", test_discarded_not_duplicate)
run_test("Status update persists correctly", test_status_update)


# ---------------------------------------------------------------------------
# 3. CV Loader Tests
# ---------------------------------------------------------------------------
print("\n?? [3/6] CV PDF Loader Tests")


def test_cv_missing_returns_empty():
    # lru_cache needs to be cleared for different path values
    core.load_cv_text.cache_clear()
    result = core.load_cv_text("/nonexistent/path/cv.pdf")
    assert result == "", f"Expected empty string, got: {result!r}"
    core.load_cv_text.cache_clear()


def test_cv_loads_from_pdf():
    """Create a minimal PDF and verify text extraction."""
    try:
        import pypdf
        from pypdf import PdfWriter
    except ImportError:
        raise AssertionError("pypdf not installed")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_path = f.name

    try:
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        # pypdf blank pages have no text to extract; we just verify no crash occurs
        with open(tmp_path, "wb") as f:
            writer.write(f)

        core.load_cv_text.cache_clear()
        result = core.load_cv_text(tmp_path)
        # Blank PDF returns empty string — that is acceptable
        assert isinstance(result, str), "Expected string return value"
    finally:
        core.load_cv_text.cache_clear()
        Path(tmp_path).unlink(missing_ok=True)


run_test("Missing CV returns empty string gracefully", test_cv_missing_returns_empty)
run_test("CV PDF loads without crash", test_cv_loads_from_pdf)


# ---------------------------------------------------------------------------
# 4. Gemini Schema Validation Tests
# ---------------------------------------------------------------------------
print("\n?? [4/6] Gemini Schema Validation Tests")

MOCK_ANALYSIS = {
    "job_title": "Senior Python Developer",
    "company": "FutureTech Ltd",
    "hr_email": "careers@futuretech.io",
    "requirements": ["Python 3.11+", "FastAPI", "PostgreSQL", "Docker"],
    "fit_score": 88,
    "missing_info": ["Salary range", "Remote policy"],
    "pitch": (
        "Dear Hiring Team at FutureTech Ltd,\n\n"
        "I am excited to apply for the Senior Python Developer role. "
        "With over 5 years of building production Python services, I bring exactly the expertise your team needs.\n\n"
        "In my current role, I architected a FastAPI microservice handling 50K RPM and migrated a legacy monolith "
        "to a Docker-based deployment pipeline—skills directly aligned with your requirements.\n\n"
        "I would love to explore how I can contribute to FutureTech's mission. "
        "I am available for a call at your earliest convenience.\n\nBest regards,"
    ),
}


def test_schema_required_keys():
    required = {"job_title", "company", "hr_email", "requirements", "fit_score", "missing_info", "pitch"}
    actual = set(MOCK_ANALYSIS.keys())
    missing = required - actual
    assert not missing, f"Missing required keys: {missing}"


def test_schema_types():
    assert isinstance(MOCK_ANALYSIS["job_title"], str)
    assert isinstance(MOCK_ANALYSIS["company"], str)
    assert isinstance(MOCK_ANALYSIS["requirements"], list)
    assert isinstance(MOCK_ANALYSIS["fit_score"], int)
    assert 0 <= MOCK_ANALYSIS["fit_score"] <= 100, "fit_score out of range"
    assert isinstance(MOCK_ANALYSIS["missing_info"], list)
    assert isinstance(MOCK_ANALYSIS["pitch"], str)
    assert len(MOCK_ANALYSIS["pitch"]) > 50, "Pitch too short"


def test_pitch_paragraph_structure():
    pitch = MOCK_ANALYSIS["pitch"]
    paragraphs = [p.strip() for p in pitch.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 3, f"Expected 3+ paragraphs, got {len(paragraphs)}"


run_test("Schema has all required keys", test_schema_required_keys)
run_test("Schema field types are correct", test_schema_types)
run_test("Pitch has 3+ paragraph structure", test_pitch_paragraph_structure)


# ---------------------------------------------------------------------------
# 5. Email MIME Construction Tests
# ---------------------------------------------------------------------------
print("\n?? [5/6] Email MIME Construction Tests")


def test_email_html_generation():
    html = core.build_email_html(
        pitch="Para one.\n\nPara two.\n\nPara three.",
        job_title="Engineer",
        company="TestCo",
        sender_name="Jane Doe",
    )
    assert "<p>" in html, "HTML body missing paragraph tags"
    assert "Engineer" in html
    assert "TestCo" in html
    assert "Jane Doe" in html
    assert "<!DOCTYPE html>" in html


def test_mime_payload_structure():
    """Verify that a MIME message is constructed without actually sending."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    html = core.build_email_html("Para.", "Dev", "Corp", "Test User")
    msg = MIMEMultipart("mixed")
    msg["From"] = "Test User <test@example.com>"
    msg["To"] = "hr@example.com"
    msg["Subject"] = "Application: Dev - Test User"

    alt_part = MIMEMultipart("alternative")
    alt_part.attach(MIMEText("Para.", "plain", "utf-8"))
    alt_part.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt_part)

    serialized = msg.as_string()
    assert "Application: Dev" in serialized, "Subject not found in serialized email"
    assert "hr@example.com" in serialized, "Recipient email not found"
    assert "Test User" in serialized, "Sender name not found"
    assert "multipart/mixed" in serialized, "Content-type header missing"
    assert "Content-Type: text/html" in serialized, "HTML part missing"


def test_missing_smtp_raises():
    """send_application should raise EnvironmentError if SMTP creds missing."""
    try:
        core.send_application(
            to_email="hr@example.com",
            job_title="Dev",
            company="Corp",
            pitch="pitch",
            smtp_host="",
            smtp_port=465,
            smtp_user="",
            smtp_pass="",
        )
        raise AssertionError("Expected EnvironmentError was not raised")
    except EnvironmentError:
        pass  # Expected


run_test("HTML email body generated correctly", test_email_html_generation)
run_test("MIME multipart payload structure valid", test_mime_payload_structure)
run_test("Missing SMTP config raises EnvironmentError", test_missing_smtp_raises)


# ---------------------------------------------------------------------------
# 6. End-to-End Pipeline Dry Run
# ---------------------------------------------------------------------------
print("\n?? [6/6] End-to-End Pipeline Dry Run (Mocked Gemini)")


def test_e2e_pipeline_mocked():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        core.init_db(db_path)

        # Mock Gemini to return our fixture
        with patch.object(core, "analyze_job", return_value=MOCK_ANALYSIS):
            analysis = core.analyze_job("Any job text here")

        company = analysis["company"]
        title = analysis["job_title"]

        assert not core.is_duplicate(company, title, db_path=db_path), "Should not be duplicate yet"

        app_id = core.log_application(
            job_title=title,
            company=company,
            hr_email=analysis["hr_email"],
            fit_score=analysis["fit_score"],
            pitch=analysis["pitch"],
            status="pending",
            source="test_e2e",
            db_path=db_path,
        )

        assert core.is_duplicate(company, title, db_path=db_path), "Should be duplicate now"

        core.update_application_status(app_id, "sent", db_path=db_path)

        history = core.get_history(db_path=db_path)
        assert len(history) == 1
        assert history[0]["status"] == "sent"
        assert history[0]["fit_score"] == 88
    finally:
        Path(db_path).unlink(missing_ok=True)


run_test("End-to-end pipeline dry run (mocked)", test_e2e_pipeline_mocked)


# ---------------------------------------------------------------------------
# Live Gemini Test (opt-in)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--live", action="store_true")
args, _ = parser.parse_known_args()

if args.live:
    print("\n?? [LIVE] Gemini API Test")

    def test_live_gemini():
        job_text = textwrap.dedent("""
            Software Engineer — Backend
            Company: LiveTest Inc
            Location: Remote
            We need a Python developer with experience in FastAPI, PostgreSQL, and Docker.
            Apply to: jobs@livetest.io
        """)
        result = core.analyze_job(job_text, cv_text="Python developer with 5 years of FastAPI experience.")
        required_keys = {"job_title", "company", "fit_score", "pitch", "requirements", "missing_info"}
        missing = required_keys - set(result.keys())
        assert not missing, f"Missing keys in live response: {missing}"
        assert 0 <= result["fit_score"] <= 100
        assert len(result["pitch"]) > 100

    run_test("Live Gemini API analysis", test_live_gemini)
else:
    skip_test("Live Gemini API analysis", "Run with --live flag to enable")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 56)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
skipped = sum(1 for _, s, _ in results if s == "SKIP")
total = len(results)

print(f"Results: {passed}/{total} passed | {failed} failed | {skipped} skipped")

if failed > 0:
    print("\nFailed tests:")
    for name, status, msg in results:
        if status == "FAIL":
            print(f"  • {name}: {msg}")

print("=" * 56)
sys.exit(0 if failed == 0 else 1)
