"""
app_streamlit.py — AutoApply-AI Streamlit Dashboard
=====================================================
Browser-based interface sharing the same core engine and SQLite database.

Layout:
  Sidebar       — Configuration & CV status
  Left column   — Job input (text / image upload)
  Center column — Analysis result + editable pitch
  Bottom        — Application history ledger (pandas DataFrame)
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import core

load_dotenv()

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoApply-AI",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Initialise DB on app start
# ---------------------------------------------------------------------------
core.init_db()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("?? AutoApply-AI")
    st.caption("Powered by Gemini 2.5 Flash")
    st.divider()

    st.subheader("?? Configuration")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    smtp_user = os.getenv("SMTP_USER", "")
    cv_path = os.getenv("CV_PDF_PATH", "./my_cv.pdf")

    st.markdown(f"**Gemini API:** {'? Set' if gemini_key else '? Not set'}")
    st.markdown(f"**SMTP User:** {'? ' + smtp_user if smtp_user else '? Not set'}")

    cv_exists = Path(cv_path).exists()
    st.markdown(f"**CV File:** {'? ' + Path(cv_path).name if cv_exists else '? Not found at ' + cv_path}")

    if not gemini_key:
        st.error("Set GEMINI_API_KEY in your .env file.")
    if not cv_exists:
        st.warning(f"Place your CV at `{cv_path}` for best pitch quality.")

    st.divider()
    st.subheader("?? Stats")
    history = core.get_history()
    sent = sum(1 for r in history if r["status"] == "sent")
    pending = sum(1 for r in history if r["status"] == "pending")
    discarded = sum(1 for r in history if r["status"] == "discarded")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sent", sent, delta=None)
    col_b.metric("Pending", pending, delta=None)
    col_c.metric("Discarded", discarded, delta=None)

    st.divider()
    if st.button("?? Refresh Stats"):
        st.rerun()


# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "app_id" not in st.session_state:
    st.session_state.app_id = None
if "pitch_text" not in st.session_state:
    st.session_state.pitch_text = ""
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""
if "send_status" not in st.session_state:
    st.session_state.send_status = None


# ---------------------------------------------------------------------------
# Main Title
# ---------------------------------------------------------------------------
st.title("?? AutoApply-AI — Job Application Copilot")
st.caption("Paste a job description or upload a flyer image, review the AI pitch, and send in one click.")
st.divider()

# ---------------------------------------------------------------------------
# Input + Analysis — Two Columns
# ---------------------------------------------------------------------------
input_col, result_col = st.columns([1, 1], gap="large")

# -- Left: Input ------------------------------------------------------------
with input_col:
    st.subheader("?? Job Input")

    input_tab, image_tab = st.tabs(["?? Paste Text / URL", "?? Upload Image"])

    with input_tab:
        job_text_input = st.text_area(
            "Paste the full job description here:",
            height=340,
            placeholder="Software Engineer at Acme Corp\n\nWe are looking for...",
        )
        analyze_text_btn = st.button("?? Analyse Job Posting", type="primary", use_container_width=True)

    with image_tab:
        uploaded_file = st.file_uploader(
            "Upload a job flyer or screenshot:",
            type=["jpg", "jpeg", "png", "webp"],
            help="The image will be OCR'd by Gemini vision.",
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded flyer", use_container_width=True)
        analyze_image_btn = st.button("?? Analyse Image", type="primary", use_container_width=True)

    # -- Analysis trigger — Text -----------------------------------------
    if analyze_text_btn:
        if not job_text_input.strip() or len(job_text_input.strip()) < 30:
            st.error("Please paste a meaningful job description (at least a few sentences).")
        else:
            with st.status("Analysing job posting...", expanded=True) as status:
                st.write("🤖 Contacting Gemini AI (trying model chain if needed)...")
                try:
                    analysis = core.analyze_job(job_text_input)
                    company = analysis.get("company", "Unknown")
                    title = analysis.get("job_title", "Unknown")
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)

                    if core.is_duplicate(company, title):
                        st.warning(
                            f"⚠️ **Duplicate detected!** You already have an application for "
                            f"**{title}** at **{company}**. Check the history ledger below."
                        )
                    else:
                        app_id = core.log_application(
                            job_title=title,
                            company=company,
                            hr_email=analysis.get("hr_email"),
                            fit_score=analysis.get("fit_score"),
                            pitch=analysis.get("pitch"),
                            status="pending",
                            source="streamlit_text",
                        )
                        st.session_state.analysis = analysis
                        st.session_state.app_id = app_id
                        st.session_state.pitch_text = analysis.get("pitch", "")
                        st.session_state.send_status = None
                except Exception as exc:
                    status.update(label="❌ Analysis failed", state="error", expanded=True)
                    err = str(exc)
                    if "unavailable" in err.lower() or "503" in err:
                        st.error("⚠️ All Gemini models are currently overloaded. Please wait 1–2 minutes and try again.")
                    else:
                        st.error(err)

    # -- Analysis trigger — Image ----------------------------------------
    if analyze_image_btn:
        if not uploaded_file:
            st.error("Please upload an image first.")
        else:
            suffix = Path(uploaded_file.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.status("Processing image...", expanded=True) as status:
                try:
                    st.write("📸 Running OCR via Gemini vision...")
                    extracted_text = core.extract_text_from_image(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)

                    if not extracted_text.strip():
                        status.update(label="❌ No text found", state="error")
                        st.error("Could not extract meaningful text from the image.")
                    else:
                        st.write(f"✅ OCR complete ({len(extracted_text)} chars). Analysing job...")
                        analysis = core.analyze_job(extracted_text)
                        company = analysis.get("company", "Unknown")
                        title = analysis.get("job_title", "Unknown")
                        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

                        if core.is_duplicate(company, title):
                            st.warning(
                                f"⚠️ **Duplicate!** Already applied to **{title}** @ **{company}**."
                            )
                        else:
                            app_id = core.log_application(
                                job_title=title,
                                company=company,
                                hr_email=analysis.get("hr_email"),
                                fit_score=analysis.get("fit_score"),
                                pitch=analysis.get("pitch"),
                                status="pending",
                                source="streamlit_image",
                            )
                            st.session_state.analysis = analysis
                            st.session_state.app_id = app_id
                            st.session_state.pitch_text = analysis.get("pitch", "")
                            st.session_state.send_status = None
                except Exception as exc:
                    Path(tmp_path).unlink(missing_ok=True)
                    status.update(label="❌ Failed", state="error", expanded=True)
                    err = str(exc)
                    if "unavailable" in err.lower() or "503" in err:
                        st.error("⚠️ All Gemini models are currently overloaded. Please wait 1–2 minutes and try again.")
                    else:
                        st.error(err)


# -- Right: Analysis Result & Pitch Editor ----------------------------------
with result_col:
    st.subheader("?? Analysis & Pitch Editor")

    analysis = st.session_state.analysis

    if analysis is None:
        st.info("?? Paste a job description or upload an image to get started.")
    else:
        score = analysis.get("fit_score", 0)
        if score >= 80:
            score_color = "??"
            score_label = "Strong Match"
        elif score >= 55:
            score_color = "??"
            score_label = "Moderate Match"
        else:
            score_color = "??"
            score_label = "Weak Match"

        # Job Card
        st.markdown(
            f"""
            <div style="background:#f0f4ff;border-radius:10px;padding:16px 20px;margin-bottom:12px;">
              <h3 style="margin:0 0 6px 0;">{analysis.get('job_title', 'Unknown Role')}</h3>
              <p style="margin:0;color:#555;font-size:15px;">?? {analysis.get('company', 'Unknown Company')}</p>
              <p style="margin:4px 0 0 0;color:#555;font-size:14px;">?? HR: <code>{analysis.get('hr_email') or 'Not found'}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Fit Score
        fit_col, req_col = st.columns([1, 2])
        with fit_col:
            st.metric(
                label=f"Fit Score {score_color}",
                value=f"{score}/100",
                delta=score_label,
                delta_color="normal" if score >= 55 else "inverse",
            )
        with req_col:
            reqs = analysis.get("requirements", [])
            if reqs:
                st.markdown("**Key Requirements:**")
                for r in reqs[:6]:
                    st.markdown(f"• {r}")

        # Missing Info
        missing = analysis.get("missing_info", [])
        if missing:
            with st.expander(f"?? Missing Information ({len(missing)} items)", expanded=False):
                for m in missing:
                    st.warning(m)

        st.divider()

        # Pitch Editor
        st.markdown("**?? Application Pitch** *(editable)*")
        edited_pitch = st.text_area(
            label="Pitch",
            value=st.session_state.pitch_text,
            height=280,
            label_visibility="collapsed",
            key="pitch_editor",
        )
        # Sync edits back to session state
        if edited_pitch != st.session_state.pitch_text:
            st.session_state.pitch_text = edited_pitch

        st.divider()

        # Notes / Suggestions chat box
        st.markdown("**\U0001f4ac Personal Notes & Mentions** *(appended to the email)*")
        st.caption("Add anything extra — referrals, availability, portfolio links, salary expectations, etc.")
        edited_notes = st.text_area(
            label="Notes",
            value=st.session_state.notes_text,
            height=120,
            label_visibility="collapsed",
            key="notes_editor",
            placeholder="e.g. I was referred by Jane Smith  |  Available to start immediately  |  Portfolio: mysite.com",
        )
        if edited_notes != st.session_state.notes_text:
            st.session_state.notes_text = edited_notes

        st.divider()

        # HR Email override
        hr_email = analysis.get("hr_email", "")
        recipient_email = st.text_input(
            "?? Recipient Email (editable)",
            value=hr_email,
            placeholder="hr@company.com",
        )

        # Action Buttons
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("? Approve & Send", type="primary", use_container_width=True):
                if not recipient_email or "@" not in recipient_email:
                    st.error("Please provide a valid recipient email address.")
                else:
                    with st.spinner("Dispatching email..."):
                        success = core.send_application(
                            to_email=recipient_email,
                            job_title=analysis["job_title"],
                            company=analysis["company"],
                            pitch=st.session_state.pitch_text,
                            notes=st.session_state.notes_text,
                        )
                    if success:
                        core.update_application_status(st.session_state.app_id, "sent")
                        st.session_state.send_status = ("success", recipient_email)
                        st.session_state.analysis = None
                        st.session_state.notes_text = ""
                        st.rerun()
                    else:
                        st.error("? Failed to send email. Check SMTP settings in .env.")

        with btn_col2:
            if st.button("? Discard", use_container_width=True):
                core.update_application_status(st.session_state.app_id, "discarded")
                st.session_state.analysis = None
                st.session_state.app_id = None
                st.session_state.pitch_text = ""
                st.session_state.notes_text = ""
                st.rerun()

    # Show success banner after send
    if st.session_state.send_status:
        status_type, email = st.session_state.send_status
        if status_type == "success":
            st.success(f"? Application sent to **{email}**! Check the history ledger below.")
            st.session_state.send_status = None


# ---------------------------------------------------------------------------
# History Ledger
# ---------------------------------------------------------------------------
st.divider()
st.subheader("?? Application History Ledger")

history = core.get_history()

if not history:
    st.info("No applications logged yet. Analyse your first job posting above!")
else:
    df = pd.DataFrame(history)

    # Format for display
    display_df = df[["id", "submitted_at", "job_title", "company", "hr_email", "fit_score", "status", "source"]].copy()
    display_df.columns = ["ID", "Date", "Job Title", "Company", "HR Email", "Fit %", "Status", "Source"]
    display_df["Date"] = display_df["Date"].str[:16]
    display_df["HR Email"] = display_df["HR Email"].fillna("—")
    display_df["Fit %"] = display_df["Fit %"].fillna("—")

    # Colour-code status
    def status_badge(val: str) -> str:
        colours = {"sent": "background-color: #d4edda; color: #155724;",
                   "discarded": "background-color: #f8d7da; color: #721c24;",
                   "pending": "background-color: #fff3cd; color: #856404;"}
        return colours.get(val, "")

    styled = display_df.style.applymap(status_badge, subset=["Status"])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "?? Export History as CSV",
        data=csv,
        file_name="autoapply_history.csv",
        mime="text/csv",
    )
