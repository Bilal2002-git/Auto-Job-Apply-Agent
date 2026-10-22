"""
app_telegram.py — AutoApply-AI Telegram Bot Interface
=======================================================
Async python-telegram-bot v20+ interface.

Supported interactions:
  /start          — Welcome message with usage instructions
  /history        — Show last 10 applications from the DB
  Text message    — Treated as raw job posting text ? analyzed by Gemini
  Photo message   — Flyer screenshot ? OCR via Gemini vision ? same flow
  Inline buttons  — Approve & Send | Discard | (Edit Pitch is prompted via reply)

Pending analyses are stored in bot_data (per-chat session dict) keyed by chat_id.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import core

load_dotenv()
logger = logging.getLogger("autoapply.telegram")

BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ---------------------------------------------------------------------------
# Session Keys
# ---------------------------------------------------------------------------
# bot_data["pending"][chat_id] = { "analysis": dict, "app_id": int }
PENDING_KEY = "pending"


def _get_pending(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> dict | None:
    return context.bot_data.get(PENDING_KEY, {}).get(chat_id)


def _set_pending(context: ContextTypes.DEFAULT_TYPE, chat_id: int, data: dict) -> None:
    if PENDING_KEY not in context.bot_data:
        context.bot_data[PENDING_KEY] = {}
    context.bot_data[PENDING_KEY][chat_id] = data


def _clear_pending(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    context.bot_data.get(PENDING_KEY, {}).pop(chat_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fit_emoji(score: int) -> str:
    if score >= 80:
        return "??"
    if score >= 55:
        return "??"
    return "??"


def _format_analysis_card(analysis: dict) -> str:
    score = analysis.get("fit_score", 0)
    reqs = analysis.get("requirements", [])
    missing = analysis.get("missing_info", [])
    req_lines = "\n".join(f"  • {r}" for r in reqs[:8])
    miss_lines = "\n".join(f"  ?? {m}" for m in missing) if missing else "  ? None"
    pitch_preview = (analysis.get("pitch", "")[:300] + "…") if len(analysis.get("pitch", "")) > 300 else analysis.get("pitch", "")

    return (
        f"*?? Job Analysis Complete*\n\n"
        f"*Role:* {analysis.get('job_title', 'Unknown')}\n"
        f"*Company:* {analysis.get('company', 'Unknown')}\n"
        f"*HR Email:* `{analysis.get('hr_email') or 'Not found'}`\n"
        f"*Fit Score:* {_fit_emoji(score)} *{score}/100*\n\n"
        f"*Key Requirements:*\n{req_lines}\n\n"
        f"*Missing Info:*\n{miss_lines}\n\n"
        f"*?? Draft Pitch Preview:*\n_{pitch_preview}_"
    )


def _action_keyboard(app_id: int, has_email: bool) -> InlineKeyboardMarkup:
    approve_text = "? Approve & Send" if has_email else "? Approve (no HR email!)"
    rows = [
        [
            InlineKeyboardButton(approve_text, callback_data=f"approve:{app_id}"),
            InlineKeyboardButton("? Discard", callback_data=f"discard:{app_id}"),
        ],
        [
            InlineKeyboardButton("?? Edit Pitch", callback_data=f"edit:{app_id}"),
            InlineKeyboardButton("?? Full Pitch", callback_data=f"full_pitch:{app_id}"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "?? *Welcome to AutoApply-AI!*\n\n"
        "I help you apply to jobs faster with AI-crafted pitches.\n\n"
        "*How to use:*\n"
        "1?? *Paste* a job description directly in chat.\n"
        "2?? *Forward* or *upload* a job flyer screenshot.\n"
        "3?? Review the analysis & pitch I generate.\n"
        "4?? Hit *Approve & Send* to dispatch your application!\n\n"
        "?? Use /history to see past applications.\n\n"
        "_Powered by Gemini 2.5 Flash + AutoApply-AI_"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    records = core.get_history()[:10]
    if not records:
        await update.message.reply_text("?? No applications on record yet.")
        return

    lines = ["*?? Recent Applications (last 10):*\n"]
    status_emoji = {"sent": "?", "discarded": "?", "pending": "?"}
    for r in records:
        emoji = status_emoji.get(r["status"], "?")
        lines.append(
            f"{emoji} *{r['job_title']}* @ {r['company']}\n"
            f"   Fit: {r.get('fit_score', '?')}% | {r['submitted_at'][:16]}"
        )

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Job Analysis Flow
# ---------------------------------------------------------------------------
async def _run_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    job_text: str,
    source: str = "telegram_text",
) -> None:
    chat_id = update.effective_chat.id
    thinking_msg = await update.effective_message.reply_text(
        "?? *Analysing job posting with Gemini 2.5 Flash…*\n_This takes 5–15 seconds._",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        analysis = core.analyze_job(job_text)
    except Exception as exc:
        await thinking_msg.edit_text(f"? Analysis failed: {exc}")
        return

    company = analysis.get("company", "Unknown")
    title = analysis.get("job_title", "Unknown")

    # Duplicate check
    if core.is_duplicate(company, title):
        await thinking_msg.edit_text(
            f"?? *Duplicate detected!*\nYou have already applied to *{title}* at *{company}*.\n\n"
            "Send the job text again to force a new record, or use /history.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Log as pending
    app_id = core.log_application(
        job_title=title,
        company=company,
        hr_email=analysis.get("hr_email"),
        fit_score=analysis.get("fit_score"),
        pitch=analysis.get("pitch"),
        status="pending",
        source=source,
    )

    _set_pending(context, chat_id, {"analysis": analysis, "app_id": app_id})

    card = _format_analysis_card(analysis)
    keyboard = _action_keyboard(app_id, bool(analysis.get("hr_email")))

    await thinking_msg.edit_text(card, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages — treat as job posting."""
    chat_id = update.effective_chat.id
    pending = _get_pending(context, chat_id)

    # Check if user is replying with an edited pitch
    if pending and update.message.reply_to_message:
        new_pitch = update.message.text.strip()
        analysis = pending["analysis"]
        analysis["pitch"] = new_pitch
        _set_pending(context, chat_id, {"analysis": analysis, "app_id": pending["app_id"]})
        await update.message.reply_text(
            "?? Pitch updated! Use the buttons above to approve or discard.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    job_text = update.message.text.strip()
    if len(job_text) < 30:
        await update.message.reply_text(
            "Please paste a full job description (at least a few sentences)."
        )
        return

    await _run_analysis(update, context, job_text, source="telegram_text")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages — OCR the image and analyze as job posting."""
    status_msg = await update.message.reply_text(
        "?? *Processing job flyer image…*", parse_mode=ParseMode.MARKDOWN
    )

    # Download the highest-resolution photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await file.download_to_drive(tmp_path)
        job_text = core.extract_text_from_image(tmp_path)
    except Exception as exc:
        await status_msg.edit_text(f"? Image processing failed: {exc}")
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not job_text.strip():
        await status_msg.edit_text("? Could not extract text from the image. Please paste the job text directly.")
        return

    await status_msg.edit_text(
        f"?? Image OCR complete ({len(job_text)} chars). Analyzing…"
    )
    await _run_analysis(update, context, job_text, source="telegram_photo")


# ---------------------------------------------------------------------------
# Callback Handlers
# ---------------------------------------------------------------------------
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    chat_id = update.effective_chat.id
    action, app_id_str = query.data.split(":", 1)
    app_id = int(app_id_str)

    pending = _get_pending(context, chat_id)

    if action == "approve":
        if not pending or pending["app_id"] != app_id:
            await query.edit_message_text("?? Session expired. Please re-paste the job posting.")
            return

        analysis = pending["analysis"]
        hr_email = analysis.get("hr_email", "")

        if not hr_email:
            await query.edit_message_reply_markup(
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("? Discard", callback_data=f"discard:{app_id}"),
                ]])
            )
            await context.bot.send_message(
                chat_id,
                "?? *No HR email found in the posting.*\n\n"
                "Please reply with the recipient email address to proceed:",
                parse_mode=ParseMode.MARKDOWN,
            )
            # Store state for manual email input
            context.chat_data["awaiting_email"] = app_id
            return

        await query.edit_message_text(
            f"?? *Sending application to `{hr_email}`…*",
            parse_mode=ParseMode.MARKDOWN,
        )

        success = core.send_application(
            to_email=hr_email,
            job_title=analysis["job_title"],
            company=analysis["company"],
            pitch=analysis["pitch"],
        )

        if success:
            core.update_application_status(app_id, "sent")
            _clear_pending(context, chat_id)
            await query.edit_message_text(
                f"? *Application sent!*\n\n"
                f"?? Dispatched to `{hr_email}`\n"
                f"?? *{analysis['job_title']}* @ *{analysis['company']}*",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await query.edit_message_text(
                "? *Failed to send email.* Check your SMTP credentials in .env and try again.",
                parse_mode=ParseMode.MARKDOWN,
            )

    elif action == "discard":
        core.update_application_status(app_id, "discarded")
        _clear_pending(context, chat_id)
        await query.edit_message_text(
            "??? Application discarded. Send another job posting whenever you're ready.",
        )

    elif action == "edit":
        await context.bot.send_message(
            chat_id,
            "?? *Edit Mode*\n\nReply to *any message* in this chat with your revised pitch text.\n"
            "It will replace the AI-generated draft.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "full_pitch":
        if pending and pending["app_id"] == app_id:
            pitch = pending["analysis"].get("pitch", "No pitch available.")
            await context.bot.send_message(
                chat_id,
                f"*?? Full Application Pitch:*\n\n{pitch}",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await context.bot.send_message(chat_id, "?? Pitch not found in session.")


async def handle_email_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle manual email address input when HR email was missing."""
    if "awaiting_email" not in context.chat_data:
        return

    app_id = context.chat_data.pop("awaiting_email")
    chat_id = update.effective_chat.id
    pending = _get_pending(context, chat_id)

    if not pending or pending["app_id"] != app_id:
        await update.message.reply_text("?? Session expired. Please re-paste the job posting.")
        return

    to_email = update.message.text.strip()
    if "@" not in to_email or "." not in to_email:
        await update.message.reply_text("? That doesn't look like a valid email. Please try again.")
        context.chat_data["awaiting_email"] = app_id
        return

    analysis = pending["analysis"]
    await update.message.reply_text(f"?? Sending to `{to_email}`…", parse_mode=ParseMode.MARKDOWN)

    success = core.send_application(
        to_email=to_email,
        job_title=analysis["job_title"],
        company=analysis["company"],
        pitch=analysis["pitch"],
    )

    if success:
        core.update_application_status(app_id, "sent")
        _clear_pending(context, chat_id)
        await update.message.reply_text(
            f"? *Application sent to `{to_email}`!*\n"
            f"*{analysis['job_title']}* @ *{analysis['company']}*",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(
            "? Failed to send email. Check your SMTP credentials in .env.",
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    if not BOT_TOKEN:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN is not set in .env.")

    core.init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # Text handler must come last to catch both job posts and email replies
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("AutoApply-AI Telegram bot starting... (Ctrl+C to stop)")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
