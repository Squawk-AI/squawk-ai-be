from config import supabase_CHAT
import uuid
from datetime import datetime, timezone
from langchain_core.messages import AIMessage, HumanMessage
import logging
from fastapi import HTTPException
from utils import retry_with_backoff
import sentry_sdk

logger = logging.getLogger(__name__)


def log_conversation(conversation_id: str, thread_id: str):
    try:
        def _insert():
            return (
                supabase_CHAT.table("conversation").insert({
                    "id": conversation_id,
                    "thread_id": thread_id,
                    "status": "initializing",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).execute()
            )
        retry_with_backoff(_insert, retries=3, initial_delay=0.5)
        logger.info(f"Logged conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Failed to log conversation {conversation_id}: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)


def set_conversation_aircraft_and_running(conversation_id: str,
                                          aircraft_type: str | None = None,
                                          tail_number: str | None = None):
    
    update_fields = {"status": "running", "updated_at": datetime.now(timezone.utc).isoformat()}
    update_fields["aircraft_type"] = aircraft_type
    if tail_number is not None:
        update_fields["tail_number"] = tail_number

    try:
        def _update():
            return (
                supabase_CHAT
                .table("conversation")
                .update(update_fields)
                .eq("id", conversation_id)
                .execute()
            )
        resp = retry_with_backoff(_update, retries=3, initial_delay=0.5)
        logger.info(f"Set running + aircraft fields for {conversation_id}")
    except Exception as e:
        logger.error("Failed to set running + aircraft fields", exc_info=True)
        sentry_sdk.capture_exception(e)


def log_last_message(state: dict):
    messages = state.get("messages", [])
    if not messages:
        return

    last = messages[-1]
    role = (
        "assistant" if isinstance(last, AIMessage)
        else "user" if isinstance(last, HumanMessage)
        else "system"
    )

    try:
        def _insert():
            return supabase_CHAT.table("message").insert({
                "id": str(uuid.uuid4()),
                "conversation_id": state["conversation_id"],
                "content": last.content,
                "role": role,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        retry_with_backoff(_insert, retries=3, initial_delay=0.5)
        logger.info(f"Logged {role} message for conversation {state['conversation_id']}")
    except Exception as e:
        logger.error(f"Failed to log message for conversation {state.get('conversation_id')}: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)


def update_conversation_status(conversation_id: str, status: str = "completed"):
    try:
        def _update():
            return (
                supabase_CHAT
                .table("conversation")
                .update({"status": status, "updated_at": datetime.now(timezone.utc).isoformat()})
                .eq("id", conversation_id)
                .execute()
            )
        retry_with_backoff(_update, retries=3, initial_delay=0.5)
        logger.info(f"Updated conversation {conversation_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update conversation status for {conversation_id}: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)


def insert_summary(conversation_id: str, summary: str):
    now = datetime.now(timezone.utc).isoformat()
    try:
        def _insert():
            return (
                supabase_CHAT
                .table("conversation_summary")
                .insert({
                    "id": str(uuid.uuid4()),
                    "conversation_id": conversation_id,
                    "summary": summary,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                .execute()
            )
        retry_with_backoff(_insert, retries=3, initial_delay=0.5)
        logger.info(f"Inserted summary for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Failed to insert conversation summary for {conversation_id}: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)

def log_feedback(conversation_id: str, thumb_feedback: bool) -> str:
    """Update ONLY the most recent conversation_summary row for this conversation."""

    # 1) Find latest summary row
    latest = (
        supabase_CHAT.table("conversation_summary")
        .select("id")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not latest.data:
        raise HTTPException(status_code=404, detail="No summary row found for conversation")

    summary_id = latest.data[0]["id"]

    # 2) Update feedback fields (no inserts)
    update = {
        "thumb_feedback": thumb_feedback,
    }

    def _update():
        return (
            supabase_CHAT.table("conversation_summary")
            .update(update)
            .eq("id", summary_id)
            .execute()
        )
    try:
        retry_with_backoff(_update, retries=3, initial_delay=0.5)
        logger.info(f"Added feedback to summary {summary_id} for conversation {conversation_id}")
        return summary_id
    except Exception as e:
        logger.error(f"Failed to add feedback for conversation {conversation_id}: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail="Failed to save feedback")