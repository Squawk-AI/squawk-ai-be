#main.py

import uuid
import re
import asyncio
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict
from db_logging import log_conversation, log_last_message, log_feedback

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from graph import build_graph, GraphState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

graph_app = None
conversation_threads: Dict[str, str] = {}
background_task_references = set()
EXIT_PATTERN = re.compile(r"^\s*(?:/)?(?:exit|quit|end)\s*$", re.IGNORECASE)


class UserResponse(BaseModel):
    user_response: str


class ConversationResponse(BaseModel):
    conversation_id: str


class PollResponse(BaseModel):
    messages: list[dict]
    paused_for_input: bool
    stopped: bool
    current_state: dict

class FeedbackBody(BaseModel):
    thumb_feedback: bool
    comment: str | None = None


def _is_graph_finished(state) -> bool:
    return not state.values.get("paused_for_input", False) and state.next == ()


def init_sentry():
    """Initialize Sentry with proper configuration for production monitoring."""
    sentry_dsn = os.getenv("SENTRY_DSN")
    if not sentry_dsn:
        logger.warning("SENTRY_DSN not found in environment variables. Sentry monitoring disabled.")
        return
    
    # Configure Sentry integrations
    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors as events
    )
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[
            FastApiIntegration(),
            sentry_logging,
            AsyncioIntegration(),
        ],
        # Performance monitoring
        traces_sample_rate=0.1,  # 10% of transactions for performance monitoring
        # Error sampling
        sample_rate=1.0,  # 100% of errors
        # Environment
        environment=os.getenv("ENVIRONMENT", "development"),
        # Release tracking (useful for deployments)
        release=os.getenv("RELEASE_VERSION", "unknown"),
        # Additional context
        before_send=lambda event, hint: event,  # Can add filtering here if needed
    )
    logger.info("Sentry monitoring initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load environment once on startup
    try:
        load_dotenv(find_dotenv(usecwd=True))
        logger.info("Environment loaded from .env")
    except Exception as e:
        logger.warning(f".env not loaded: {e}")
    
    # Initialize Sentry monitoring
    init_sentry()
    
    logger.info("Starting SquawkAI server...")
    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app.state.graph_app = build_graph(checkpointer)
        yield
    logger.info("Shutting down SquawkAI server...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Squawk AI LangGraph FastAPI Server Running"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    try:
        # Basic health checks
        checks = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "graph_app_initialized": app.state.graph_app is not None,
            "conversation_threads_count": len(conversation_threads),
            "background_tasks_count": len(background_task_references)
        }
        
        # Check if we can connect to database (basic check)
        try:
            # This is a lightweight check - just verify we can import and access config
            from config import supabase_CHAT
            checks["database_accessible"] = True
        except Exception as e:
            checks["database_accessible"] = False
            checks["database_error"] = str(e)
        
        return checks
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }


@app.get("/create", response_model=ConversationResponse)
async def create_conversation(request: Request):
    graph_app = app.state.graph_app
    if graph_app is None:
        raise HTTPException(status_code=500, detail="Application is not initialized.")

    conversation_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    conversation_threads[conversation_id] = thread_id

    first = request.headers.get("X-User-First-Name", "")
    last = request.headers.get("X-User-Last-Name", "")
    org = request.headers.get("X-User-Org", None)
    logger.debug(f"X-User-Org header: {org!r}")

    # Add Sentry context for better error tracking
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("conversation_id", conversation_id)
        scope.set_tag("org", org or "unknown")
        scope.set_context("user", {
            "first_name": first,
            "last_name": last,
            "org": org
        })

    log_conversation(conversation_id, thread_id)

    logger.info(f"Creating conversation {conversation_id} for user: {first} {last}")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    async def run_initial_graph():
        try:
            await graph_app.ainvoke(
                {
                    "conversation_id": conversation_id,
                    "org": org,
                    "messages": [],
                    "qa_history": []
                },
                config=config
            )
        except Exception as e:
            logger.error(f"Initial graph error: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
        finally:
            background_task_references.discard(task)

    task = asyncio.create_task(run_initial_graph())
    background_task_references.add(task)

    return {"conversation_id": conversation_id, "org": org}


@app.get("/poll/{conversation_id}", response_model=PollResponse)
async def poll_state(conversation_id: str):
    graph_app = app.state.graph_app
    if graph_app is None:
        raise HTTPException(status_code=500, detail="Application is not initialized.")

    thread_id = conversation_threads.get(conversation_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        current_state_snapshot = await graph_app.aget_state(config)
        state_values = current_state_snapshot.values

        messages = state_values.get("messages", [])
        def convert_role(lc_type: str) -> str:
            if lc_type == "human":
                return "user"
            elif lc_type == "ai":
                return "assistant"
            return lc_type  # fallback

        formatted_messages = [
            {
                "role": convert_role(msg.type),
                "content": msg.content,
                **msg.additional_kwargs
            }
            for msg in messages
        ]

        paused = state_values.get("paused_for_input", False)
        stopped = _is_graph_finished(current_state_snapshot)

        return {
            "messages": formatted_messages,
            "paused_for_input": paused,
            "stopped": stopped,
            "current_state": state_values
        }
    except Exception as e:
        logger.error(f"Error polling state: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reply/{conversation_id}")
async def post_user_response(conversation_id: str, user_response: UserResponse):
    graph_app = app.state.graph_app
    if graph_app is None:
        raise HTTPException(status_code=500, detail="Application is not initialized.")

    thread_id = conversation_threads.get(conversation_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        current_state_snapshot = await graph_app.aget_state(config)
        state_values = current_state_snapshot.values

        if not state_values.get("paused_for_input", False):
            raise HTTPException(status_code=400, detail="Not waiting for user input.")

        update_payload = {
            "messages": [HumanMessage(content=user_response.user_response)],
            "paused_for_input": False
        }

        log_last_message({
            "conversation_id": conversation_id,
            "messages": [HumanMessage(content=user_response.user_response)]
        })

        if state_values.get("aircraft_type") is not None and state_values.get("user_issue") is None:
            update_payload["user_issue"] = user_response.user_response
            update_payload["messages"].append(
                AIMessage(
                    content="Generating preliminary fault tree...",
                    additional_kwargs={"ui_hint": "running"}
                )
            )
        elif state_values.get("user_issue") and not EXIT_PATTERN.match(user_response.user_response):
            update_payload["messages"].append(
                AIMessage(
                    content="Updating hypotheses...",
                    additional_kwargs={"ui_hint": "running"}
                )
            )
        elif EXIT_PATTERN.match(user_response.user_response):
            update_payload["messages"].append(
                AIMessage(
                    content="Generating report...",
                    additional_kwargs={"ui_hint": "running"}
                )
            )

        await graph_app.aupdate_state(config, update_payload)

        async def resume_graph():
            try:
                await graph_app.ainvoke(None, config=config)
            except Exception as e:
                logger.error(f"Resume graph error: {e}", exc_info=True)
            finally:
                background_task_references.discard(task)

        task = asyncio.create_task(resume_graph())
        background_task_references.add(task)

        return {"status": "User response received."}
    except Exception as e:
        logger.error(f"Reply error: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/{conversation_id}")
async def post_feedback(conversation_id: str, body: FeedbackBody):
    """
    Add feedback to the most recent conversation_summary row for this conversation.
    Never inserts new rows.
    """
    try:
        updated_id = log_feedback(
            conversation_id=conversation_id,
            thumb_feedback=body.thumb_feedback,
        )
        return {"status": "Feedback saved"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Feedback save failed for {conversation_id}: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail="Failed to save feedback")
