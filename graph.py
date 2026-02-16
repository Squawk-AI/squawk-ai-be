import operator
from typing import Annotated, List, Optional, TypedDict, Dict, Any
import json
import re
from config import supabase_OV, supabase_VS
from prompt import (
    base_prompt,
    fault_tree_prompt,
    question_prompt,
    update_tree_prompt,
    decision_prompt,
    diagnostic_pattern_block_decide,
    summary_prompt,
)
from llm_select import get_llm_for_node, embeddings, TABLE_NAME
from utils import retry_with_backoff, tavily_search
import logging
logger = logging.getLogger(__name__)
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

import sentry_sdk
from langsmith import traceable
from db_logging import log_last_message, update_conversation_status, insert_summary, set_conversation_aircraft_and_running
from utils import send_email


#Langgraph state
class GraphState(TypedDict):
    aircraft_type: Optional[str]
    aircraft_overview: Optional[dict]
    org: Optional[str]
    tail_number: Optional[str]
    conversation_id: str
    config_details: Optional[str]

    user_issue: Optional[str]
    research: Optional[Dict[str, Any]]
    diagnostic_memo: Optional[str]
    past_questions: Optional[List[str]]
    fault_tree: Optional[dict]
    
    paused_for_input: Optional[bool]
    should_exit: Optional[bool]
    messages: Annotated[List[AnyMessage], operator.add]
    qa_history: Annotated[List[Dict[str, str]], operator.add]
    early_exit: Optional[bool]


##Query vector store
def to_pgvector(vec):
    return f"'[{','.join(f'{v:.6f}' for v in vec)}]'::vector"

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: ask_for_aircraft_type
# ─────────────────────────────────────────────────────────────────────────────
def ask_for_aircraft_type(state: GraphState) -> dict:
    org = state.get("org")

    #fetch tail numbers for the org if it exists and type if not
    try:
        if org:
            response = (
                supabase_OV.table("org_aircraft")
                .select("tail_number")
                .eq("org", org)
                .execute()
            )

            tail_numbers = [row["tail_number"] for row in response.data or []]
            tail_numbers.sort()

            if tail_numbers:
                message = AIMessage(
                    content="Select an aircraft to report an issue on:",
                    additional_kwargs={
                        "ui_hint": "pill-select",
                        "options": tail_numbers
                    }
                )
                return {
                    "messages": [message],
                    "paused_for_input": True
                }

        type_response = (
            supabase_OV.table("org_aircraft")
            .select("aircraft_type")
            .is_("org", None)
            .execute()
        )

        aircraft_types = sorted({row["aircraft_type"] for row in type_response.data if row.get("aircraft_type")})

        if aircraft_types:
            message = AIMessage(
                content="Select an aircraft type to begin:",
                additional_kwargs={
                    "ui_hint": "pill-select",
                    "options": aircraft_types
                }
            )
        else:
            message = AIMessage(content="No aircraft types found in the system.")

        log_last_message({**state, "messages": [message]})
        return {
            "messages": [message],
            "paused_for_input": True
        }

    except Exception as e:
        message = AIMessage(content=f"Error retrieving aircraft list: {e}")
        log_last_message({**state, "messages": [message]})
        return {
            "messages": [message],
            "paused_for_input": True
        }


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: store_aircraft_selection
# ─────────────────────────────────────────────────────────────────────────────
def store_aircraft_selection(state: GraphState) -> dict:
    messages = state.get("messages", [])
    org = state.get("org")
    latest_human_message = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )

    if not latest_human_message:
        return {}

    user_input = latest_human_message.content.strip()

    # Case 1: If org is set, the user selected a tail number
    if org:
        try:
            result = (
                supabase_OV.table("org_aircraft")
                .select("tail_number", "aircraft_type", "config_details")
                .eq("org", org)
                .eq("tail_number", user_input)
                .single()
                .execute()
            )
            if result.data:
                set_conversation_aircraft_and_running(state["conversation_id"], result.data["aircraft_type"], result.data["tail_number"])
                return {
                    "messages": [],
                    "tail_number": result.data["tail_number"],
                    "aircraft_type": result.data["aircraft_type"],
                    "config_details": result.data["config_details"]
                }
        except Exception as e:
            logger.warning(f"Tail number lookup failed for org {org}: {e}")

    # Case 2: org is blank, user selected a public aircraft type
    set_conversation_aircraft_and_running(state["conversation_id"], user_input, None)
    return {
        "messages": [],
        "aircraft_type": user_input
    }


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: load_aircraft_overview_and_squawks
# ─────────────────────────────────────────────────────────────────────────────
def load_aircraft_overview(state: GraphState) -> dict:
    aircraft_type = state.get("aircraft_type")
    config_details = state.get("config_details")
    if not aircraft_type:
        return {}

    try:
        response = (
            supabase_OV.table("aircraft_overviews")
            .select("*")
            .eq("aircraft_type", aircraft_type)
            .limit(1)
            .execute()
        )
        records = response.data or []

        if not records: #*Need to generate overview from internet is aircraft type isn't in DB*
            out = {
                "messages": [AIMessage(content=f"No structured overview found for aircraft type: {aircraft_type}. General knowledge will be used.", additional_kwargs={"ui_hint": "status"})],
                "paused_for_input": True
            }
            log_last_message({**state, **out})
            return out

        overview = records[0]
        if config_details:
            overview["config_details"] = config_details
        
        # concise debug: count config fields if present
        logger.info(f"[overview] loaded for {aircraft_type}; has_config_details={'config_details' in overview}")

        out = {
            "messages": [AIMessage(content=f"Loaded structured overview for aircraft type: {aircraft_type}.", additional_kwargs={"ui_hint": "status"})],
            "aircraft_overview": overview
        }
        log_last_message({**state, **out})
        return out

    except Exception as e:
        out = {
            "messages": [AIMessage(content=f"Error retrieving aircraft overview: {str(e)}", additional_kwargs={"ui_hint": "status"})],
            "paused_for_input": True
        }
        log_last_message({**state, **out})
        return out

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: ask_for_issue
# ─────────────────────────────────────────────────────────────────────────────
def ask_for_issue(state: GraphState) -> dict:
    message = AIMessage(content="Briefly describe the issue you are experiencing with the aircraft.")
    
    log_last_message({**state, "messages": [message]})
    
    return {
        "messages": [message],
        "paused_for_input": True
    }

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: do_research
# ─────────────────────────────────────────────────────────────────────────────
@traceable(tags=["retrieval", "RAG"])
def do_research(state: GraphState) -> dict:
    user_issue = state.get("user_issue") or ""
    aircraft_type = state.get("aircraft_type")
    aircraft_overview = state.get("aircraft_overview")
    org = state.get("org")
    conversation_id = state.get("conversation_id")

    # Add Sentry context for research operations
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("node", "do_research")
        scope.set_tag("aircraft_type", aircraft_type or "unknown")
        scope.set_tag("org", org or "unknown")
        scope.set_context("research", {
            "user_issue_length": len(user_issue),
            "has_aircraft_overview": bool(aircraft_overview),
            "conversation_id": conversation_id
        })

    if not aircraft_overview:
        return {
            "messages": [AIMessage(content="Missing aircraft overview.")],
            "paused_for_input": True
        }
    if not aircraft_type:
        return {
            "messages": [AIMessage(content="Missing aircraft type.")],
            "paused_for_input": True
        }

    ##*OpenAI rewrites the user's issue before RAG, we should consider the same
    ##*Also should load any specs/overviews as just documents, not embeddings
    ##*Add lookup of past Squawks from CSV

    # 1) Embed query
    try:
        vec = retry_with_backoff(lambda: embeddings.embed_query(user_issue), retries=3, initial_delay=1.0)
    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        return {
            "messages": [AIMessage(content=f"Embedding failed: {e}")],
            "paused_for_input": True
        }

    query_vector_sql = to_pgvector(vec)

    # 2) Manuals / documentation vector search
    try:
        manual_sql = f"""
            SELECT id, content, page_number, section_title, aircraft_model, document_id, system_category,
                   embedding <#> {query_vector_sql} AS distance
            FROM {TABLE_NAME}
            WHERE aircraft_model = '{aircraft_type}'
            ORDER BY distance ASC
            LIMIT 20;
        """
        manual_resp = retry_with_backoff(lambda: supabase_VS.rpc("execute_sql", {"sql": manual_sql}).execute(), retries=3, initial_delay=1.0)
        manual_rows = manual_resp.data or []
        manual_context = "\n".join([row["content"] for row in manual_rows])
        logger.info(f"[manuals] rows={len(manual_rows)}")

    except Exception as e:
        logger.error(f"SQL vector search failed: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        return {
            "messages": [AIMessage(content=f"SQL vector search failed: {e}")],
            "paused_for_input": True
        }

    # 3) Past squawks (same style as your other backend)
    past_squawks: List[dict] = []
    past_squawks_summary = ""
    if org and aircraft_type:
        try:
            squawk_sql = f"""
                SELECT discrepancy_id, issue, corrective_action, tail_number, aircraft_type, status,
                    issue_embedding_vector <#> {query_vector_sql} AS distance
                FROM squawk_text_embedding_3_small
                WHERE org = '{org}'
                AND aircraft_type = '{aircraft_type}'
                AND issue_embedding_vector <#> {query_vector_sql} <= -0.5
                ORDER BY distance ASC
                LIMIT 10;
            """
            squawk_resp = retry_with_backoff(lambda: supabase_VS.rpc("execute_sql", {"sql": squawk_sql}).execute(), retries=3, initial_delay=1.0)
            matches = squawk_resp.data or []
            past_squawks = [
                {
                    "issue": m["issue"],
                    "corrective_action": m["corrective_action"],
                    "tail_number": m["tail_number"],
                    "aircraft_type": m["aircraft_type"],
                    "status": m["status"],
                    "distance": m["distance"],
                }
                for m in matches
            ]
            if past_squawks:
                count = len(past_squawks)
                past_squawks_summary = f"Past related squawks for this fleet: {count}"
            else:
                past_squawks_summary = "No related squawks found for this fleet."
            logger.info(f"[squawks] matches={len(past_squawks)}")

        except Exception as e:
            # Non-fatal: continue without past squawks
            past_squawks_summary = "[lookup failed]"
            logger.error(f"[squawks] lookup failed: {e}")
            sentry_sdk.capture_exception(e)

    # 4) Tavily web search
    web_findings = tavily_search(user_issue, aircraft_type) if user_issue else {"note": "no user_issue text"}
    # Make it lighter/summarized for prompt
    web_summary = ""
    if isinstance(web_findings, dict) and not web_findings.get("error"):
        # Tavily returns fields like "answer" and "results" (array with title/url/content)
        ans = web_findings.get("answer")
        if ans:
            web_summary += f"Tavily Answer:\n{ans}\n\n"
        results = web_findings.get("results", [])[:5]
        if results:
            web_summary += "Top Web Results:\n" + "\n".join(
                f"- {r.get('title','(untitled)')}: {r.get('url','')}" for r in results
            )
        logger.info(f"[web] results={len(web_findings.get('results', []) if isinstance(web_findings, dict) else [])}")
    else:
        web_summary = f"[web search unavailable: {web_findings.get('error', 'unknown error')}]"
        logger.warning("[web] search unavailable")

    research_bundle = {
        "manual_context": manual_context,
        "past_squawks": past_squawks,
        "past_squawks_summary": past_squawks_summary or "[none found]",
        "web_findings": web_findings,
        "web_summary": web_summary,
    }

    return {
        "research": research_bundle,
        "paused_for_input": False
    }


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: build_fault_tree
# ─────────────────────────────────────────────────────────────────────────────
@traceable(tags=["LLM call"])
def build_fault_tree(state: GraphState) -> dict:
    user_issue = state.get("user_issue")
    aircraft_overview = state.get("aircraft_overview")
    research = state.get("research") or {}
    conversation_id = state.get("conversation_id")

    # Add Sentry context for fault tree generation
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("node", "build_fault_tree")
        scope.set_tag("aircraft_type", state.get("aircraft_type") or "unknown")
        scope.set_context("fault_tree", {
            "has_aircraft_overview": bool(aircraft_overview),
            "has_research": bool(research),
            "user_issue_length": len(user_issue) if user_issue else 0,
            "conversation_id": conversation_id
        })

    if not aircraft_overview:
        return {
            "messages": [AIMessage(content="Missing aircraft overview.")],
            "paused_for_input": True
        }

    if not research:
        return {
            "messages": [AIMessage(content="Research bundle not found. Run do_research first.")],
            "paused_for_input": True
        }

    manual_context = research.get("manual_context", "")
    past_squawks_summary = research.get("past_squawks_summary", "No related squawks found for this fleet.")
    web_summary = research.get("web_summary", "[no web findings]")

    prompt = base_prompt + fault_tree_prompt + f"""
Aircraft Overview:
{json.dumps(aircraft_overview, indent=2)}

Pilot Report:
{user_issue}

Context from manuals/vector search:
{manual_context}

Past Squawks & Actions:
{past_squawks_summary}

External Web Research (Tavily):
{web_summary}
"""

    llm = get_llm_for_node("build_fault_tree")
    msg_cls = HumanMessage if isinstance(llm, ChatAnthropic) else SystemMessage
    response = retry_with_backoff(lambda: llm.invoke([msg_cls(content=prompt)]), retries=3, initial_delay=1.0, operation_name="build_fault_tree.invoke")

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        fault_tree = json.loads(raw)
        return {
            "fault_tree": fault_tree,
            "paused_for_input": False
        }
    except Exception as e:
        logger.error(f"Fault tree generation failed: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        out = {
            "messages": [AIMessage(content=f"⚠️ Fault tree generation failed: {e}\n\nRaw output:\n{response.content}")],
            "paused_for_input": True
        }
        log_last_message({**state, **out})
        return out
        

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: generate_question
# ─────────────────────────────────────────────────────────────────────────────
@traceable(tags=["LLM call"])
def generate_question(state: GraphState) -> dict:
    aircraft_type = state.get("aircraft_type")
    aircraft_overview = state.get("aircraft_overview")
    fault_tree = state.get("fault_tree")
    diagnostic_memo = state.get("diagnostic_memo") or state.get("user_issue")
    past_questions = state.get("past_questions", []) or []
    qa_history = state.get("qa_history", [])

    if not fault_tree:
        return {
            "messages": [AIMessage(content="Fault tree is missing. Cannot generate questions.")],
            "paused_for_input": False,
            "past_questions": [],
        }

    qa_count = sum(1 for p in qa_history if "q" in p and "a" in p)

    context = {
        "aircraft_type": aircraft_type,
        "aircraft_overview": aircraft_overview,
        "diagnostic_memo": diagnostic_memo,
        "qa_count": qa_count,
        "fault_tree": fault_tree,
        "qa_history": qa_history
    }

    diagnostic_pattern_block = """
## Diagnostic Pattern: helpful for diagnosis and comprises the information maintenance needs:
- Context, Conditions & Timing: Phase of flight or event (climb, cruise, final, during pre-flight or post-flight), weather, and airport conditions.
- Pilot Actions: around the time of the event to diagnose/troubleshoot in flight, to reset/re-engage systems, or to address the issue, and the outcome of such actions.
- Annunciators/Warnings: Lights, messages, aural alerts, includes CAS messages.
- Sensory Input: Sounds, vibrations, smells.
- Aircraft Behavior: Pitch, roll, yaw, control feel, trim behavior.
- Aircraft History: Recent maintenance or previous similar occurrences, that a pilot (not the mechanic) would reasonably have knowledge about.
- State of Primary Involved Systems or Other Systems: Attempts to reset/re-engage systems, status of other instruments.
- Important Info for Future or Tricky Issues: Anything maintenance might like to keep an eye on, even if the issue resolved itself; the fact it occurred at all warrants collecting some info.
- Catch all question to be asked at the end: "Are there any other observations you believe may be relevant? For example, anything notable about the __________ (ask about categories that have not been asked about yet)?"
""".strip()

    prompt = (
        base_prompt
        + question_prompt
        + f"\nPrevious questions:\n{past_questions}"
        + f"\ncontext:\n{json.dumps(context, indent=2)}"
        + "\n" + diagnostic_pattern_block
    )

    llm = get_llm_for_node("generate_question")
    msg_cls = HumanMessage if isinstance(llm, ChatAnthropic) else SystemMessage
    response = retry_with_backoff(lambda: llm.invoke([msg_cls(content=prompt)]), retries=3, initial_delay=1.0, operation_name="generate_question.invoke")

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        questions = parsed.get("questions", [])

        # If model suggests exit or yields no questions, fall back to a catch-all question
        if not questions:
            questions = [
                "Are there any other observations you believe may be relevant? For example, anything notable about the systems or conditions we haven't covered yet?"
            ]

        # Ask ONLY the first question; store the remainder in state
        first = questions[0].strip() if isinstance(questions[0], str) else str(questions[0])
        message = AIMessage(
            content=first + (" Type 'exit' at any time to submit the issue without further details." if qa_count == 0 else "")
        )
        log_last_message({**state, "messages": [message]})

        logger.debug(f"[generate_question] questions={questions}")

        return {
            "messages": [message],
            "past_questions": [q if isinstance(q, str) else str(q) for q in questions[1:]],
            "paused_for_input": True,
            "should_exit": False,
        }

    except Exception as e:
        message = AIMessage(content=f"❌ Question generation failed: {e}\n\nRaw output:\n{response.content}")
        log_last_message({**state, "messages": [message]})
        return {"messages": [message], "paused_for_input": True}


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: update_tree
# ─────────────────────────────────────────────────────────────────────────────
@traceable(tags=["LLM call"])
def update_tree(state: GraphState) -> dict:
    messages = state.get("messages", [])
    fault_tree = state.get("fault_tree")
    should_exit = state.get("should_exit", False)
    qa_history = state.get("qa_history", [])

    if not fault_tree:
        return {}

    latest_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if latest_human and latest_human.content.strip().lower() in {"exit","/exit","quit","end"}:
        logger.info("[exit] user requested early exit")
        return {
        "paused_for_input": False,
        "should_exit": True,
        "qa_history": [],
        "early_exit": True,
        }
    latest_ai = next((m for m in reversed(messages[:-1]) if isinstance(m, AIMessage)), None) if latest_human else None
    qa_append = []
    if latest_ai and latest_human:
        qa_append = [{"q": latest_ai.content.strip(), "a": latest_human.content.strip()}]
    else:
        # Guard first update: skip when there is no fresh Q/A pair
        logger.info("[update_tree] no latest Q/A pair; skipping tree update this turn")
        return {
            "paused_for_input": False,
            "should_exit": should_exit,
            "qa_history": [],
        }


    prompt = base_prompt + update_tree_prompt + (
        f"Previous Fault Tree:\n{json.dumps(fault_tree, indent=2)}\n\n"
        f"Most recent question and answer (use ONLY this to update):\n{json.dumps(qa_append, indent=2)}\n\n"
        "Important: Update confidence ONLY for node(s) directly affected by the most recent answer. "
        "Leave EVERY OTHER node's confidence EXACTLY unchanged (blank stays blank). Do NOT globally re-rate."
    )

    llm = get_llm_for_node("update_tree")
    msg_cls = HumanMessage if isinstance(llm, ChatAnthropic) else SystemMessage
    response = retry_with_backoff(lambda: llm.invoke([msg_cls(content=prompt)]), retries=3, initial_delay=1.0, operation_name="update_tree.invoke")

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        updated_tree = parsed.get("updated_tree", fault_tree)

        return {
            "fault_tree": updated_tree,
            "paused_for_input": False,
            "should_exit": should_exit,
            "qa_history": qa_append,
        }

    except Exception as e:
        out = {
            "messages": [AIMessage(content=f"❌ Tree update failed: {e}\n\nRaw output:\n{response.content}")],
            "paused_for_input": True
        }
        log_last_message({**state, **out})
        return out
        

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: decide_next_action
# ─────────────────────────────────────────────────────────────────────────────
@traceable(tags=["LLM call"])
def decide_next_action(state: GraphState) -> dict:
    fault_tree = state.get("fault_tree")
    qa_history = state.get("qa_history", [])
    early_exit = state.get("early_exit", False)
    predecided_exit = state.get("should_exit", False)

    if not fault_tree:
        return {
            "messages": [AIMessage(content="Fault tree is missing. Cannot decide next action.")],
            "paused_for_input": False,
        }

    # Respect any prior hard exit (e.g., user typed 'exit')
    if early_exit or predecided_exit:
        logger.info("[decision] honoring prior exit flag (early_exit or should_exit set)")
        return {
            "should_exit": True,
            "paused_for_input": False,
        }

    qa_count = sum(1 for p in qa_history if "q" in p and "a" in p)

    # Fast-path: on the first question cycle, skip LLM and continue the interview
    if qa_count <= 1:
        logger.info("[decision] skipping LLM on first Q&A; continuing interview")
        return {
            "should_exit": False,
            "paused_for_input": False,
        }

    context = {
        "qa_count": qa_count,
        "fault_tree": fault_tree,
        "qa_history": qa_history,
    }

    prompt = (
        base_prompt
        + decision_prompt
        + f"\ncontext:\n{json.dumps(context, indent=2)}"
        + "\n" + diagnostic_pattern_block_decide
    )

    llm = get_llm_for_node("generate_question")  # keep same model family/speed profile
    msg_cls = HumanMessage if isinstance(llm, ChatAnthropic) else SystemMessage
    response = retry_with_backoff(
        lambda: llm.invoke([msg_cls(content=prompt)]),
        retries=3,
        initial_delay=1.0,
        operation_name="decide_next_action.invoke",
    )

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        should_exit = bool(parsed.get("exit", False))
        reason = parsed.get("reasoning") or ""
        logger.info(f"[decision] exit={should_exit} qa_count={qa_count} reason={reason[:200]}")

        return {
            "should_exit": should_exit,
            "paused_for_input": False,
        }

    except Exception as e:
        message = AIMessage(content=f"❌ Decision step failed: {e}\n\nRaw output:\n{response.content}")
        log_last_message({**state, "messages": [message]})
        return {"messages": [message], "paused_for_input": True}


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: summarize_issue
# ─────────────────────────────────────────────────────────────────────────────
@traceable(tags=["LLM call"])
def summarize_issue(state: GraphState) -> dict:
    issue = state.get("user_issue")
    prior_diagnostic_memo = state.get("diagnostic_memo") or ""
    fault_tree = state.get("fault_tree")
    should_exit = state.get("should_exit", False)   
    qa_history = state.get("qa_history", [])
    early_exit = state.get("early_exit", False)
    research = state.get("research", {})

    if not fault_tree:
        return {}

    prompt = base_prompt + summary_prompt + (
        f"Reported Issue:\n{issue}\n\n"
        f"Previous Memo:\n{prior_diagnostic_memo}\n\n"
        f"Fault Tree:\n{json.dumps(fault_tree, indent=2)}\n\n"
        f"Q&A History:\n{json.dumps(qa_history, indent=2)}\n\n"
        f"Past Related Squawks in this Fleet:\n{research.get('past_squawks_summary', 'No related squawks found for this fleet.')}"
        ) + ("Pilot requested to exit the interview without providing further detail. Do not mention this in the summary, this is just for your information so you know why there is very little detail available. Still include likely causes and recommended actions but keep them broad and brief, and based solely on the fault tree." if early_exit else "")

    llm = get_llm_for_node("summarize_issue")
    msg_cls = HumanMessage if isinstance(llm, ChatAnthropic) else SystemMessage
    response = retry_with_backoff(lambda: llm.invoke([msg_cls(content=prompt)]), retries=3, initial_delay=1.0, operation_name="summarize_issue.invoke")

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        memo = parsed.get("memo", prior_diagnostic_memo)
        likely_causes = parsed.get("likely_causes", [])
        recommended_actions = parsed.get("recommended_actions", [])

        out = {
            "diagnostic_memo": memo.strip(),
            "paused_for_input": False,
            "should_exit": should_exit,
        }

        if should_exit:
            full_memo = f"""{memo.strip()}

### Likely Root Causes
{chr(10).join(f"{i}. {c}" for i, c in enumerate(likely_causes, 1)) if likely_causes else "None identified."}

### Recommended Actions
{chr(10).join(
    f"{i}. {a['action']}"
    + (f" — {a['manual_component']}" if a.get('manual_component') else "")
    + (f" ({a['reference_section']})" if a.get('reference_section') else "")
    for i, a in enumerate(recommended_actions, 1)
) if recommended_actions else "No specific actions recommended."}
""".strip()

            out["diagnostic_memo"] = full_memo

        return out

    except Exception as e:
        message = AIMessage(content=f"❌ Memo update failed: {e}\n\nRaw output:\n{response.content}")
        log_last_message({**state, "messages": [message]})
        return {
            "messages": [message],
            "paused_for_input": True
        }


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Node: wrapup_report
# ─────────────────────────────────────────────────────────────────────────────
def wrapup_report(state: GraphState) -> dict:
    aircraft_type = state.get("aircraft_type", "Unknown")
    user_issue = state.get("user_issue", "Not specified")
    diagnostic_memo = state.get("diagnostic_memo", "")
    conversation_id = state.get("conversation_id", "N/A")
    aircraft_tail_number = state.get("tail_number", "N/A")
    research = state.get("research", {})
    early_exit = state.get("early_exit", False)

    # Add Sentry context for report generation
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("node", "wrapup_report")
        scope.set_tag("aircraft_type", aircraft_type)
        scope.set_tag("conversation_id", conversation_id)
        scope.set_context("report", {
            "early_exit": early_exit,
            "has_tail_number": aircraft_tail_number != "N/A",
            "diagnostic_memo_length": len(diagnostic_memo),
            "has_research": bool(research)
        })

    past_squawks_summary = research.get("past_squawks_summary", "No related squawks found for this fleet.")

    tail_line = f"**Tail Number:** {aircraft_tail_number}\n\n" if aircraft_tail_number and aircraft_tail_number != "N/A" else ""
    early_exit_line = "**This report was generated without further input from the pilot, as they requested to exit the interview early.** Any information provided below is based on general knowledge of the issue.\n\n" if early_exit else ""

    report = f"""# {user_issue} on {aircraft_type}

{tail_line}**Aircraft type:** {aircraft_type}  
**Conversation ID:** {conversation_id}  

---

### Reported Issue

{user_issue}

{early_exit_line}### Diagnostic Summary

{diagnostic_memo}

### Past Squawks and Actions

{past_squawks_summary}

---

_This report was generated by [Squawk AI](https://bettersquawk.com)._
"""

    report_status_message = AIMessage(content="Report sent to maintenance", additional_kwargs={"ui_hint": "status"})
    report_message = AIMessage(content=report)
    log_last_message({**state, "messages": [report_message]})

    try:
        update_conversation_status(conversation_id)
        insert_summary(conversation_id, report)
    except Exception as e:
        logger.error(f"Failed to update status or insert summary: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)

    try:
        send_email(user_issue, report)
    except Exception as e:
        logger.error(f"Failed to send email: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)

    return {
        "messages": [report_message, report_status_message],
        "paused_for_input": False
    }


def build_graph(checkpointer):
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("ask_for_aircraft_type", ask_for_aircraft_type)
    workflow.add_node("store_aircraft_selection", store_aircraft_selection)
    workflow.add_node("load_aircraft_overview", load_aircraft_overview)
    workflow.add_node("ask_for_issue", ask_for_issue)
    workflow.add_node("do_research", do_research)
    workflow.add_node("build_fault_tree", build_fault_tree)
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("update_tree", update_tree)
    workflow.add_node("decide_next_action", decide_next_action)
    workflow.add_node("summarize_issue", summarize_issue)
    workflow.add_node("wrapup_report", wrapup_report)

    # Entry + Setup
    workflow.set_entry_point("ask_for_aircraft_type")
    workflow.add_edge("ask_for_aircraft_type", "store_aircraft_selection")
    workflow.add_edge("store_aircraft_selection", "load_aircraft_overview")
    workflow.add_edge("load_aircraft_overview", "ask_for_issue")
    workflow.add_edge("ask_for_issue", "do_research")
    workflow.add_edge("do_research", "build_fault_tree")
    workflow.add_edge("build_fault_tree", "generate_question")
    workflow.add_edge("generate_question", "update_tree")
    workflow.add_edge("update_tree", "decide_next_action")
    
    #Diagnostic loop
    workflow.add_conditional_edges(
        "decide_next_action",
        lambda state: "summarize_issue" if state.get("should_exit", False) else "generate_question",
        {
            "generate_question": "generate_question",
            "summarize_issue": "summarize_issue",
        }
    )

    # From summarize_issue, proceed directly to wrapup
    workflow.add_edge("summarize_issue", "wrapup_report")
    workflow.add_edge("wrapup_report", END)
    
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "store_aircraft_selection",
            "do_research",
            "update_tree"
        ],
    )

