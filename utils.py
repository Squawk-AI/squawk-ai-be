import os, json, re, time, random, logging
import requests, smtplib
from typing import Optional, TypedDict, Dict, Any, List, Callable, Tuple, Type
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import markdown
import sentry_sdk

logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: Callable[[], Any],
    *,
    retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 8.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.25,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    operation_name: str | None = None,
) -> Any:
    attempt = 0
    delay = max(0.0, float(initial_delay))
    while True:
        try:
            return func()
        except exceptions as exc:
            attempt += 1
            if attempt >= retries:
                logger.error(
                    f"Retry exhausted{f' for {operation_name}' if operation_name else ''}: {exc}",
                    exc_info=True,
                )
                sentry_sdk.capture_exception(exc)
                raise
            sleep_for = min(delay, max_delay) * (1.0 + random.uniform(-jitter, jitter))
            logger.warning(
                f"Retrying{f' {operation_name}' if operation_name else ''} in {sleep_for:.2f}s (attempt {attempt}/{retries-1}) due to: {exc}"
            )
            time.sleep(max(0.0, sleep_for))
            delay *= backoff_factor

def tavily_search(user_issue: str, aircraft_type: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
    """
    Tavily web search with aircraft context.
    Requires env var TAVILY_API_KEY.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "Missing TAVILY_API_KEY"}

    # Build a richer query string
    parts = []
    if aircraft_type:
        parts.append(f"Aircraft type: {aircraft_type}")
    if user_issue:
        parts.append(f"Issue: {user_issue}")
    parts.append(f"Context: pilot reported, troubleshooting guidance")
    query = " | ".join(parts)

    try:
        def _do_request():
            resp = requests.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": "advanced",
                    "include_images": False,
                    "max_results": max_results,
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()

        return retry_with_backoff(_do_request, retries=3, initial_delay=1.0, operation_name="tavily_search")
    except Exception as e:
        return {"error": f"Tavily search failed: {e}"}


# --- EMAIL UTILITIES (moved from email_utils.py) ---
def _load_smtp_config() -> Dict[str, Any]:
    return {
        "server": os.environ.get("SMTP_SERVER"),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "username": os.environ.get("SMTP_USERNAME"),
        "password": os.environ.get("SMTP_PASSWORD"),
        "address": os.environ.get("SMTP_ADDRESS"),
    }


def send_email(subject: str, markdown_text: str, recipient: str = "mechanic@bettersquawk.com") -> None:
    # Check if email is enabled for this environment
    email_env = os.environ.get("EMAIL_ENV", "dev").lower()
    if email_env != "prod":
        logger.info(f"Email sending disabled (EMAIL_ENV={email_env}). Would send to {recipient}: {subject}")
        return
    
    smtp = _load_smtp_config()
    required = ["server", "username", "password", "address"]
    missing = [k for k in required if not smtp.get(k)]
    if missing:
        logger.error(f"SMTP not configured; missing: {', '.join(missing)}")
        raise ValueError(f"Missing required SMTP environment variables: {', '.join(missing)}")

    body_html = markdown.markdown(markdown_text)

    msg = MIMEMultipart()
    msg['From'] = smtp['address']
    msg['To'] = recipient
    msg['Subject'] = Header(subject, 'utf-8')
    msg.attach(MIMEText(body_html, 'html', 'utf-8'))

    def _send():
        with smtplib.SMTP(smtp['server'], smtp['port']) as server:
            server.starttls()
            server.login(smtp['username'], smtp['password'])
            server.send_message(msg)
        return True

    retry_with_backoff(_send, retries=3, initial_delay=1.0, operation_name="smtp_send")
    logger.info(f"Email sent successfully to {recipient}: {subject}")