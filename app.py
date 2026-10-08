import os
import json
import logging
import re
from typing import Optional

try:
    from fastapi import FastAPI  # type: ignore[import-not-found]
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
    from fastapi.responses import FileResponse  # type: ignore[import-not-found]
except ImportError:
    FastAPI = None
    CORSMiddleware = None
    FileResponse = None

try:
    from pydantic import BaseModel  # type: ignore[import-not-found]
except ImportError:
    class BaseModel:  # type: ignore[no-redef]
        pass

try:
    import ollama  # type: ignore[import-not-found]
except ImportError:
    ollama = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grass-whisper")

app = FastAPI(title="GrassWhisper", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONTACTS_FILE = os.path.join(APP_DIR, "contacts.json")

CONVERSATION_WINDOW_SECONDS = 120
EVENT_WINDOW_SECONDS = 300
REPEATED_CONTACT_WINDOW_SECONDS = 300


def load_contacts():
    if os.path.exists(CONTACTS_FILE):
        try:
            with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return ["Mom", "Sarah (Boss)", "Spouse"]


def save_contacts_to_disk(contacts):
    try:
        with open(CONTACTS_FILE, "w", encoding="utf-8") as f:
            json.dump(contacts, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving contacts: {e}")


session_data = {
    "walk_active": False,
    "walk_context": "Walking outdoors in nature. Only interrupt for life safety, urgent family issues, or critical work server outages.",
    "muted_count": 0,
    "urgent_count": 0,
    "history": [],
    "audio_privacy_mode": "safe",
    "vip_contacts": load_contacts(),
    "merge_count": 0,
}


class ContactRequest(BaseModel):
    name: str


class WalkContextRequest(BaseModel):
    context: str


class WalkSetRequest(BaseModel):
    active: bool
    context: str = ""


class NotificationItem(BaseModel):
    app_name: str
    sender: str
    message: str


class AudioPrivacyRequest(BaseModel):
    mode: str


def get_available_model():
    if ollama is None:
        return "offline-rules"

    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]

        for preferred in [
            "llama3.2:1b",
            "llama3.2:latest",
            "qwen2.5:0.5b",
            "gemma4:e2b",
        ]:
            for name in model_names:
                if preferred in name:
                    return name

        if model_names:
            return model_names[0]

    except Exception as e:
        logger.warning(f"Could not list models: {e}")

    return "llama3.2:1b"


def make_audio_whisper(
    notif: NotificationItem,
    urgency_score: int,
    override_text: str = "",
    extra_context: str = "",
) -> str:
    sender = notif.sender.strip()
    app_name = notif.app_name.strip() or "an app"
    source = sender or app_name

    banking_terms = [
        "bank",
        "banksecure",
        "upi",
        "wallet",
        "finance",
        "credit",
        "debit",
        "payment",
    ]

    privacy_mode = session_data["audio_privacy_mode"]

    if urgency_score < 6:
        return ""

    if override_text:
        if privacy_mode == "full":
            return override_text

        if override_text.lower().endswith("please check when safe."):
            return override_text

        return f"{override_text} Please check when safe."

    if privacy_mode == "full":
        prefix = "Critical alert" if urgency_score >= 9 else "Important message"
        return f"{prefix} from {source}: {notif.message[:180]}"

    if any(term in f"{app_name} {sender}".lower() for term in banking_terms):
        if urgency_score >= 9:
            return "Critical notification from your banking app. Please check your phone."

        return "Important notification from your banking app. Please check your phone."

    if urgency_score >= 9:
        return f"Critical notification from {source}. Please check your phone."

    return f"Important message from {source}. Please check when safe."


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_sender(sender: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", sender.lower()) or "unknown"


def message_has_call_signal(message: str) -> bool:
    text = message.lower()

    return any(
        term in text
        for term in [
            "missed call",
            "missed video call",
            "called you",
            "tried calling",
            "calling you",
            "voice call",
            "video call",
            "call back",
            "call me",
            "phone call",
        ]
    )


def message_topic(message: str) -> str:
    text = clean_text(message)
    lower = text.lower()

    category = None

    for name, terms in [
        ("meeting", ["meeting", "standup", "sync", "conference"]),
        ("exam", ["exam", "test", "quiz"]),
        ("appointment", ["appointment", "doctor", "dentist"]),
        ("flight", ["flight", "boarding", "gate"]),
        ("deadline", ["deadline", "due"]),
        ("interview", ["interview"]),
        ("delivery", ["delivery", "package", "shipment"]),
    ]:
        if any(term in lower for term in terms):
            category = name
            break

    time_match = re.search(
        r"\b(?:at|around)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        lower,
    )

    day = (
        "tomorrow"
        if "tomorrow" in lower
        else "today"
        if "today" in lower
        else None
    )

    if category == "meeting":
        if day and time_match:
            return f"{day}'s meeting at {time_match.group(0).split(' ', 1)[1]}"

        if day:
            return f"{day}'s meeting"

        if time_match:
            return f"meeting at {time_match.group(0).split(' ', 1)[1]}"

        return "the meeting"

    if category:
        if day:
            return f"{day}'s {category}"

        return f"the {category}"

    return "this update"


def event_tokens(message: str):
    text = clean_text(message).lower()

    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "is",
        "are",
        "be",
        "can",
        "you",
        "your",
        "we",
        "our",
        "this",
        "that",
        "with",
        "from",
        "about",
        "please",
        "join",
        "tomorrow",
        "today",
        "me",
        "i",
        "my",
        "it",
        "has",
        "have",
        "will",
        "could",
        "whether",
    }

    raw = re.findall(r"[a-z0-9']+", text)

    tokens = {
        token
        for token in raw
        if len(token) >= 3 and token not in stopwords
    }

    times = re.findall(
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        text,
    )

    for time_value in times:
        tokens.add(time_value.replace(" ", ""))

    for category, terms in [
        ("meeting", ["meeting", "standup", "sync", "conference"]),
        ("exam", ["exam", "test", "quiz"]),
        ("appointment", ["appointment", "doctor", "dentist"]),
        ("flight", ["flight", "boarding", "gate"]),
        ("deadline", ["deadline", "due"]),
        ("interview", ["interview"]),
    ]:
        if any(term in text for term in terms):
            tokens.add(category)

    return tokens


def event_signature(message: str):
    tokens = event_tokens(message)

    if not tokens:
        return None

    category = None

    for candidate in [
        "meeting",
        "exam",
        "appointment",
        "flight",
        "deadline",
        "interview",
    ]:
        if candidate in tokens:
            category = candidate
            break

    if not category:
        return None

    lower = message.lower()

    day = (
        "tomorrow"
        if "tomorrow" in lower
        else "today"
        if "today" in lower
        else ""
    )

    times = re.findall(
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        lower,
    )

    time_value = times[0].replace(" ", "") if times else ""

    return f"{category}|{day}|{time_value}"


def event_similarity(message_a: str, message_b: str) -> float:
    sig_a = event_signature(message_a)
    sig_b = event_signature(message_b)

    if not sig_a or not sig_b:
        return 0.0

    if sig_a == sig_b:
        return 1.0

    cat_a, day_a, time_a = sig_a.split("|")
    cat_b, day_b, time_b = sig_b.split("|")

    if cat_a != cat_b:
        return 0.0

    if day_a and day_b and day_a != day_b:
        return 0.0

    if time_a and time_b and time_a != time_b:
        return 0.0

    tokens_a = event_tokens(message_a)
    tokens_b = event_tokens(message_b)

    common = tokens_a & tokens_b
    union = tokens_a | tokens_b

    overlap = len(common) / len(union) if union else 0.0

    if time_a == time_b and time_a:
        overlap += 0.25

    if day_a == day_b and day_a:
        overlap += 0.15

    return min(1.0, overlap)


def recent_history(seconds: int):
    import time

    now = time.time()

    return [
        item
        for item in session_data["history"]
        if now - float(item.get("created_at", 0)) <= seconds
    ]


def conversation_context(notif: NotificationItem):
    target_app = clean_text(notif.app_name).lower()
    target_sender = normalize_sender(notif.sender)

    matches = []

    for item in recent_history(CONVERSATION_WINDOW_SECONDS):
        if (
            clean_text(item.get("app_name", "")).lower() == target_app
            and normalize_sender(item.get("sender", "")) == target_sender
        ):
            matches.append(item)

    matches.sort(
        key=lambda item: float(item.get("created_at", 0))
    )

    return matches


def related_event_context(notif: NotificationItem):
    candidates = []

    for item in recent_history(EVENT_WINDOW_SECONDS):
        score = event_similarity(
            notif.message,
            item.get("message", ""),
        )

        if score >= 0.45:
            candidates.append((score, item))

    candidates.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    return candidates


def repeated_contact_context(notif: NotificationItem):
    target_sender = normalize_sender(notif.sender)
    matches = []

    for item in recent_history(REPEATED_CONTACT_WINDOW_SECONDS):
        if normalize_sender(item.get("sender", "")) == target_sender:
            matches.append(item)

    return matches


def build_conversation_audio(
    notif: NotificationItem,
    count: int,
    latest_message: str,
    topic_hint: str,
    urgency_score: int,
):
    sender = (
        notif.sender.strip()
        or notif.app_name.strip()
        or "Someone"
    )

    privacy_mode = session_data["audio_privacy_mode"]

    if privacy_mode == "safe":
        if topic_hint and topic_hint != "this update":
            return f"{sender} sent {count} messages about {topic_hint}."

        return (
            f"{sender} sent {count} messages. "
            "Please check when safe."
        )

    latest = clean_text(latest_message)
    lower = latest.lower()

    for phrase in [
        "can you",
        "could you",
        "would you",
        "will you",
    ]:
        if phrase in lower:
            remainder = latest[
                lower.find(phrase) + len(phrase):
            ].strip(" .?!")

            if remainder:
                return (
                    f"{sender} sent {count} messages "
                    f"and is asking whether you can {remainder}."
                )

    if latest.endswith("?"):
        return (
            f"{sender} sent {count} messages "
            f"and is asking: {latest}"
        )

    return (
        f"{sender} sent {count} messages. "
        f"Latest: {latest[:180]}"
    )


def build_repeated_contact_audio(sender: str, count: int):
    if count >= 3:
        return (
            f"You have repeated contact from {sender}. "
            "This may be urgent."
        )

    return (
        f"You have repeated contact from {sender}. "
        "Please check when safe."
    )


def build_duplicate_event_audio(
    event_topic: str,
    full_mode: bool,
    latest_message: str,
):
    if full_mode:
        return (
            f"You have one update about {event_topic}: "
            f"{clean_text(latest_message)[:180]}"
        )

    return f"You have one update about {event_topic}."


def deterministic_classification(
    notif: NotificationItem,
    model_data: dict,
    matched_vip: Optional[str],
):
    text = clean_text(notif.message)
    lower = text.lower()
    app = clean_text(notif.app_name).lower()
    sender = clean_text(notif.sender)

    direct_apps = {
        "whatsapp",
        "telegram",
        "signal",
        "messenger",
        "facebook messenger",
        "sms",
        "messages",
        "imessage",
        "slack",
        "discord",
        "teams",
        "microsoft teams",
        "google chat",
        "chat",
        "email",
        "gmail",
        "outlook",
        "mail",
    }

    critical_terms = [
        "emergency",
        "accident",
        "hospital",
        "ambulance",
        "fire",
        "help me",
        "help!",
        "urgent",
        "critical",
        "p0",
        "security breach",
        "account hacked",
        "account compromised",
        "fraud alert",
        "card blocked",
        "card stolen",
        "otp",
        "one-time password",
        "one time password",
        "verification code",
        "security code",
        "login code",
        "authentication code",
        "confirm code",
        "missed call",
        "missed video call",
        "tried calling",
        "calling you",
        "call me now",
        "call me immediately",
        "need you now",
        "need you immediately",
    ]

    important_terms = [
        "deadline",
        "due today",
        "due tomorrow",
        "exam",
        "test",
        "interview",
        "meeting",
        "standup",
        "sync",
        "appointment",
        "doctor",
        "dentist",
        "flight",
        "boarding",
        "gate changed",
        "rescheduled",
        "schedule changed",
        "changed from",
        "moved to",
        "starts at",
        "starts in",
        "ends at",
        "can you",
        "could you",
        "would you",
        "will you",
        "please call",
        "call me",
        "need you",
        "reply",
        "respond",
        "confirm",
        "approve",
        "join",
        "available",
        "where are you",
        "are you coming",
        "are you free",
        "tomorrow at",
        "today at",
        "tonight",
        "by 5",
        "by 6",
        "by 7",
    ]

    low_terms = [
        "liked your post",
        "liked your photo",
        "reacted to your message",
        "started following you",
        "followed you",
        "new follower",
        "mentioned you in a story",
        "new reel",
        "new story",
        "suggested for you",
        "recommended for you",
        "hilarious dog video",
        "funny dog video",
        "look at this",
        "lol",
        "haha",
        "meme",
        "sale",
        "off your next order",
        "discount",
        "coupon",
        "promo",
        "promotional",
        "sponsored",
        "order shipped",
        "has shipped",
        "shipping confirmation",
        "delivery update",
        "package delivered",
        "sent 7 messages",
        "sent 8 messages",
        "sent you messages",
        "battery",
        "screen time",
        "weather update",
        "daily reminder",
        "weekly summary",
        "digest",
        "newsletter",
    ]

    if any(term in lower for term in critical_terms):
        return (
            9,
            True,
            "Critical signal detected by GrassWhisper safety rules.",
        )

    if (
        any(term in lower for term in low_terms)
        and not any(term in lower for term in important_terms)
    ):
        return (
            2,
            False,
            "Routine or promotional notification with no immediate action required.",
        )

    model_score = model_data.get("urgency_score", 0)

    try:
        model_score = max(
            1,
            min(10, int(model_score)),
        )
    except (TypeError, ValueError):
        model_score = 1

    has_direct_intent = any(
        term in lower
        for term in important_terms
    )

    is_direct_channel = (
        app in direct_apps
        or any(
            name in app
            for name in direct_apps
        )
    )

    if has_direct_intent:
        return (
            max(7, model_score),
            True,
            "The message asks for action, a response, or a time-sensitive change.",
        )

    if is_direct_channel and sender and text:
        casual_only = {
            "lol",
            "haha",
            "ok",
            "okay",
            "thanks",
            "thx",
            "nice",
            "great",
            "wow",
            "hello",
            "hi",
            "hey",
        }

        if len(text.split()) <= 2 and lower in casual_only:
            return (
                4,
                False,
                "Short casual message with no explicit action request.",
            )

        return (
            max(6, model_score),
            True,
            "Direct human communication should remain visible during a screen-free walk.",
        )

    if model_score >= 9:
        return (
            9,
            True,
            model_data.get("reason")
            or "The local model identified an immediate action.",
        )

    if model_score >= 6:
        return (
            model_score,
            True,
            model_data.get("reason")
            or "The local model identified a potentially important notification.",
        )

    return (
        max(3, model_score),
        False,
        model_data.get("reason")
        or "No immediate action is required.",
    )


def finalize_presentation(
    notif: NotificationItem,
    result: dict,
    prior_conversations: list,
    related_events: list,
    repeated_contacts: list,
):
    conversation_count = len(prior_conversations) + 1
    original_urgent = result["is_urgent"]

    topic_hint = message_topic(notif.message)
    repeated_count = len(repeated_contacts) + 1

    repeated_signal = (
        repeated_count >= 2
        and (
            message_has_call_signal(notif.message)
            or any(
                message_has_call_signal(
                    item.get("message", "")
                )
                for item in repeated_contacts
            )
        )
    )

    critical_repeated = repeated_count >= 3

    event_group = (
        related_events[0][1]
        if related_events
        else None
    )

    event_merge = (
        event_group is not None
        and not prior_conversations
    )

    replace_keys = []

    if prior_conversations:
        replace_keys.extend(
            item.get("presentation_key")
            for item in prior_conversations
            if item.get("presentation_key")
        )

    if event_group:
        replace_keys.extend(
            item.get("presentation_key")
            for _, item in related_events
            if item.get("presentation_key")
        )

    if repeated_signal:
        if critical_repeated:
            result["urgency_score"] = 9
            result["is_urgent"] = True
            result["interruption_level"] = "critical"

        elif result["urgency_score"] < 6:
            result["urgency_score"] = 6
            result["is_urgent"] = True
            result["interruption_level"] = "important"

        if result["is_urgent"] and not original_urgent:
            session_data["muted_count"] = max(
                0,
                session_data["muted_count"] - 1,
            )
            session_data["urgent_count"] += 1

        result["repeated_contact"] = True
        result["repeated_contact_count"] = repeated_count

        result["aggregation_summary"] = (
            build_repeated_contact_audio(
                notif.sender.strip() or "someone",
                repeated_count,
            )
        )

        result["conversation_count"] = conversation_count

        result["presentation_key"] = (
            f"repeat:{normalize_sender(notif.sender)}"
        )

        result["replace_display_keys"] = list(
            dict.fromkeys(replace_keys)
        )

        result["audio_whisper"] = make_audio_whisper(
            notif,
            result["urgency_score"],
            build_repeated_contact_audio(
                notif.sender.strip() or "someone",
                repeated_count,
            ),
        )

        result["audio_announced"] = bool(
            result["audio_whisper"]
        )

        return result

    if conversation_count > 1:
        result["conversation_count"] = conversation_count

        result["conversation_summary"] = (
            build_conversation_audio(
                notif,
                conversation_count,
                notif.message,
                topic_hint,
                result["urgency_score"],
            )
        )

        result["aggregation_summary"] = (
            result["conversation_summary"]
        )

        result["presentation_key"] = (
            f"conversation:"
            f"{clean_text(notif.app_name).lower()}:"
            f"{normalize_sender(notif.sender)}"
        )

        result["replace_display_keys"] = list(
            dict.fromkeys(replace_keys)
        )

        if result["is_urgent"]:
            prior_spoken = any(
                item.get("audio_announced")
                for item in prior_conversations
            )

            prior_max_score = max(
                (
                    int(
                        item.get(
                            "urgency_score",
                            0,
                        )
                    )
                    for item in prior_conversations
                ),
                default=0,
            )

            if (
                prior_spoken
                and result["urgency_score"] <= prior_max_score
            ):
                result["audio_whisper"] = ""
                result["audio_announced"] = False
            else:
                result["audio_whisper"] = (
                    make_audio_whisper(
                        notif,
                        result["urgency_score"],
                        result["conversation_summary"],
                    )
                )

                result["audio_announced"] = bool(
                    result["audio_whisper"]
                )

        else:
            result["audio_whisper"] = ""
            result["audio_announced"] = False

        return result

    if event_merge:
        event_topic = message_topic(
            notif.message
        )

        result["duplicate_merged"] = True
        result["duplicate_count"] = (
            len(related_events) + 1
        )

        result["event_topic"] = event_topic

        result["aggregation_summary"] = (
            f"One update about {event_topic}."
        )

        result["presentation_key"] = (
            event_group.get(
                "presentation_key"
            )
            or f"event:{event_signature(notif.message)}"
        )

        result["replace_display_keys"] = list(
            dict.fromkeys(replace_keys)
        )

        session_data["merge_count"] += 1

        if result["is_urgent"]:
            prior_spoken = any(
                item.get("audio_announced")
                for _, item in related_events
            )

            prior_max_score = max(
                (
                    int(
                        item.get(
                            "urgency_score",
                            0,
                        )
                    )
                    for _, item in related_events
                ),
                default=0,
            )

            if (
                prior_spoken
                and result["urgency_score"] <= prior_max_score
            ):
                result["audio_whisper"] = ""
                result["audio_announced"] = False

            else:
                result["audio_whisper"] = (
                    make_audio_whisper(
                        notif,
                        result["urgency_score"],
                        build_duplicate_event_audio(
                            event_topic,
                            session_data[
                                "audio_privacy_mode"
                            ] == "full",
                            notif.message,
                        ),
                    )
                )

                result["audio_announced"] = bool(
                    result["audio_whisper"]
                )

        else:
            result["audio_whisper"] = ""
            result["audio_announced"] = False

        return result

    result["conversation_count"] = 1
    result["duplicate_merged"] = False
    result["duplicate_count"] = 1
    result["repeated_contact"] = False
    result["repeated_contact_count"] = 1

    result["presentation_key"] = (
        f"notification:{result['created_at']}"
    )

    result["replace_display_keys"] = []

    result["audio_announced"] = bool(
        result.get("audio_whisper")
    )

    return result


@app.get("/api/status")
def get_status():
    model = get_available_model()

    return {
        "walk_active": session_data["walk_active"],
        "walk_context": session_data["walk_context"],
        "muted_count": session_data["muted_count"],
        "urgent_count": session_data["urgent_count"],
        "total_notifications": len(session_data["history"]),
        "audio_privacy_mode": session_data["audio_privacy_mode"],
        "model": model,
        "vip_contacts": session_data["vip_contacts"],
        "merge_count": session_data["merge_count"],
    }


@app.post("/api/settings/audio-privacy")
def set_audio_privacy(req: AudioPrivacyRequest):
    mode = req.mode.strip().lower()

    if mode not in ("safe", "full"):
        mode = "safe"

    session_data["audio_privacy_mode"] = mode

    return {
        "audio_privacy_mode": mode
    }


@app.get("/api/contacts")
def get_contacts():
    return session_data["vip_contacts"]


@app.post("/api/contacts")
def add_contact(req: ContactRequest):
    name = req.name.strip()

    if (
        name
        and name not in session_data["vip_contacts"]
    ):
        session_data["vip_contacts"].append(name)
        save_contacts_to_disk(
            session_data["vip_contacts"]
        )

    return session_data["vip_contacts"]


@app.delete("/api/contacts/{name}")
def delete_contact(name: str):
    session_data["vip_contacts"] = [
        c
        for c in session_data["vip_contacts"]
        if c.lower() != name.lower()
    ]

    save_contacts_to_disk(
        session_data["vip_contacts"]
    )

    return session_data["vip_contacts"]


@app.post("/api/walk/set")
def set_walk(req: WalkSetRequest):
    was_active = session_data["walk_active"]

    session_data["walk_active"] = req.active

    if req.context.strip():
        session_data["walk_context"] = req.context.strip()

    if req.active and not was_active:
        session_data["muted_count"] = 0
        session_data["urgent_count"] = 0
        session_data["history"] = []
        session_data["merge_count"] = 0

    return {
        "walk_active": session_data["walk_active"],
        "walk_context": session_data["walk_context"],
    }


@app.post("/api/walk/toggle")
def toggle_walk(
    req: Optional[WalkContextRequest] = None
):
    session_data["walk_active"] = (
        not session_data["walk_active"]
    )

    if req and req.context:
        session_data["walk_context"] = (
            req.context
        )

    if session_data["walk_active"]:
        session_data["muted_count"] = 0
        session_data["urgent_count"] = 0
        session_data["history"] = []
        session_data["merge_count"] = 0

    return {
        "walk_active": session_data["walk_active"],
        "walk_context": session_data["walk_context"],
    }


@app.post("/api/walk/reset")
def reset_session():
    session_data["muted_count"] = 0
    session_data["urgent_count"] = 0
    session_data["history"] = []
    session_data["merge_count"] = 0

    return {
        "status": "reset_complete",
        "walk_active": session_data["walk_active"],
    }


@app.get("/api/walk/history")
def get_history():
    return session_data["history"]


@app.post("/api/triage")
def triage_notification(
    notif: NotificationItem
):
    import time
    import datetime

    now_str = datetime.datetime.now().strftime(
        "%H:%M:%S"
    )

    created_at = time.time()
    message_lower = notif.message.lower()

    otp_terms = [
        "otp",
        "one-time password",
        "one time password",
        "verification code",
        "security code",
        "login code",
        "authentication code",
        "confirm code",
    ]

    birthday_terms = [
        "happy birthday",
        "happy bday",
        "birthday wishes",
        "b'day",
    ]

    prior_conversations = conversation_context(
        notif
    )

    related_events = related_event_context(
        notif
    )

    repeated_contacts = repeated_contact_context(
        notif
    )

    if any(
        term in message_lower
        for term in otp_terms
    ):
        result = {
            "is_urgent": True,
            "is_vip": False,
            "urgency_score": 9,
            "interruption_level": "critical",
            "reason": (
                "Verification and OTP codes are "
                "always treated as time-sensitive."
            ),
            "audio_whisper": make_audio_whisper(
                notif,
                9,
            ),
            "app_name": notif.app_name,
            "sender": notif.sender,
            "message": notif.message,
            "timestamp": now_str,
            "created_at": created_at,
        }

        if prior_conversations:
            result["conversation_count"] = (
                len(prior_conversations) + 1
            )

            result["presentation_key"] = (
                f"conversation:"
                f"{clean_text(notif.app_name).lower()}:"
                f"{normalize_sender(notif.sender)}"
            )

            result["replace_display_keys"] = [
                item.get("presentation_key")
                for item in prior_conversations
                if item.get("presentation_key")
            ]

            result["audio_whisper"] = (
                make_audio_whisper(
                    notif,
                    9,
                    build_conversation_audio(
                        notif,
                        len(prior_conversations) + 1,
                        notif.message,
                        message_topic(
                            notif.message
                        ),
                        9,
                    ),
                )
            )

        session_data["urgent_count"] += 1

        session_data["history"].insert(
            0,
            result,
        )

        return result

    if any(
        term in message_lower
        for term in birthday_terms
    ):
        result = {
            "is_urgent": False,
            "is_vip": False,
            "urgency_score": 1,
            "interruption_level": "queued",
            "reason": (
                "Birthday wishes are social "
                "messages and do not require "
                "an interruption."
            ),
            "audio_whisper": "",
            "app_name": notif.app_name,
            "sender": notif.sender,
            "message": notif.message,
            "timestamp": now_str,
            "created_at": created_at,
        }

        result = finalize_presentation(
            notif,
            result,
            prior_conversations,
            related_events,
            repeated_contacts,
        )

        session_data["muted_count"] += 1

        session_data["history"].insert(
            0,
            result,
        )

        return result

    matched_vip = None

    for vip in session_data["vip_contacts"]:
        if (
            vip.lower() in notif.sender.lower()
            or notif.sender.lower() in vip.lower()
        ):
            matched_vip = vip
            break

    model_name = get_available_model()

    system_prompt = f"""You are GrassWhisper, a private audio gatekeeper for outdoor walks.

The user is outside and wants to stay screen-free.

Walk filter:
{session_data['walk_context']}

Decide whether the user needs to act now.

Rules:
1. OTPs and verification codes are always critical.
2. Birthday wishes are queued silently.
3. VIP contacts do not automatically create an interruption.
4. Delivery confirmations and routine updates are usually queued.
5. Emergencies, safety issues, deadlines, schedule changes, meetings, appointments, direct requests, questions, and messages requiring a response can be important.
6. A notification count alone is not enough to make a message urgent.
7. Protect screen-free peace.
8. Ordinary direct human messages should normally be IMPORTANT unless they are clearly casual and non-actionable.
9. Critical situations use urgency_score 9-10.
10. Important situations use urgency_score 6-8.
11. Only clearly low-value notifications use urgency_score 1-5.
12. Never classify a normal personal message as low priority simply because it does not contain the word urgent.

Return only valid JSON with:
action_needed_now
is_urgent
urgency_score
reason
audio_whisper
"""

    user_prompt = f"""Incoming Notification:

App: {notif.app_name}

Sender: {notif.sender}

Saved VIP contact: {matched_vip or "No"}

Message:
"{notif.message}"

Return JSON only.
"""

    try:
        if ollama is None:
            raise RuntimeError(
                "Ollama is unavailable"
            )

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0.1
            },
            format="json",
        )

        content = (
            response["message"]["content"]
            .strip()
        )

        if "```json" in content:
            content = (
                content
                .split("```json")[1]
                .split("```")[0]
                .strip()
            )

        elif "```" in content:
            content = (
                content
                .split("```")[1]
                .split("```")[0]
                .strip()
            )

        data = json.loads(content)

    except Exception as e:
        logger.error(
            f"Ollama inference error: {e}"
        )

        low_signal_terms = [
            "liked your post",
            "liked your photo",
            "new follower",
            "started following you",
            "reacted to your message",
            "new reel",
            "new story",
            "sale",
            "discount",
            "coupon",
            "promo",
            "promotional",
            "sponsored",
            "order shipped",
            "has shipped",
            "shipping confirmation",
            "delivery update",
            "package delivered",
            "newsletter",
            "weekly summary",
            "daily reminder",
            "weather update",
            "battery",
            "screen time",
            "meme",
            "funny dog video",
            "hilarious dog video",
        ]

        critical_fallback_terms = [
            "emergency",
            "accident",
            "hospital",
            "ambulance",
            "fire",
            "urgent",
            "critical",
            "p0",
            "security breach",
            "account hacked",
            "account compromised",
            "fraud alert",
            "card blocked",
            "card stolen",
            "call me now",
            "call me immediately",
            "need you now",
            "need you immediately",
        ]

        important_fallback_terms = [
            "meeting",
            "exam",
            "test",
            "interview",
            "appointment",
            "doctor",
            "dentist",
            "flight",
            "boarding",
            "gate",
            "deadline",
            "due today",
            "due tomorrow",
            "rescheduled",
            "schedule changed",
            "changed from",
            "moved to",
            "starts at",
            "starts in",
            "can you",
            "could you",
            "would you",
            "will you",
            "please call",
            "call me",
            "need you",
            "reply",
            "respond",
            "confirm",
            "approve",
            "join",
            "available",
            "are you free",
            "are you coming",
            "where are you",
            "tomorrow at",
            "today at",
            "tonight",
        ]

        if any(
            term in message_lower
            for term in critical_fallback_terms
        ):
            fallback_score = 9
            fallback_urgent = True
            fallback_reason = (
                "Critical action detected by "
                "local fallback rules."
            )

        elif any(
            term in message_lower
            for term in important_fallback_terms
        ):
            fallback_score = 7
            fallback_urgent = True
            fallback_reason = (
                "Important action or response "
                "detected by local fallback rules."
            )

        elif any(
            term in message_lower
            for term in low_signal_terms
        ):
            fallback_score = 2
            fallback_urgent = False
            fallback_reason = (
                "Low-value notification detected "
                "by local fallback rules."
            )

        else:
            fallback_score = 6
            fallback_urgent = True
            fallback_reason = (
                "Direct or human notification kept "
                "visible rather than silently discarded."
            )

        data = {
            "action_needed_now": fallback_urgent,
            "is_urgent": fallback_urgent,
            "urgency_score": fallback_score,
            "reason": fallback_reason,
            "audio_whisper": "",
        }

    urgency_score, is_urgent, policy_reason = (
        deterministic_classification(
            notif,
            data,
            matched_vip,
        )
    )

    if urgency_score >= 9:
        interruption_level = "critical"
        is_urgent = True

    elif urgency_score >= 6:
        interruption_level = "important"
        is_urgent = True

    else:
        interruption_level = "queued"
        is_urgent = False

    model_reason = clean_text(
        str(data.get("reason", ""))
    )

    reason = (
        policy_reason
        if policy_reason
        else model_reason
        or "No immediate action is required."
    )

    if is_urgent:
        session_data["urgent_count"] += 1
    else:
        session_data["muted_count"] += 1

    result = {
        "is_urgent": is_urgent,
        "interruption_level": interruption_level,
        "is_vip": bool(matched_vip),
        "urgency_score": urgency_score,
        "reason": reason,
        "audio_whisper": make_audio_whisper(
            notif,
            urgency_score,
        ),
        "app_name": notif.app_name,
        "sender": notif.sender,
        "message": notif.message,
        "timestamp": now_str,
        "created_at": created_at,
    }

    result = finalize_presentation(
        notif,
        result,
        prior_conversations,
        related_events,
        repeated_contacts,
    )

    session_data["history"].insert(
        0,
        result,
    )

    return result


@app.get("/")
def serve_index():
    return FileResponse(
        os.path.join(
            APP_DIR,
            "index.html",
        )
    )


@app.get("/offline.css")
def serve_styles():
    return FileResponse(
        os.path.join(
            APP_DIR,
            "offline.css",
        ),
        media_type="text/css",
    )


if __name__ == "__main__":
    try:
        import uvicorn  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "uvicorn is required to run this app. Install it with: pip install uvicorn"
        ) from exc

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )