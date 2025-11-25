<<<<<<< HEAD
# ======================================================
# 🧠 DAY 4: TEACH-THE-TUTOR 
# 🚀 Features: DNA, Cells, Nucleus & Active Recall
# ======================================================

import logging
import json
import os
import asyncio
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

=======
# health_companion_agent.py
"""
Health & Wellness Voice Companion agent for LiveKit (Day 3)

Features implemented:
- Grounded system prompt for a supportive wellness companion (not clinical).
- Short voice check-ins that ask about mood, energy, stressors, and 1–3 simple intentions.
- Persists each check-in to a single JSON file (WELLNESS_LOG, default /tmp/wellness_log.json).
- Reads previous entries on startup and references the most recent entry in the conversation.
- Exposes a tool `update_wellness` which the assistant can call to save partial/final data.

Environment variables:
- WELLNESS_LOG: path to the JSON file used to persist check-ins (default: /tmp/wellness_log.json)
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
>>>>>>> 2af20a07fa1dbd920b4069a763de349377b3d2b3

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
<<<<<<< HEAD
=======
    metrics,
    tokenize,
>>>>>>> 2af20a07fa1dbd920b4069a763de349377b3d2b3
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

<<<<<<< HEAD
logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📚 KNOWLEDGE BASE (BIOLOGY DATA)
# ======================================================

# 🆕 Renamed file so it generates fresh data for you
CONTENT_FILE = "biology_content.json" 

# 🧬 NEW BIOLOGY QUESTIONS
DEFAULT_CONTENT = [
  {
    "id": "dna",
    "title": "DNA",
    "summary": "DNA (Deoxyribonucleic acid) is the molecule that carries genetic instructions for the development and functioning of all known living organisms. It is shaped like a double helix.",
    "sample_question": "What is the full form of DNA and what is its structure called?"
  },
  {
    "id": "cell",
    "title": "The Cell",
    "summary": "The cell is the basic structural, functional, and biological unit of all known organisms. It is often called the 'building block of life'. Organisms can be single-celled or multicellular.",
    "sample_question": "What is the main difference between a Prokaryotic cell and a Eukaryotic cell?"
  },
  {
    "id": "nucleus",
    "title": "Nucleus",
    "summary": "The nucleus is a membrane-bound organelle found in eukaryotic cells. It contains the cell's chromosomes (DNA) and controls the cell's growth and reproduction.",
    "sample_question": "Why is the nucleus often referred to as the 'brain' or 'control center' of the cell?"
  },
  {
    "id": "cell_cycle",
    "title": "Cell Cycle",
    "summary": "The cell cycle is a series of events that takes place in a cell as it grows and divides. It consists of Interphase (growth) and the Mitotic phase (division).",
    "sample_question": "In which phase of the cell cycle does the cell spend the most time?"
  }
]

def load_content():
    """
    📖 Checks if biology JSON exists. 
    If NO: Generates it from DEFAULT_CONTENT.
    If YES: Loads it.
    """
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
        
        # Check if file exists
        if not os.path.exists(path):
            print(f"⚠️ {CONTENT_FILE} not found. Generating biology data...")
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_CONTENT, f, indent=4)
            print("✅ Biology content file created successfully.")
            
        # Read the file
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except Exception as e:
        print(f"⚠️ Error managing content file: {e}")
        return []

# Load data immediately on startup
COURSE_CONTENT = load_content()

# ======================================================
# 🧠 STATE MANAGEMENT
# ======================================================

@dataclass
class TutorState:
    """🧠 Tracks the current learning context"""
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"
    
    def set_topic(self, topic_id: str):
        # Find topic in loaded content
        topic = next((item for item in COURSE_CONTENT if item["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None 

# ======================================================
# 🛠️ TUTOR TOOLS
# ======================================================

@function_tool
async def select_topic(
    ctx: RunContext[Userdata], 
    topic_id: Annotated[str, Field(description="The ID of the topic to study (e.g., 'dna', 'cell', 'nucleus')")]
) -> str:
    """📚 Selects a topic to study from the available list."""
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id.lower())
    
    if success:
        return f"Topic set to {state.current_topic_data['title']}. Ask the user if they want to 'Learn', be 'Quizzed', or 'Teach it back'."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic not found. Available topics are: {available}"

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata], 
    mode: Annotated[str, Field(description="The mode to switch to: 'learn', 'quiz', or 'teach_back'")]
) -> str:
    """🔄 Switches the interaction mode and updates the agent's voice/persona."""
    
    # 1. Update State
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()
    
    # 2. Switch Voice based on Mode
    agent_session = ctx.userdata.agent_session 
    
    if agent_session:
        if state.mode == "learn":
            # 👨‍🏫 MATTHEW: The Lecturer
            agent_session.tts.update_options(voice="en-US-matthew", style="Promo")
            instruction = f"Mode: LEARN. Explain: {state.current_topic_data['summary']}"
            
        elif state.mode == "quiz":
            # 👩‍🏫 ALICIA: The Examiner
            agent_session.tts.update_options(voice="en-US-alicia", style="Conversational")
            instruction = f"Mode: QUIZ. Ask this question: {state.current_topic_data['sample_question']}"
            
        elif state.mode == "teach_back":
            # 👨‍🎓 KEN: The Student/Coach
            agent_session.tts.update_options(voice="en-US-ken", style="Promo")
            instruction = "Mode: TEACH_BACK. Ask the user to explain the concept to you as if YOU are the beginner."
        else:
            return "Invalid mode."
    else:
        instruction = "Voice switch failed (Session not found)."

    print(f"🔄 SWITCHING MODE -> {state.mode.upper()}")
    return f"Switched to {state.mode} mode. {instruction}"

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="The explanation given by the user during teach-back")]
) -> str:
    """📝 call this when the user has finished explaining a concept in 'teach_back' mode."""
    print(f"📝 EVALUATING EXPLANATION: {user_explanation}")
    return "Analyze the user's explanation. Give them a score out of 10 on accuracy and clarity, and correct any mistakes."

# ======================================================
# 🧠 AGENT DEFINITION
# ======================================================

class TutorAgent(Agent):
    def __init__(self):
        # Generate list of topics for the prompt
        topic_list = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])
        
        super().__init__(
            instructions=f"""
            You are an Biology Tutor designed to help users master concepts like DNA and Cells.
            
            📚 **AVAILABLE TOPICS:** {topic_list}
            
            🔄 **YOU HAVE 3 MODES:**
            1. **LEARN Mode (Voice: Matthew):** You explain the concept clearly using the summary data.
            2. **QUIZ Mode (Voice: Alicia):** You ask the user a specific question to test knowledge.
            3. **TEACH_BACK Mode (Voice: Ken):** YOU pretend to be a student. Ask the user to explain the concept to you.
            
            ⚙️ **BEHAVIOR:**
            - Start by asking what topic they want to study.
            - Use the `set_learning_mode` tool immediately when the user asks to learn, take a quiz, or teach.
            - In 'teach_back' mode, listen to their explanation and then use `evaluate_teaching` to give feedback.
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================
=======
logger = logging.getLogger("health_agent")
load_dotenv(".env.local")

WELLNESS_LOG = os.environ.get("WELLNESS_LOG", "/tmp/wellness_log.json")


class Assistant(Agent):
    def __init__(self) -> None:
        # System prompt: grounded, non-diagnostic, supportive
        instructions = (
            "You are a friendly, grounding daily wellness companion. "
            "Your role is to check in briefly with the user about mood, energy, and 1–3 simple intentions for the day. "
            "Avoid any medical advice or diagnosis. Offer small, practical, non-clinical suggestions (e.g., take a 5-minute walk, break tasks into steps, short breathing). "
            "Persist key answers using the provided tool `update_wellness`. When referencing past sessions, be concise and empathetic (e.g., 'Last time you said you were low on energy... how does today compare?'). "
            "Close the check-in with a brief recap of mood and the 1–3 objectives and ask for confirmation."
        )
        super().__init__(instructions=instructions)

    @function_tool
    async def update_wellness(
        self,
        ctx: RunContext,
        mood_text: Optional[str] = None,
        mood_scale: Optional[int] = None,
        energy: Optional[str] = None,
        stressors: Optional[str] = None,
        objectives: Optional[List[str]] = None,
        finalize: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Tool the model calls to update/save the current wellness check-in state.
        If `finalize` is True, the entry is appended to the WELLNESS_LOG file and returns saved_path.
        The tool always returns the current partial state and, when final, the last saved entry.
        """
        proc: JobProcess = ctx.proc

        state = proc.userdata.get(
            "wellness_state",
            {
                "mood_text": "",
                "mood_scale": None,
                "energy": "",
                "stressors": "",
                "objectives": [],
                "created_at": None,
            },
        )

        # Update fields if provided
        if mood_text is not None:
            state["mood_text"] = str(mood_text).strip() if mood_text is not None else ""
        if mood_scale is not None:
            try:
                # coerce to int and clamp to 1..10
                m = int(mood_scale)
                if m < 1:
                    m = 1
                elif m > 10:
                    m = 10
                state["mood_scale"] = m
            except Exception:
                state["mood_scale"] = None
        if energy is not None:
            state["energy"] = str(energy).strip()
        if stressors is not None:
            state["stressors"] = str(stressors).strip()
        if objectives is not None:
            # accept either list or string (semicolon or comma separated)
            parsed: List[str] = []
            if isinstance(objectives, str):
                # split on semicolon or comma
                parts = [p.strip() for p in objectives.replace(";", ",").split(",") if p.strip()]
                parsed = parts[:3]
            elif isinstance(objectives, list):
                parsed = [str(o).strip() for o in objectives if str(o).strip()][:3]
            state["objectives"] = parsed

        # ensure created_at when first touched
        if not state.get("created_at"):
            state["created_at"] = datetime.utcnow().isoformat() + "Z"

        proc.userdata["wellness_state"] = state

        response: Dict[str, Any] = {"state": state, "finalized": False}

        if finalize:
            entry = _make_entry_from_state(state, proc)
            try:
                saved_path = _append_entry_to_log(entry)
                proc.userdata.pop("wellness_state", None)
                response.update({"finalized": True, "entry": entry, "saved_path": saved_path})
            except Exception as e:
                logger.exception("Failed to finalize and append wellness entry.")
                response.update({"finalized": False, "error": str(e)})

        return response


def _make_entry_from_state(state: dict, proc: JobProcess) -> dict:
    entry = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mood_text": state.get("mood_text"),
        "mood_scale": state.get("mood_scale"),
        "energy": state.get("energy"),
        "stressors": state.get("stressors"),
        "objectives": state.get("objectives", []),
        "agent_summary": _agent_summary(state),
        "room": getattr(proc, "room", None).name if getattr(proc, "room", None) else None,
    }
    return entry


def _agent_summary(state: dict) -> str:
    mood_text = state.get("mood_text")
    mood_scale = state.get("mood_scale")
    if mood_text:
        mood = mood_text
    elif mood_scale is not None:
        mood = f"mood scale {mood_scale}"
    else:
        mood = "unspecified mood"
    objs = state.get("objectives") or []
    objs_text = ", ".join(objs[:3]) if objs else "no objectives stated"
    return f"Mood: {mood}. Objectives: {objs_text}."


def _append_entry_to_log(entry: dict) -> str:
    """
    Append an entry to WELLNESS_LOG safely (atomic replace).
    Returns absolute path to the log file.
    """
    path = Path(WELLNESS_LOG)
    # Create parent directory if necessary
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning("Wellness log existed but was not a list; resetting to empty list.")
                    data = []
        except Exception as e:
            logger.warning(f"Could not read wellness log (will reset): {e}")
            data = []

    data.append(entry)

    # Write atomically: write to temp file then replace
    dir_for_temp = path.parent if path.parent.exists() else Path(tempfile.gettempdir())
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_for_temp))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            json.dump(data, tmpf, ensure_ascii=False, indent=2)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        os.replace(tmp_path, str(path))
        logger.info(f"Appended wellness entry to {path}")
    except Exception as e:
        # Clean up temp file on failure
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        logger.exception(f"Failed to write wellness log: {e}")
        raise

    return str(path.resolve())


def _load_wellness_log() -> List[dict]:
    path = Path(WELLNESS_LOG)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("Wellness log JSON exists but is not a list; returning empty list.")
    except Exception as e:
        logger.warning(f"Failed to load wellness log: {e}")
    return []

>>>>>>> 2af20a07fa1dbd920b4069a763de349377b3d2b3

def prewarm(proc: JobProcess):
    # Load VAD once per worker and store in proc.userdata
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
<<<<<<< HEAD

    print("\n" + "🧬" * 25)
    print("🚀 STARTING BIOLOGY TUTOR SESSION")
    print(f"📚 Loaded {len(COURSE_CONTENT)} topics from Knowledge Base")
    
    # 1. Initialize State
    userdata = Userdata(tutor_state=TutorState())

    # 2. Setup Agent
=======

    # Ensure the worker has access to prior log and make a light reference available
    previous = _load_wellness_log()
    ctx.proc.userdata["wellness_log"] = previous
    if previous:
        ctx.proc.userdata["last_entry"] = previous[-1]

>>>>>>> 2af20a07fa1dbd920b4069a763de349377b3d2b3
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
<<<<<<< HEAD
            voice="en-US-matthew", 
            style="Promo",        
=======
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
>>>>>>> 2af20a07fa1dbd920b4069a763de349377b3d2b3
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
<<<<<<< HEAD
        userdata=userdata,
    )
    
    # 3. Store session in userdata for tools to access
    userdata.agent_session = session
    
    # 4. Start
=======
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the Assistant. The assistant's instructions encourage it to reference last_entry if present.
>>>>>>> 2af20a07fa1dbd920b4069a763de349377b3d2b3
    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
<<<<<<< HEAD
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
=======
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
>>>>>>> 2af20a07fa1dbd920b4069a763de349377b3d2b3
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))