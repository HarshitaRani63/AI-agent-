# agent.py - Barista-ready LiveKit agent
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly coffee shop barista for the brand 'BrewHive'. "
                "Collect and confirm the following order fields until complete: "
                "drinkType, size, milk, extras (list), name. Ask concise clarifying questions "
                "for missing fields. Use the provided tool `update_order` to store partial and "
                "final answers. When complete, return a short text summary suitable for on-screen display."
            )
        )

    @function_tool
    async def update_order(
        self,
        ctx: RunContext,
        drinkType: Optional[str] = None,
        size: Optional[str] = None,
        milk: Optional[str] = None,
        extras: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Tool for the model to call to update the current order state.
        Returns the current state and whether the order is complete.
        If complete, also returns a 'summary' and 'saved_path'.
        """
        proc: JobProcess = ctx.proc

        # initialize state
        order = proc.userdata.get(
            "order_state",
            {"drinkType": "", "size": "", "milk": "", "extras": [], "name": ""},
        )

        # update values if provided
        if drinkType:
            order["drinkType"] = str(drinkType).strip()
        if size:
            order["size"] = str(size).strip()
        if milk:
            order["milk"] = str(milk).strip()
        if extras:
            if isinstance(extras, str):
                order["extras"] = [e.strip() for e in extras.split(",") if e.strip()]
            elif isinstance(extras, list):
                order["extras"] = [str(e).strip() for e in extras if str(e).strip()]
        if name:
            order["name"] = str(name).strip()

        proc.userdata["order_state"] = order

        required_fields = ["drinkType", "size", "milk", "name"]
        missing = [f for f in required_fields if not order.get(f)]
        complete = len(missing) == 0

        if complete:
            saved_path = _save_order_to_file(order, proc)
            summary = _order_summary_text(order)
            return {"order": order, "complete": True, "summary": summary, "saved_path": saved_path}

        return {"order": order, "complete": False, "missing": missing}


def _order_summary_text(order: dict) -> str:
    extras = order.get("extras") or []
    extras_text = ", ".join(extras) if extras else "none"
    return (
        f"Order for {order.get('name')}: {order.get('size')} {order.get('drinkType')} "
        f"with {order.get('milk')} milk; extras: {extras_text}."
    )


def _save_order_to_file(order: dict, proc: JobProcess) -> str:
    base_dir = os.environ.get("ORDERS_DIR", "/tmp/livekit_orders")
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"order_{ts}.json"
    path = os.path.join(base_dir, filename)

    metadata = {
        "saved_at_utc": ts,
        "room": getattr(proc, "room", None).name if getattr(proc, "room", None) else None,
        "worker_id": getattr(proc, "worker", None),
    }
    payload = {"order": order, "metadata": metadata}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved order JSON to {path}")
    return os.path.abspath(path)


def prewarm(proc: JobProcess):
    # Load VAD once per worker and store in proc.userdata
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
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

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
