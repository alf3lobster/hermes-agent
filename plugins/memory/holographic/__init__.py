"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


def _load_stack_role(role: str) -> dict:
    """Read a model role from ~/.alfred/config/stack.yaml.

    role: dotted path under llm, e.g. "api.haiku" → llm.api.haiku block.
    Returns dict with "model", "endpoint", "provider" keys.
    Raises KeyError if the role or file is missing.
    """
    stack_path = Path.home() / ".alfred" / "config" / "stack.yaml"
    try:
        import yaml
        with open(stack_path, encoding="utf-8") as f:
            stack = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise KeyError(f"stack.yaml not found at {stack_path}") from exc
    node = stack.get("llm", {})
    for part in role.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"llm.{role} not found in stack.yaml")
        node = node[part]
    if not isinstance(node, dict):
        raise KeyError(f"llm.{role} is not a dict in stack.yaml")
    return node


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))
        # Per-turn extraction state (used when auto_extract_trigger == 'per_turn')
        self._turn_counts: dict[str, int] = {}
        self._session_buffers: dict[str, deque] = {}
        self._extracting_sessions: set[str] = set()

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._config.get("auto_extract", False):
            return
        trigger = str(self._config.get("auto_extract_trigger", "session_end")).strip()
        if trigger != "per_turn":
            return

        every_n = int(self._config.get("auto_extract_every_n_turns", 8))
        window = int(self._config.get("auto_extract_window_turns", 20))
        sid = session_id or "_default"

        if sid not in self._session_buffers:
            # maxlen = window * 2: one deque entry per role (user + assistant per turn)
            self._session_buffers[sid] = deque(maxlen=window * 2)
            self._turn_counts[sid] = 0

        buf = self._session_buffers[sid]
        if user_content:
            buf.append({"role": "user", "content": user_content})
        if assistant_content:
            buf.append({"role": "assistant", "content": assistant_content})

        self._turn_counts[sid] = self._turn_counts.get(sid, 0) + 1
        count = self._turn_counts[sid]

        if count % every_n == 0 and sid not in self._extracting_sessions:
            messages_snapshot = list(buf)
            t = threading.Thread(
                target=self._auto_extract_from_buffer,
                args=(sid, messages_snapshot),
                daemon=True,
            )
            t.start()

    def _auto_extract_from_buffer(self, session_id: str, messages: list) -> None:
        """Daemon-thread extraction from the per-session sliding window buffer."""
        self._extracting_sessions.add(session_id)
        try:
            if not self._store:
                return

            dry_run = bool(self._config.get("auto_extract_dry_run", False))
            model, provider, endpoint = self._resolve_extraction_model()

            lines = []
            for msg in messages:
                role = msg.get("role", "")
                text = self._extract_text_from_content(msg.get("content", "")).strip()
                if text:
                    lines.append(f"{role.upper()}: {text[:800]}")
            conversation = "\n\n".join(lines)
            if not conversation.strip():
                return

            raw = self._call_extraction_model(conversation, model, provider, endpoint=endpoint)
            candidates = self._parse_extraction_output(raw)

            stored = 0
            skipped = 0
            for content, category, tags in candidates:
                if self._store.fact_exists_similar(content):
                    skipped += 1
                    logger.debug("Holographic per-turn extract: duplicate skipped: %s", content[:80])
                    continue
                if dry_run:
                    logger.info(
                        "Holographic per-turn extract [DRY RUN] would store: [%s] %s",
                        category, content,
                    )
                    stored += 1
                    continue
                try:
                    self._store.add_fact(content, category=category, tags=tags)
                    stored += 1
                except Exception as exc:
                    logger.debug("Holographic: failed to store extracted fact: %s", exc)

            label = "[DRY RUN] " if dry_run else ""
            logger.info(
                "Holographic per-turn extract %s(session=%s): %d facts %s, %d duplicates skipped",
                label, session_id, stored,
                "would be stored" if dry_run else "stored",
                skipped,
            )
        except Exception as exc:
            logger.warning(
                "Holographic per-turn extract failed (session=%s): %s", session_id, exc
            )
        finally:
            self._extracting_sessions.discard(session_id)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.get("auto_extract", False):
            return
        if not self._store:
            return

        trigger = str(self._config.get("auto_extract_trigger", "session_end")).strip()
        if trigger == "per_turn":
            # Extraction happens in sync_turn daemon threads; clean up session state.
            sid = getattr(self, "_session_id", "_default") or "_default"
            self._turn_counts.pop(sid, None)
            self._session_buffers.pop(sid, None)
            return

        if not messages:
            return
        self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction ----------------------------------------------------

    def _resolve_extraction_model(self) -> tuple[str, str, str]:
        """Return (model, provider, endpoint) from stack.yaml role or fallback config keys."""
        stack_role = str(self._config.get("auto_extract_stack_role", "")).strip()
        model = "claude-haiku-4-5-20251001"
        provider = "anthropic"
        endpoint = ""
        if stack_role:
            try:
                role_cfg = _load_stack_role(stack_role)
                model = str(role_cfg.get("model", model))
                provider = str(role_cfg.get("provider", provider))
                endpoint = str(role_cfg.get("endpoint", ""))
            except KeyError as exc:
                logger.warning(
                    "Holographic auto-extract: stack.yaml role %r not found (%s), using config keys",
                    stack_role, exc,
                )
                model = str(self._config.get("auto_extract_model", model)).strip()
                provider = str(self._config.get("auto_extract_provider", provider)).strip().lower()
        else:
            model = str(self._config.get("auto_extract_model", model)).strip()
            provider = str(self._config.get("auto_extract_provider", provider)).strip().lower()
        return model, provider, endpoint

    # Extraction prompt sent to the LLM. Keep it tight — output must be parseable.
    _EXTRACTION_PROMPT = """\
Review the following conversation and extract facts worth storing permanently in a structured memory system.

RULES:
- Extract facts, not conversation. Write "Delivery scripts live at /path/..." not "User asked about delivery scripts."
- One fact per output line. Each fact must be self-contained — no pronouns without referents.
- Only extract facts that are specific, durable, and would be useful to recall in a future session.
- Mine BOTH user and assistant turns — tool outputs, confirmed paths, resolved errors, and synthesised answers often contain the most durable facts.
- Skip: pleasantries, clarifying questions, transient status ("I'm looking at it now"), general knowledge.
- Maximum 15 facts. Prefer fewer, high-quality facts over many weak ones.

OUTPUT FORMAT — one fact per line, exactly:
FACT [category] [tag1,tag2] fact content here

category must be one of: tool | project | user_pref | general
tags: 2-4 lowercase hyphenated keywords (e.g. delivery,scripts,macbook)

CONVERSATION:
{conversation}

Output only FACT lines. No preamble, no explanation."""

    def _extract_text_from_content(self, content) -> str:
        """Pull plain text from a message content field (str or list of blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return ""

    def _format_conversation(self, messages: list, max_turns: int) -> str:
        """Render the last max_turns user/assistant turns as plain text.

        max_turns=0 means include all turns.
        """
        relevant = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        if max_turns > 0:
            relevant = relevant[-max_turns:]

        lines = []
        for msg in relevant:
            role = msg.get("role", "")
            text = self._extract_text_from_content(msg.get("content", "")).strip()
            if text:
                lines.append(f"{role.upper()}: {text[:800]}")
        return "\n\n".join(lines)

    def _call_extraction_model(
        self, conversation: str, model: str, provider: str, endpoint: str = ""
    ) -> str:
        """Make a synchronous LLM call and return the raw text response.

        provider: "api" or "anthropic" → Anthropic SDK
                  "local" → Ollama OpenAI-compatible chat endpoint
        """
        prompt = self._EXTRACTION_PROMPT.format(conversation=conversation)

        if provider in ("api", "anthropic"):
            import anthropic
            try:
                from agent.anthropic_adapter import resolve_anthropic_token
                api_key = resolve_anthropic_token() or ""
            except Exception:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise RuntimeError("No Anthropic API key available for extraction")
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""

        if provider == "local":
            import urllib.request
            base = (endpoint or "http://localhost:11434").rstrip("/")
            url = f"{base}/v1/chat/completions"
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]

        raise ValueError(f"Unsupported auto_extract_provider: {provider!r}")

    def _parse_extraction_output(self, raw: str) -> list[tuple[str, str, str]]:
        """Parse FACT lines into (content, category, tags) tuples."""
        valid_categories = {"tool", "project", "user_pref", "general"}
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("FACT "):
                continue
            parts = line[5:].split(None, 2)  # strip "FACT ", split into [cat, tags, content]
            if len(parts) < 3:
                continue
            category, tags, content = parts
            category = category.strip("[]").lower()
            tags = tags.strip("[]")
            content = content.strip()
            if category not in valid_categories:
                category = "general"
            if content:
                results.append((content, category, tags))
        return results

    def _auto_extract_facts(self, messages: list) -> None:
        """LLM-based session-end fact extraction using a cheap fast model.

        Passes the last N turns to the configured extraction model, parses
        structured FACT lines, deduplicates against existing facts, and stores
        new facts.  Falls back to regex extraction if the LLM call fails.
        """
        model, provider, endpoint = self._resolve_extraction_model()
        max_turns = int(self._config.get("auto_extract_max_turns", 80))
        min_turns = int(self._config.get("auto_extract_min_turns", 5))
        dry_run = bool(self._config.get("auto_extract_dry_run", False))

        relevant_count = sum(
            1 for m in messages if m.get("role") in ("user", "assistant")
        )
        if relevant_count < min_turns:
            logger.debug(
                "Holographic auto-extract: only %d turns, skipping (min=%d)",
                relevant_count, min_turns,
            )
            return

        try:
            conversation = self._format_conversation(messages, max_turns)
            if not conversation.strip():
                return

            raw = self._call_extraction_model(conversation, model, provider, endpoint=endpoint)
            candidates = self._parse_extraction_output(raw)

            stored = 0
            skipped = 0
            for content, category, tags in candidates:
                if self._store.fact_exists_similar(content):
                    skipped += 1
                    logger.debug("Holographic auto-extract: duplicate skipped: %s", content[:80])
                    continue
                if dry_run:
                    logger.info("Holographic auto-extract [DRY RUN] would store: [%s] %s", category, content)
                    stored += 1
                    continue
                try:
                    self._store.add_fact(content, category=category, tags=tags)
                    stored += 1
                except Exception as exc:
                    logger.debug("Holographic: failed to store extracted fact: %s", exc)

            if dry_run:
                logger.info(
                    "Holographic auto-extract [DRY RUN]: %d facts would be stored, %d duplicates skipped — set auto_extract_dry_run: false to commit",
                    stored, skipped,
                )
            else:
                logger.info(
                    "Holographic auto-extract: %d new facts stored, %d duplicates skipped",
                    stored, skipped,
                )

        except Exception as exc:
            logger.warning(
                "Holographic LLM auto-extraction failed (%s), falling back to regex", exc
            )
            self._auto_extract_facts_regex(messages)

    def _auto_extract_facts_regex(self, messages: list) -> None:
        """Regex-based fallback extraction — sparse but zero-dependency."""
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]
        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = self._extract_text_from_content(msg.get("content", ""))
            if not content or len(content) < 10:
                continue
            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="user_pref")
                        extracted += 1
                    except Exception:
                        pass
                    break
            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="project")
                        extracted += 1
                    except Exception:
                        pass
                    break
        if extracted:
            logger.info("Holographic regex fallback extracted %d facts", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
