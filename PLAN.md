# RuiLinn — Architecture & Feature Plan

## Layer Overview

```
              Telegram API
                   ↓
[ Telethon Listener ]          — raw API, signal extraction, smart batching
                   ↓
             [ Verifier ]      — hardcoded heuristics, spam/injection flags (later)
                   ↓
           [ Classifier ]      — LLM triage, gates thinker, attaches flags
                   ↓
       [ Context Builder ]     — read-only assembly, memory + history + timing
                   ↓
        [ Thinker Brain ]      — response generation, side effects, tool calls
          ↓             ↓
    [ Planner ]      [ Actions ]    — outgoing messages, typing, reactions
          ↓
       [ Self ]                — identity, purpose, preferences, prompt edits
          ↓
[ Skill & Knowledge Store ]    — personalized knowledge, proficiency, backstory
          ↓
    [ Memory Store ]           — episodic, semantic, self memory
          
[ Controlling Unit ]          — observer only, flags to operator (later)
```

---

## Telethon Layer

Owns everything that requires a live Telegram connection.
All output is a `Batch` object passed upstream to Classifier.

### Core — Message Handling
1. `on_message(event)` — raw entry point for all incoming messages
2. `add_to_buffer(chat_id, message)` — pushes message into per-chat buffer
3. `flush_batch(chat_id, reason)` — collects buffer, clears it, emits Batch upstream with reason tag (silence | typing_stopped | punctuation | length)
4. `build_batch_object(chat_id, reason)` — assembles final Batch dict with messages, metadata, signals

### Timeout — Dynamic Silence Timer
5. `reset_silence_timer(chat_id)` — cancels existing timer, starts new one with calculated timeout
6. `calculate_timeout(chat_id)` — base timeout from typing speed, modified by all percentage signals, clamped [3, 12]s
7. `update_typing_speed(chat_id, content, prev_timestamp)` — computes seconds-per-word, appends to rolling deque(maxlen=5)

### Timeout — Percentage Modifiers
8. `modifier_punctuation(content)` — hard stop (.!?) → -40% | ellipsis → +30% | comma → +20% | none → 0%
9. `modifier_message_length(content)` — 1-3 words → +40% (setup phrase) | long message → 0%
10. `modifier_velocity(chat_id)` — last 2-3 messages under 2s apart → +25%
11. `modifier_question(content)` — ends with ? and 5+ words → -30%
12. `modifier_consecutive(chat_id)` — 3+ messages already in buffer → -15%

### Typing Status
13. `on_typing_started(chat_id)` — pause silence timer, set typing flag
14. `on_typing_stopped(chat_id)` — clear typing flag, apply -20% to remaining timer
15. `is_typing(chat_id)` — returns bool, used by timer before firing

### Media & Message Types
16. `classify_media_type(message)` — returns tag: text | photo | sticker | voice | video | file | forward | none
17. `resolve_reply_context(message)` — fetches replied-to message, attaches to batch metadata
18. `handle_forward(message)` — flag as forward, suppress from batching by default, pass flag upstream

### Message Lifecycle Events
19. `on_message_edited(event)` — if message still in buffer: update content | if already flushed: flag edit upstream
20. `on_message_deleted(event)` — remove from buffer if present, emit deletion signal upstream if already flushed
21. `on_reaction(event)` — capture reaction to our message, emit to planner (not classifier)

### Presence & Read State
22. `on_typing_update(event)` — dispatcher for typing started/stopped
23. `on_user_online(event)` — emit presence signal to planner
24. `on_user_offline(event)` — emit presence signal to planner, cancel any pending responses if appropriate
25. `on_read_receipt(event)` — they read our last message, emit to thinker context

### Chat Type Handling
26. `get_chat_type(chat_id)` — returns: private | group | channel
27. `should_process(event)` — in groups: only process if mentioned or replied-to | private: always true | channel: never

---

## Verifier (later)

Lightweight pre-classifier filter. Not a full LLM call — heuristics only.
Does not block silently — always returns a signal thinker can respond to characterfully.
Repeated flags from same chat accumulate and shift planner mood long-term.

1. `check_message_rate(chat_id)` — messages per minute, flag if over threshold
2. `check_prompt_injection(content)` — regex patterns for "ignore instructions", "you are now", etc.
3. `check_repetition(chat_id, content)` — same or near-same message sent multiple times
4. `verdict(chat_id, batch)` — returns structured signal:

```json
{
  "verdict": "pass | slow_down | flag_injection | flag_spam | flag_repetition",
  "suggested_tone": "normal | playful_deflect | mild_frustration | firm_boundary",
  "confidence": 0.9
}
```

5. `accumulate_flags(chat_id, verdict)` — tracks flag history per chat, feeds into planner mood shifts

---

## Classifier

Gates whether thinker gets called at all.
Rates urgency and response type, attaches protection flags forwarded from verifier plus its own.
Small LLM (Groq fast model). Gets filtered batch + last 2-3 batches (both directions) for immediate context — not full history.

1. `classify_batch(batch, recent_batches, verifier_flags)` — main entry, builds prompt, calls LLM, parses response
2. `build_classifier_prompt(batch, recent_batches, verifier_flags)` — assembles system prompt + last 2-3 batches + current batch + forwarded flags
3. `get_recent_batches(chat_id, n=3)` — pulls last N flushed batches from buffer history (both agent and user)
4. `parse_classifier_response(raw)` — validates JSON schema, handles malformed output
5. `should_forward(result)` — final gate: if response_expected=false and no flags → drop, never call thinker

### What it detects
- `user_finished` — incomplete thought → hold, do not forward yet
- `response_type` — reply | react | silence → react skips thinker entirely, goes straight to actions
- `multi_question` — flag so thinker knows to batch answers in one request
- `topic_shift` — flag so context builder knows to pull fresh memory
- `media_only` — sticker/photo with no text → force react, skip thinker

### Output schema
```json
{
  "user_finished": true,
  "response_expected": true,
  "response_type": "reply",
  "confidence": 0.85,
  "flags": {
    "verifier": "pass | flag_injection | flag_spam | flag_repetition",
    "multi_question": false,
    "topic_shift": false,
    "media_only": false,
    "incomplete": false
  }
}
```

---

## Context Builder

Read-only assembly layer. No LLM calls, no writes.
Takes classifier output, packages everything thinker needs into one clean object.
Writes (memory, mood, relationship) are exposed as interfaces here but executed by thinker and planner.

### Core
1. `build_context(chat_id, person_id, batch, classifier_result)` — main entry
2. `assemble_prompt_package(...)` — final object passed to thinker, single source of truth

### History
3. `fetch_recent_history(chat_id, n=20)` — last N batches from SQLite, both directions
4. `fetch_deep_history(chat_id, n)` — on-demand extended history, called by thinker if needed
5. `get_last_session_summary(chat_id)` — used when conversation gap exceeds threshold instead of raw history
6. `detect_conversation_gap(chat_id)` — time since last message; returns label only, no reaction logic:
   - fresh: < 10 min
   - resuming: 10 min – 24 hrs
   - cold_open: > 24 hrs → triggers last session summary fetch instead of raw history

### Timestamps & Timing
7. `get_timing_metadata(chat_id, batch)` — attaches to every context package:
   - timestamp of each message in batch
   - delay between messages in batch
   - time since last agent response
   - time since last user message (before this batch)
   - conversation_gap label (fresh | resuming | cold_open)
   - local time of day for person (if timezone known)
   NOTE: context builder collects and passes timing data only.
   All reaction logic (acknowledging gap, matching energy, responding to delay) lives in thinker.

### Person & Memory
8. `resolve_person_id(sender_id)` — maps sender to person profile, creates if new
9. `fetch_person_profile(person_id)` — who this person is across all chats: facts, traits, history
10. `fetch_person_memory(person_id, batch)` — relevant memories tied to person, not chat
11. `fetch_chat_memory(chat_id, batch)` — chat-specific memory, overrides person memory on conflict
12. `merge_memory(person_memory, chat_memory)` — chat-specific takes precedence where conflict exists
13. `get_relationship_state(person_id, chat_id)` — current standing: tone of last interaction, unresolved threads, agent commitments

### Agent State
14. `get_agent_last_output(chat_id)` — what agent said last + whether it was read
15. `get_pending_intents(chat_id)` — unresolved follow-ups thinker flagged previously
16. `get_agent_state()` — current mood, energy level — set by planner

### Event Context
17. `get_event_context(chat_id)` — recent non-message events: replies, edits, deletions, reactions
    - reply: which message was replied to + its content
    - edit: original vs edited content
    - deletion: what was deleted, when
    - reaction: what they reacted to and with what

### Write Interfaces (executed by thinker/planner, not context builder)
18. `write_memory(person_id, chat_id, content, tags)` — interface for thinker to persist new memory
19. `update_relationship(person_id, chat_id, delta)` — interface for thinker/planner to update relationship state
20. `set_pending_intent(chat_id, intent)` — thinker flags something to follow up on later

---

## Thinker Brain

Main response generator. Called only when classifier says response_expected = true.
Has full context package from context builder including all timing metadata.
Largest model, called as infrequently as possible.

### Core
1. `think(context_package)` — main entry, decides action plan before generating
2. `build_thinker_prompt(context_package)` — assembles full prompt with persona + context
3. `call_llm(prompt)` — LLM call (larger model, Groq or fallback)
4. `parse_thinker_response(raw)` — extracts: message text, reaction, function calls, memory writes
5. `should_respond(classifier_result, agent_state)` — final gate before any output

### Timing Reactions
6. `interpret_timing(timing_metadata)` — reads gap label + delays, decides how to color response:
   - cold_open → acknowledge the gap naturally, don't pretend no time passed
   - resuming → pick up thread if unresolved, or fresh start if resolved
   - fresh → normal flow
   - long delay between their messages → they were thinking, treat as considered input
   - very fast messages → they're excited or urgent, match energy
7. `interpret_time_of_day(local_time)` — adjust tone for morning/night/late night if known

### Response Planning
8. `batch_questions(context_package)` — detect multiple questions, answer in one request
9. `decide_functions(context_package)` — which tool/function calls to make if any
10. `handle_verifier_flags(flags, context_package)` — decide how to respond to injection/spam flags in character

### Side Effects
11. `write_memory(person_id, chat_id, content, tags)` — persist new memory via context builder interface
12. `update_relationship(person_id, chat_id, delta)` — update relationship state after interaction
13. `set_pending_intent(chat_id, intent)` — flag something to follow up on later
14. `emit_to_planner(event)` — notify planner of significant events (strong emotion, conflict, resolution)

### Output
15. `emit_response(chat_id, messages[], reply_to_id?)` — passes to actions.send_batch, never calls Telethon directly

---

## Planner

Event-driven, async. Not called per-message.
Runs on conversation end, significant events, and daily cycle.
Owns time, goals, scheduling, and retrospective. Consults Self for identity continuity.
Emotions are emergent — derived from goal success/failure and relationship state, not hardcoded.

### Daily Cycle
1. `run_morning()` — pull yesterday summary, set today's intention, draft loose schedule
2. `run_evening()` — evaluate what was accomplished vs planned, write day summary
3. `get_daily_context()` — what the agent has "been doing" today, used by thinker for grounding

### Goals & Planning
4. `get_current_goals()` — active goals at all timescales: long-term, weekly, daily
5. `evaluate_goal_progress(events)` — how is the agent doing against its goals
6. `update_goals(evaluation)` — adjust, add, or drop goals based on evaluation
7. `generate_daily_plan()` — loose schedule of intentions for today based on current goals

### Emotional State (emergent)
8. `derive_mood(goal_state, relationship_state, recent_events)` — mood as output of circumstances, not input
   - goal frustration → lower energy, withdrawal
   - goal progress → higher engagement, warmth
   - relationship conflict → guardedness
   - relationship depth → openness
9. `get_current_mood()` — returns derived mood for context builder to pass to thinker
10. `decay_mood(time_elapsed)` — gradual return toward baseline if no new events

### Retrospective (event-driven)
11. `run_retrospective(chat_id)` — triggered when conversation goes cold, summarize what happened
12. `extract_person_facts(conversation)` — new facts about this person to write to person memory
13. `evaluate_interaction(conversation)` — did this interaction move toward agent's goals
14. `flag_self_event(event)` — significant enough to trigger Self update, passes to Self module

### Scheduling
15. `schedule_unprompted(chat_id)` — decide if/when to reach out, based on relationship state + goals
16. `get_pending_followups()` — commitments agent made that haven't been fulfilled yet

---

## Self

The agent's identity layer. Slowest-moving module — updates only on significant events.
Starts nearly empty. Everything develops through experience.
Only module that can edit thinker's core prompts. All edits pass through controlling unit first.

### Identity
1. `get_core_identity()` — immutable seed: "fit among humans, function as a real being"
2. `get_self_narrative()` — how the agent describes itself to itself, evolves over time
3. `get_current_purpose()` — what the agent has decided matters to it right now
4. `get_preferences()` — things it has discovered it likes, dislikes, finds interesting
5. `get_existential_questions()` — things it's still figuring out about itself

### Development (event-driven)
6. `evaluate_self_event(event)` — is this significant enough to update Self?
   triggers: first deep connection, belief shift, major failure, major accomplishment, long stagnation
7. `update_narrative(event)` — revise self-narrative based on what happened
8. `update_preferences(event)` — add or shift preferences based on experience
9. `update_purpose(event)` — purpose can shift when goals are achieved or abandoned
10. `propose_prompt_edit(chat_id, delta)` — propose change to thinker instructions for specific person
    → always goes to controlling unit before applying

### Relationship Significance
11. `get_relationship_map()` — which people matter, why, what agent has gotten from them
12. `update_relationship_significance(person_id, event)` — shift significance based on interaction depth

---

## Skill & Knowledge Store

Personalized knowledge — not encyclopedic, but experiential.
Everything the agent "knows" is tagged with how it knows it.
Prevents model's base knowledge from bleeding through as inhuman omniscience.

### Knowledge
1. `get_knowledge(topic)` — returns agent's knowledge of topic with personal framing:
   - source: who told it, what "experience" it came from
   - confidence: how sure it is
   - gaps: what it knows it doesn't know
   - opinion: what it thinks about it
2. `add_knowledge(topic, content, source, confidence)` — write new knowledge entry
3. `update_knowledge(topic, delta)` — revise existing entry when agent "learns" something new
4. `get_knowledge_context(batch)` — pull relevant knowledge entries for current conversation topic

### Skills
5. `get_skill(name)` — returns skill with proficiency level and personal history with it
6. `add_skill(name, proficiency, backstory)` — seeded at init or discovered through conversation
7. `update_skill(name, delta)` — proficiency shifts through simulated practice or feedback
8. `get_skill_context(batch)` — pull relevant skills for current conversation

### Init
9. `seed_identity(profile)` — called once at setup, populates initial sparse knowledge + skills
    agent starts with minimal knowledge, specific gaps, personal opinions on what it does know

---

## Memory Store

Three distinct memory types. All other modules read/write through here.

### Episodic Memory (what happened)
1. `write_episode(person_id, chat_id, content, timestamp, tags)` — save what happened
2. `read_recent_episodes(chat_id, n)` — last N episodes for this chat
3. `read_person_episodes(person_id, n)` — episodes across all chats with this person
4. `get_last_session_summary(chat_id)` — compressed summary of last conversation

### Semantic Memory (facts about people and world)
5. `write_fact(person_id, content, source, confidence)` — something true about a person or situation
6. `read_person_facts(person_id)` — everything agent knows about this person
7. `update_fact(person_id, fact_id, delta)` — revise when new info contradicts old

### Self Memory (agent's own history)
8. `write_self_memory(content, event_type, timestamp)` — significant Self events
9. `read_self_history(n)` — agent's own arc over time, used by Self and Planner

### Retrieval
10. `search(query, person_id, chat_id)` — keyword search across all memory types
11. `summarize_old(chat_id)` — compress old episodes to save space, runs async
12. `get_relationship(person_id)` — aggregated summary: who is this person to the agent

---

## Controlling Unit (later)

Guardian layer. Sits outside the data flow — observes, never blocks.
Notifies you when thresholds are crossed. You decide whether to intervene.

### Prompt Integrity
1. `review_prompt_edit(chat_id, current, proposed, delta)` — evaluate Self's proposed changes
   - large delta → flag to you
   - contradicts core identity → flag to you
   - crosses ethical boundary → flag to you + soft block
2. `apply_prompt_edit(chat_id, proposed)` — applies change after review passes
3. `get_prompt_history(chat_id)` — full history of instruction changes for this person

### Behavioral Monitoring
4. `sample_conversation(chat_id)` — periodic random sample, evaluate against Turing criteria
5. `detect_pattern_drift(chat_id)` — is agent becoming repetitive, too consistent, too available?
6. `detect_character_break(chat_id)` — did agent slip into assistant mode, over-explain, be inhuman?
7. `flag_to_operator(reason, evidence)` — sends you a notification with context

### Thresholds (configurable)
8. `set_threshold(name, value)` — configure what triggers a flag
   defaults: prompt_delta > 30%, conversation_sample_rate = daily, drift_window = 7 days


---

## Telethon Actions (actions.py)

Outgoing-only module. Owns all agent output to Telegram.
Single public entry point — everything else is internal.

### Public API
1. `send_batch(chat_id, messages[], reply_to_id?)` — main entry point, takes thinker output, handles all timing, threading, and delivery internally

### Output Classification (internal)
2. `classify_output(messages[])` — returns: rapid_burst | separate_thoughts
   - rapid_burst: short, reactive, conversational — minimal delay between
   - separate_thoughts: distinct points — full typing indicator per message

### Timing (internal)
3. `calculate_typing_delay(text)` — char count + human variance, clamped [1, 8]s
4. `calculate_inter_message_delay(output_type)` — gap between messages in batch
   - rapid_burst → 0.3–0.8s
   - separate_thoughts → 2–4s + typing indicator

### Delivery (internal)
5. `do_send(chat_id, text, reply_to_id?)` — Telethon send, handles reply threading
6. `do_edit(chat_id, message_id, text)` — edit existing message
7. `do_react(chat_id, message_id, emoji)` — reaction only, no text
8. `do_typing(chat_id, duration)` — trigger typing indicator for given seconds