# Agentic Research System

An AI agent that takes a research **goal**, autonomously **plans** a sequence
of sub-tasks, **reasons** step-by-step about which tools to use, **acts**
by calling those tools, **observes** the results, and **responds** with a
single coherent report — built for the Techvruk "Agentic AI Systems" contest.

This is not a one-shot prompt. It is a small, inspectable orchestration
layer around an LLM that implements two well-known agentic design
patterns together:

- **Plan-and-Execute** — decompose the goal into an ordered plan *before* acting.
- **ReAct (Reason + Act)** — within each plan step, loop `reason -> tool call -> observation` until the step is answered.

---

## 1. Problem / Task Chosen

**Research Agent**: *"Given a topic, search, summarize, and compile a short report."*
(from the contest's example task list)

Given any free-text goal (e.g. *"Research the current landscape of small
modular nuclear reactors and summarize key players, timelines, and
challenges"*), the agent independently:

1. Breaks the goal into concrete research steps.
2. Investigates each step using tools (web search, calculator, file I/O),
   deciding for itself when it has enough information.
3. Carries forward everything it has learned as it moves through steps.
4. Produces one final Markdown report that answers the original goal.

## 2. System Architecture

```mermaid
flowchart TD
    U[User goal / CLI input] --> P[Planner<br/>LLM: decompose goal into steps]
    P --> M1[(Agent Memory<br/>plan + state)]
    M1 --> LOOP

    subgraph LOOP [For each plan step]
        direction TB
        E[Executor: ReAct loop] -->|reason| D{LLM decides:<br/>call a tool or answer?}
        D -->|tool_use| T[Tool Registry]
        T -->|web_search| WS[Web Search<br/>Tavily / DuckDuckGo / mock]
        T -->|calculator| CA[Calculator]
        T -->|read/write_file| FI[File I/O sandbox]
        WS --> OBS[Observation fed back to LLM]
        CA --> OBS
        FI --> OBS
        OBS --> D
        D -->|final answer| F[Step finding]
    end

    F --> M2[(Agent Memory<br/>updated with finding)]
    M2 -->|next step, with full context| LOOP
    M2 -->|all steps done| S[Synthesizer<br/>LLM: compile full transcript into final report]
    S --> R[Final Markdown report]
```

### Component map

| Module | Responsibility | Agentic principle |
|---|---|---|
| `src/planner.py` | Ask the LLM to decompose the goal into an ordered JSON plan | Planning |
| `src/executor.py` | Run the ReAct loop (reason → tool call → observe) for one step | Reasoning + Tool use |
| `src/tools/*` | `web_search`, `calculator`, `read_file`/`write_file` — pluggable, sandboxed | Execution / external world |
| `src/memory.py` | Append-only record of the plan, every tool call, and every step's finding | State across steps |
| `src/agent.py` | Orchestrator: wires planner → executor loop → synthesizer | Workflow orchestration |
| `src/llm_client.py` | Backend-agnostic LLM interface (`AnthropicLLMClient` + offline `MockLLMClient`) | Practical LLM integration |
| `src/cli.py` | Text-based entry point | User input |

### Why this shape

- **Separation of planning from acting.** The planner never touches tools;
  the executor never re-decides the overall plan. Each LLM call has one
  job, which keeps prompts small and behaviour debuggable — you can log
  and inspect the plan independently of how any single step was answered.
- **State is explicit, not implicit in a chat history.** `AgentMemory`
  is a typed object (`plan`, `steps[].tool_traces`, `steps[].finding`).
  Every step's prompt includes a *summary* of prior findings
  (`memory.context_for_step`), not the entire raw transcript — this keeps
  token usage bounded as the number of steps grows, while still giving
  later steps the context they need.
- **Tools are sandboxed and pluggable.** Every tool implements one
  `Tool` interface (`src/tools/__init__.py`); adding a capability means
  writing one class and passing it into `ToolRegistry` — no orchestration
  changes. `CalculatorTool` uses an AST whitelist (no `eval`); file tools
  reject path traversal outside a workspace directory.
- **Graceful degradation, end to end.** If `ANTHROPIC_API_KEY` is unset,
  `build_llm_client()` transparently falls back to a deterministic
  `MockLLMClient` so the full plan→act→observe→respond loop can be
  demoed and unit-tested with zero network access. If a search provider
  fails (rate-limited, blocked, offline), `WebSearchTool` catches the
  error and falls back to the next provider rather than crashing the run
  (see the sample run below, where exactly this happened).
- **Iteration guards everywhere.** `MAX_PLAN_STEPS` and
  `MAX_REACT_ITERATIONS_PER_STEP` bound both stages so a confused LLM
  can't loop forever or blow up the token budget.

## 3. Setup / Run Instructions

```bash
git clone <this-repo-url>
cd agentic-research-system
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY for real LLM calls.
# (Optional) set TAVILY_API_KEY for higher-quality live web search.
export $(cat .env | xargs)   # or use `python-dotenv` / your shell's preferred method

python -m src.cli --goal "Research the current landscape of small modular nuclear reactors (SMRs) and summarize key players, timelines, and challenges"
```

Run the offline test suite (no API key or network needed):

```bash
python -m unittest discover -s tests -v
```

### Without an API key

The agent still runs end-to-end with `MockLLMClient`, which follows a
scripted plan→tool-call→answer policy. This is intentional: it lets a
grader/reviewer exercise the full agentic workflow with **zero setup**,
and it's exactly how `tests/test_agent.py` verifies the loop
deterministically in CI.

## 4. Sample Input / Output

**Input:**
```
python -m src.cli --goal "Research the current landscape of small modular nuclear reactors (SMRs) and summarize key players, timelines, and challenges"
```

**Console trace (actual run, offline mode — see `examples/sample_run_console.log`):**
```
WARNING | No ANTHROPIC_API_KEY found — falling back to MockLLMClient (offline demo mode)
INFO    | WebSearchTool using provider: duckduckgo
INFO    | === New run === GOAL: Research the current landscape of small modular nuclear reactors...
INFO    | Generated plan with 2 step(s)
INFO    |   Step 1: Gather background facts about: GOAL: Research the current landscape of SMRs...
INFO    |   Step 2: Identify recent developments / current state
WARNING | Search provider 'duckduckgo' failed (403 Forbidden); falling back to mock.
INFO    | Step 1 -> tool `web_search`({'query': '...'})
INFO    | Step 1 finished after 2 iteration(s)
INFO    | Step 2 -> tool `web_search`({'query': '...'})
INFO    | Step 2 finished after 2 iteration(s)
INFO    | === Run complete ===
```

Note the `WARNING | Search provider 'duckduckgo' failed ... falling back to mock`
line — that's the tool-failure fallback in `WebSearchTool` firing for
real (DuckDuckGo blocked the sandboxed request), demonstrating the
system degrading gracefully instead of crashing the run.

**Final output** (`examples/sample_run_output.md`, offline mode):
```markdown
# Mock Final Report

This is a mock, offline synthesis of all step findings (no API key was configured).
```

With `ANTHROPIC_API_KEY` set, the same command produces a real multi-section
Markdown report with genuine findings per step, e.g. sections like
*Executive Summary*, *Key Players (NuScale, Rolls-Royce SMR, X-energy...)*,
*Timelines*, and *Regulatory / Cost Challenges*, each grounded in the
tool observations gathered during the ReAct loop.

## 5. Extending the Agent

- **New tool** → subclass `Tool` in `src/tools/`, implement `run(**kwargs)`,
  add it to the list passed into `ToolRegistry` in `src/agent.py`.
- **New LLM provider** → implement `LLMClient.chat(...)` in `src/llm_client.py`
  (same pattern as `AnthropicLLMClient`); no other file needs to change.
- **Conditional / branching workflows** → `ResearchAgent.run()` currently
  executes the plan sequentially; it's a natural extension point for
  adding conditional branches (e.g. skip a step if an earlier finding
  already answers it) or parallel step execution.

## 6. Contest Compliance Notes

- Core loop is genuinely agentic: LLM-driven planning, per-step reasoning
  with tool selection, execution via real tool calls, explicit state
  carried across steps, and a final synthesized response — not a single
  prompt/response.
- Uses the Anthropic Claude API (`anthropic` SDK) as the primary LLM,
  with an offline mock backend for zero-cost grading/demo.
- All core logic (planner, executor, memory, tool sandboxing) is
  original and written for this contest.
