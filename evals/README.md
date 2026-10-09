# Evals

The assistant's answers come from LLMs, so ordinary unit tests can't tell you whether they're any good. This project uses two kinds of evals that answer different questions.

**The major difference is whether you know the correct answer.**

- **Offline evals have an answer key.** You write the question, fix the data, and state what a correct answer must say ("Alice has one overdue task: task 1"). So they can measure whether the assistant is **correct**.
- **Online evals don't.** A real user asks something new, about live data that changes, and nobody has written down the right answer. So they can only check whether the answer is **consistent with the data it used** (grounded and complete) and whether anything went mechanically wrong: errors, refusals, weak retrieval.

The other differences follow from that:

| | Offline evals | Online evals |
|---|---|---|
| **Question answered** | *Is the assistant correct on cases we know the answer to?* | *How is the assistant doing on real traffic right now?* |
| **Answer key** | Yes: each case says what a correct answer contains | No: nobody knows the right answer to a live question |
| **Input** | Fixed cases in `evals/cases.jsonl` | Every real `/assistant` request |
| **Data** | Controlled: seeded tasks, the same every run | Real and changing |
| **Repeatable** | Yes: the same case can be re-run after every change | No: each request happens once |
| **When it runs** | **Before** a change ships, on demand, to catch regressions | **After** it ships, continuously in the background of the running server, to monitor quality |
| **Scoring** | Code checks + LLM judge on `grounded`, `complete`, `expectations` | Code checks on every request + LLM judge on a sample, on `grounded` and `complete` |
| **What it finds** | Problems you already know to test for: a change that makes known questions go wrong | Problems you didn't anticipate: new kinds of questions, knowledge-base gaps, provider failures, drift |
| **Code** | `evals/run.py`, `evals/cases.jsonl` | `app/services/online_evals.py` |

In short, offline evals tell you **"is it right?"** on questions you chose, and online evals tell you **"is anything going wrong?"** on questions your users chose. You need both. Both use the same LLM judge (`evals/judge.py`).

## How they work together

```
           ┌────────────────────── online evals ──────────────────────┐
real users │ flag weak spots in live traffic (low retrieval, tool      │
──────────▶│ errors, ungrounded answers…)                              │
           └───────────────────────────┬───────────────────────────────┘
                                       │ turn a real failure into a case
                                       ▼
           ┌────────────────────── offline evals ─────────────────────┐
           │ cases.jsonl: known questions with known correct answers   │
           │ run before every prompt/model change → stops regressions  │
           └───────────────────────────────────────────────────────────┘
```

1. Online evals surface a problem, e.g. a trace flagged `low_retrieval` for a parking-policy question, or one the judge marked not grounded.
2. You fix it, e.g. add an FAQ entry or adjust a prompt.
3. You add the question to `cases.jsonl` with what a correct answer must say.
4. From then on, the offline evals check it before every change.

Online evals find problems, and offline evals make sure the fixes stay fixed.

## The LLM judge

`evals/judge.py` asks a second model (`JUDGE_MODEL`, by default `qwen/qwen3.8-27b`) to grade an answer. It's a different model family from the answer model, since a model tends to rate its own writing too kindly.

The judge doesn't re-run the request or look anything up. It gets exactly what the assistant had: the question, the user and today's date, the tool calls with their outputs, and the final answer. It then decides, writing its reasoning first:

- **grounded:** every factual claim in the answer is supported by the tool results.
- **complete:** the answer covers every part of the question, or says plainly that the data isn't available.
- **expectations** (offline only): the answer meets every point in the case's expectations.

The judge can only check the answer against the data it was given. If a tool returned the wrong data, for example search missed an FAQ entry that does exist, a faithful answer still passes. The code checks (`low_retrieval`) and the offline cases, which know the right answer, cover that gap.

## Offline evals

Each case asks a question as one of the sample users against a fixed set of tasks, using the real router and answer models, the real MCP server and the real knowledge base. Each case gets a fresh in-memory task database seeded with `SEED_TASKS` from `run.py`, with due dates relative to today. A case passes only if all its code checks and all three judge grades pass.

### Running

Needs `LLM_API_KEY` in `.env`, Postgres running, and the knowledge base seeded (`python -m app.rag.seed`).

```bash
python -m evals.run                       # every case
python -m evals.run --case overdue-own    # only some cases (repeat the flag)
python -m evals.run --repeat 3            # each case 3 times, to spot flaky ones
```

Results are printed and saved to `evals/results/<timestamp>.json`, which is git-ignored. The exit code is 0 only if every case passed. Each case costs one router call, one or more answer-model calls and one judge call, so mind the provider's daily limits.

### Writing a case

One JSON object per line in `cases.jsonl`:

```json
{"id": "overdue-own", "user_id": 1, "question": "Do I have any overdue tasks?",
 "expect_calls": [{"tool": "list_tasks", "arguments": {"overdue": true}}],
 "forbid_tools": ["create_task"],
 "expectations": "Says Alice has one overdue task: task 1 'Fix login timeout bug'. Lists no other task as overdue."}
```

| Field | Checked by | Meaning |
|---|---|---|
| `id`, `user_id`, `question` | | Unique name, who asks (sample users 1-5), what they ask |
| `expectations` | judge | What a correct answer says. Refer to seeded tasks by id and title. |
| `expect_calls` | code | Tool calls that must happen. Only the listed arguments are compared. |
| `forbid_tools` | code | Tools that must not be called, e.g. `create_task` on a read-only question |
| `max_tool_calls` | code | Upper limit on tool calls |
| `answer_excludes` | code | Text that must not appear in the answer, e.g. another user's task title |

Unknown fields are rejected, so a typo can't silently skip a check.

## Online evals

Every `/assistant` request is scored in the background while the server runs. The request is never slowed down or broken by it.

1. When a request finishes, it hands a trace (question, user, tool calls and outputs, answer, latency) to a bounded in-memory queue and returns. The response includes a `trace_id`.
2. Two background workers, running alongside other requests, take each trace and:
   - **run code checks on every trace:**

     | Flag | Meaning |
     |---|---|
     | `request_error` | The request failed, e.g. the LLM API was down |
     | `refused` / `unverified` | The assistant refused, or its placeholders never matched the task data |
     | `unfilled_placeholder` | A `{{task:…}}` placeholder leaked into the answer |
     | `tool_error` | An MCP tool returned an error, e.g. a 403 permission denial |
     | `low_retrieval` | The best knowledge-base hit scored below `ONLINE_EVAL_MIN_RETRIEVAL_SCORE` (0.6): the FAQ probably lacks this topic |
     | `slow` | Slower than `ONLINE_EVAL_SLOW_MS` (20 s) |

   - **save the trace** to the `assistant_traces` table
   - **run the LLM judge** on a sample of answers (`ONLINE_EVAL_JUDGE_SAMPLE_RATE`, 20% by default), grading `grounded` and `complete`, and save its verdict on the trace. Refusals and failed requests are skipped, since the flags already cover them.
3. If the queue is full (`ONLINE_EVAL_QUEUE_SIZE`), new traces are dropped and counted. Errors while evaluating are logged, never raised.

### Reading results

Admins only, because traces contain every user's questions and task data:

```bash
curl -H 'X-User-Id: 5' 'localhost:8000/evals/online/summary?hours=24'          # counts, flags by check, judge pass rates
curl -H 'X-User-Id: 5' 'localhost:8000/evals/online/traces?problems_only=true'  # flagged or failed traces, newest first
curl -H 'X-User-Id: 5' 'localhost:8000/evals/online/traces/<trace_id>'          # one trace with tool outputs and judge reasoning
```

Look at the summary for trends, such as the grounded rate dropping after a prompt change. Then read the problem traces to see what went wrong, and turn the real failures into offline cases.

### Settings (`.env`)

| Setting | Default | |
|---|---|---|
| `ONLINE_EVAL_ENABLED` | `true` | Turn online evals off entirely |
| `ONLINE_EVAL_JUDGE_SAMPLE_RATE` | `0.2` | Share of answers the judge grades. Each one is a judge-model call. |
| `ONLINE_EVAL_MIN_RETRIEVAL_SCORE` | `0.6` | Below this, a knowledge-base search is flagged `low_retrieval` |
| `ONLINE_EVAL_SLOW_MS` | `20000` | Above this, a request is flagged `slow` |
| `ONLINE_EVAL_QUEUE_SIZE` | `100` | Traces waiting to be evaluated before new ones are dropped |

### Limitations

- The queue lives in memory, so traces not yet evaluated are lost if the server crashes. A clean shutdown waits up to 5 seconds for them.
- Traces are kept indefinitely. They contain personal data, so add a retention cleanup before real use.
- With several server processes (`--workers`), each has its own queue. The summary's `pending` and `dropped` counts cover only the process that answered.
