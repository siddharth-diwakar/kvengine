Project 2: Mini inference engine with a paged KV cache manager and spec decoding.  This is the centerpiece, and it absorbs four items from the thread at once: KV cache manager, paged attention, dynamic shapes, and spec decoding. Build a serving engine for a small open model (Qwen/Llama 1–3B) that does: weight loading, a block-based KV cache allocator (your slab-pool experience from the order book transfers directly — it's the same problem, memory pools under churn), continuous batching, and then speculative decoding on top. Benchmark tokens/sec and cache utilization against naive HuggingFace generation. Read the vLLM paged attention paper and skim SGLang's RadixAttention to understand the design space — mentioning the differences between them (prefix caching via radix tree vs. pure paging) is exactly the kind of thing that lands in interviews.

Phase 0 — Foundations. On your Mac: load Qwen2.5-0.5B with HuggingFace, but don't use the built-in .generate(). Write your own loop: feed tokens in, get logits out (the model's scores for every possible next token), pick the highest one, append, repeat. Then pass the past_key_values object between steps — that's you touching the KV cache directly for the first time. Goal: your loop's output exactly matches .generate() with greedy decoding. This exact-match test is your correctness anchor for the entire project.

Phase 1 — Own the cache. Replace HuggingFace's KV cache object with tensors you allocate and manage yourself, still one contiguous chunk per request, still one request at a time. This phase exists because it isolates "do I understand the cache's shape and indexing" from "is my allocator correct" — debugging both at once is misery. Same test: exact output match.

Phase 2 — The paged allocator. The centerpiece. Pre-allocate one big tensor of N blocks at startup. Write a BlockManager: free list, allocate/free, per-request block tables. Your attention computation now has to gather K and V from scattered blocks using the block table. Write real tests here — allocate/free churn, out-of-memory behavior, no block ever owned by two requests. This is where your slab-pool instincts pay off, and it's the part interviews will dig into.

Phase 3 — Continuous batching. Multiple requests in flight: a waiting queue, a running batch, and a scheduler loop that admits requests when blocks are available and retires them on completion. The classic subtlety you'll hit: new requests need a long "prefill" pass (process the whole prompt) while running ones need one-token "decode" passes — deciding how to mix these is a real design decision worth writing up.

Phase 4 — Benchmark on Kaggle. Scripted, not notebook-interactive: fire M concurrent requests with realistic length distributions at (a) your engine and (b) a tuned HuggingFace baseline (fp16, batched) on the same T4. Report throughput (tokens/sec), latency percentiles, and cache utilization (% of allocated blocks holding real data — this is the number that shows why paging wins). Results dumped to JSON, committed to the repo.

Phase 5 — Speculative decoding (optional but high-value). Add Qwen2.5-0.5B as the draft model for a 3B target. The subtle part is verification: accepting the longest prefix of draft tokens the big model agrees with. Benchmark the speedup and the acceptance rate.

Deliverable: one repo — engine code, tests, benchmark scripts, and a README that explains the design and shows the numbers with a chart. The README is half the project's interview value; write it like an explanation to a peer, including what didn't work.

## How Claude should work on this project

These apply to every session in this repo.

**Response length.** Keep responses focused, brief, and concise. Keep disclaimers and caveats short, and spend most of the response on the main answer. When asked to explain something, give a high-level summary unless an in-depth explanation is specifically requested.

**Progress updates.** Before the first tool call, say in one sentence what you're about to do. While working, give a brief update only when you find something important or change direction. When you finish, lead with the outcome: the first sentence answers "what happened" or "what did you find," with supporting detail after it.

**Teaching as you go.** After finishing a piece of work, explain what it does in plain language — the user is learning this material, not just shipping it. Explain the *why* behind design choices and name the gotchas hit along the way. This explanation is the point of the response, so it does not count as padding.

**Commits.** The user commits and pushes; Claude does not run git commit. At the end of every phase, write out the commit message for the user to copy. Make it a real summary of what happened in the phase — design decisions, headline numbers, and bugs hit — not just a file list.

**Written deliverables.** Match the length of written documents (README, design notes, benchmark writeups) to what the task needs: cover the substance, but do not pad with filler sections, redundant summaries, or boilerplate.

**Task scope.** Deliver what was asked, at the scope intended. Make routine judgment calls yourself, and check in only when different readings of the request would lead to materially different work. If the request seems mistaken or a better approach exists, say so in a sentence and continue with the task as asked rather than quietly narrowing, widening, or transforming it. Finish the whole task, and stop short of actions clearly beyond what was asked.

**Subagents.** Delegate only for large, genuinely independent and parallelizable work, such as a wide multi-file investigation. Do not delegate work finishable in a handful of tool calls, and do not use subagents to verify your own work. Prefer one agent over several; keep spawn counts low.

**Corrections.** Only correct an earlier statement when the error would change the code, conclusions, or decisions. State corrections plainly and briefly, then continue. For slips that change nothing, fix and move on without noting it.