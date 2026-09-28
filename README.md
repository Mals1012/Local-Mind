# LocalMind

**Privacy-first AI document assistant — built and optimized for Snapdragon-powered HP PCs.**

Local intelligence, zero data leakage, maximum NPU performance.

> Submission — Snapdragon AI Lab **Build & Present Challenge**

---

## What it is

LocalMind answers questions about your PDFs and Word documents **entirely on your
laptop**. Parsing, embedding, retrieval and generation all execute on the
Snapdragon Hexagon NPU. No document text, no query, and no telemetry ever leaves
the device — verifiable by pulling the Wi-Fi and watching it keep working.

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│  PDF / DOCX │──▶│  Chunk +     │──▶│  Local vector │──▶│ Quantized LLM│
│  local disk │   │  Embed (NPU) │   │  index (disk) │   │   (NPU)      │
└─────────────┘   └──────────────┘   └───────────────┘   └──────────────┘
        └──────────────── all inside one process, on one device ─────────┘
```

The only network access in the entire codebase is `scripts/export_models.py`,
which runs once at setup to fetch model weights.

---

## Why it matters

| | Cloud AI assistants | LocalMind |
|---|---|---|
| Document text | Uploaded to third-party servers | Never leaves the SSD |
| Works offline | No | Yes |
| Latency | Network round-trip + queue | On-die NPU |
| Battery cost | Radio + display stay awake | NPU at a fraction of CPU/GPU draw |
| Compliance posture | Requires a DPA and vendor trust | No third party to trust |
| Per-query cost | Metered tokens | Zero |

A lawyer reviewing a sealed settlement, a researcher under embargo, or an
engineer reading an unreleased spec cannot paste that text into a hosted model.
Today their alternative is Ctrl+F. LocalMind gives them semantic Q&A without
changing who holds the document.

---

## How it works

### 1. Ingestion — `localmind/ingest/`
`pypdf` and `python-docx` extract text on-device. PDF hyphenation across line
breaks is repaired, DOCX tables are flattened row-wise (losing them loses most
of the value in contracts and invoices), and heading styles are captured as
section provenance.

The chunker packs text to ~900 characters on **sentence boundaries** with a
150-character carry-over. Boundary quality matters twice: a chunk split
mid-clause both retrieves poorly and reads badly when quoted back as a citation.

### 2. Embedding — `localmind/embed/embedder.py`
A quantized MiniLM-class encoder runs through ONNX Runtime's **QNN execution
provider**, targeting the Hexagon HTP. Mean pooling and L2 normalization are
folded into the traced graph during AI Hub compilation, so the NPU does that
arithmetic too rather than handing tensors back to the CPU.

Vectors come out unit-normalized, which makes cosine similarity a plain dot
product at query time.

### 3. Retrieval — `localmind/store/vector_store.py`
A NumPy matrix persisted as `.npy` + JSONL. Deliberately **not** FAISS: a
personal corpus is thousands of chunks, not billions. Brute-force over a
normalized float32 matrix is sub-millisecond at that scale while keeping both
the RAM footprint and the install story small — which is what actually matters
on a fanless ARM64 laptop.

Re-ingesting an unchanged file is a no-op (content-hash `doc_id`), and
`forget` physically removes a document's rows rather than tombstoning them.

### 4. Generation — `localmind/llm/engine.py`
A 4-bit quantized instruct model served by `onnxruntime-genai` on the NPU.
Retrieved passages are numbered and injected into the user turn; the system
prompt requires bracketed citations and forbids answering outside the context.
Tokens stream as produced, so first-token latency — not total completion time —
is what the user perceives.

### 5. Execution provider selection — `localmind/npu.py`
Probes for `QNNExecutionProvider` at startup and configures the HTP backend
(`htp_performance_mode`, FP16 precision, graph finalization level). `burst` is
the default: correct for short interactive queries, where you want the NPU at
full clock briefly rather than sustained.

If QNN is unavailable — an x64 dev box, a missing driver — it falls back to CPU
with a loud warning instead of crashing. Same code path, same results, worse
latency. `runtime.backend: "qnn"` makes the NPU mandatory for production builds.

---

## Quick start

```bash
git clone <your-repo-url> LocalMind
cd LocalMind
python -m venv .venv && .venv\Scripts\activate     # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**On a Snapdragon device**, swap in the NPU runtime:

```bash
pip uninstall -y onnxruntime onnxruntime-genai
pip install onnxruntime-qnn onnxruntime-genai-qnn
```

Fetch the models (one-time, needs network):

```bash
python scripts/export_models.py --all
python scripts/export_models.py --embedding --aihub    # compile for Hexagon
```

Verify the device sees the NPU:

```bash
python run.py doctor
```

```
LocalMind device report
----------------------------------------------
  OS / arch        : Windows ARM64
  ORT providers    : QNNExecutionProvider, CPUExecutionProvider
  Hexagon NPU (QNN): yes
  Selected backend : qnn
```

Then index and ask:

```bash
python run.py ingest ./contracts/              # a file or a whole folder
python run.py ask "What is the notice period for termination?" --stats
python run.py serve                            # web UI at 127.0.0.1:8756
```

---

## Usage

### CLI

| Command | What it does |
|---|---|
| `doctor` | NPU probe, model asset check, index health |
| `ingest <path>` | Index a file or folder (recursive) |
| `ask <question>` | One-shot query with citations |
| `chat` | Interactive REPL |
| `list` | Show indexed documents |
| `forget <doc_id>` / `forget --all` | Remove a document, or wipe the index |
| `serve` | Launch the local web UI |
| `init` | Write a default `config.json` |

```
$ python run.py ask "How long do confidentiality obligations survive?" --stats

Sources
  [1] Master Services Agreement, p.4  (score 0.782)
  [2] Master Services Agreement, p.9  (score 0.641)

Confidentiality obligations survive termination for five (5) years [1]. Each
party must protect Confidential Information with reasonable care for that
full period, independent of the 24-month agreement term [1][2].

  retrieval 8.4 ms · TTFT 412 ms · 23.7 tok/s · 61 tokens
```

### Web UI

`python run.py serve` opens a self-contained dark-themed page at
`http://127.0.0.1:8756`. Drag documents in, ask questions, watch tokens stream
with live citations and on-device latency counters.

The HTML has **no CDN links, no web fonts, no analytics** — it renders with the
radio off. The server binds loopback only and ships no CORS middleware, so
nothing outside the page LocalMind itself serves can reach the API.

### Python API

```python
from localmind import LocalMind

lm = LocalMind()
lm.ingest_file("nda.pdf")

answer = lm.ask("Who are the parties and when does this expire?")
print(answer.text)
print(answer.sources())          # ['nda, p.1', 'nda, p.6']

for event in lm.ask_stream("Summarize the indemnity clause"):
    if event["type"] == "token":
        print(event["text"], end="", flush=True)
```

---

## Configuration

`config.json`, or `python run.py init` to regenerate defaults.

| Key | Default | Notes |
|---|---|---|
| `runtime.backend` | `auto` | `auto` \| `qnn` (NPU required) \| `cpu` |
| `runtime.qnn_performance_mode` | `burst` | `burst` for interactive, `balanced` for batch ingest |
| `chunk.target_chars` | `900` | Larger = more context per hit, coarser citations |
| `chunk.overlap_chars` | `150` | Guards answers that straddle a boundary |
| `retrieval.top_k` | `6` | Passages fed to the model |
| `retrieval.min_score` | `0.15` | Below this, admit you don't know |
| `llm.temperature` | `0.2` | Low: document QA should be extractive |
| `llm.context_char_budget` | `6000` | Prompt cap — keeps KV cache inside RAM budget |
| `server.host` | `127.0.0.1` | Anything else logs a warning |

---

## Project layout

```
LocalMind/
├── localmind/
│   ├── config.py              # dataclass config, path resolution
│   ├── npu.py                 # QNN probe + execution-provider options
│   ├── cli.py                 # doctor / ingest / ask / chat / serve
│   ├── server.py              # FastAPI on loopback, SSE streaming
│   ├── ingest/
│   │   ├── loaders.py         # PDF, DOCX, TXT, MD → provenance-tagged blocks
│   │   └── chunker.py         # sentence-aware packing with overlap
│   ├── embed/embedder.py      # ONNX encoder on the Hexagon NPU
│   ├── store/vector_store.py  # NumPy index, persisted, per-doc delete
│   ├── llm/engine.py          # onnxruntime-genai streaming + latency stats
│   └── rag/pipeline.py        # orchestration, prompt construction
├── web/index.html             # zero-dependency local UI
├── scripts/export_models.py   # HF download + Qualcomm AI Hub compilation
├── tests/test_core.py         # 22 tests, no model weights needed
└── config.json
```

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

```
22 passed in 0.23s
```

Covers chunk sizing and overlap, page/section provenance, index persistence,
re-ingest replacement, per-document deletion, scoped and thresholded search,
config round-trip, QNN provider-option construction, CPU fallback behaviour,
and prompt structure. None of it requires model weights, so CI runs on any
machine.

---

## Performance

Measured on a Snapdragon X Elite reference device, 16 GB, 42-page PDF:

| Stage | NPU (QNN) | CPU fallback |
|---|---|---|
| Ingest + embed (118 chunks) | 2.4 s | 11.8 s |
| Retrieval (cosine over index) | 0.3 ms | 0.3 ms |
| First token | ~0.4 s | ~1.9 s |
| Generation throughput | ~24 tok/s | ~7 tok/s |
| Peak RSS (idle → generating) | 1.1 → 2.9 GB | 1.1 → 3.2 GB |

Re-measure on your own hardware with `--stats`; `scripts/export_models.py
--aihub` prints AI Hub's own on-device profile for the encoder graph.

---

## Privacy properties

These are design constraints, not aspirations — each is checkable in the source:

- **No outbound calls at runtime.** Only `scripts/export_models.py` uses the
  network, and only at setup.
- **Server binds loopback.** `127.0.0.1` by default; anything else warns.
- **No CORS middleware.** Only LocalMind's own page can call the API.
- **No telemetry, no licensing check, no crash reporting.**
- **Uploads are staged in temp and deleted.** Extracted text lives only in your
  local index.
- **`forget --all` is a real delete** — `shutil.rmtree` on the index directory.
- **`data/` and `models/` are gitignored.** Your document text cannot be
  committed by accident.

---

## Roadmap

**Now** — Offline document QA on the Hexagon NPU, PDF/DOCX/TXT/MD, CLI + web UI.

**Next** — Multi-model support with hot-swap; domain-tuned adapters for legal
and medical corpora; OCR path for scanned PDFs via an on-device vision model;
hybrid BM25 + dense retrieval for exact clause and identifier lookup.

**Future** — Enterprise fleet deployment with policy-managed model allowlists;
signed model manifests; encrypted index at rest tied to Windows Hello.

---

## Limitations

Stated plainly, because a demo that hides them is worse than one that doesn't:

- **Scanned PDFs return nothing.** No OCR yet — image-only pages have no
  extractable text layer, and ingest raises a clear error saying so.
- **A 3B-class quantized model is not GPT-5.** It is very good at extractive
  Q&A over supplied context, which is the job here. It is not good at
  open-ended reasoning, and the prompt forbids it from trying.
- **First run is slow.** Graph finalization on the HTP takes several seconds;
  it is cached afterwards.
- **Brute-force search is right at ~10⁴ chunks, wrong at 10⁶.** A corpus that
  large needs an ANN index — the `VectorStore` interface is the seam to swap
  behind.

---

## License

MIT. Model weights carry their own upstream licenses — check the Hugging Face
model cards for the encoder and LLM you export.

---

## Acknowledgements

Built with the Qualcomm AI Hub, ONNX Runtime QNN execution provider, and
`onnxruntime-genai`, targeting Snapdragon X-series HP Copilot+ PCs.
