# [Presentation link](https://predicting-clinical-tria-bqou0l0.gamma.site/)
# Clinical Trial Success Predictor

This part of the project holds the data, documents, and artifacts used to build the RAG component and to model endpoint success in oncology trials. The layout is organized for reproducibility, easy navigation, and reuse of individual pieces.

## Directory layout
```text
src/
|-- clinical_agent/           # core agent logic: retrieval, priors, LLM runner
notebooks/
|-- RAG_clin_trials.ipynb     # notebook to build the RAG corpus/index
data/
|-- results/                  # optional batch-eval logs (jsonl)
|-- rag/
|   |-- rag_data/             # PDF knowledge-base documents
|   |-- rag_index/            # FAISS index and chunk metadata
|   `-- docs_manifest.csv     # manifest with per-document metadata
`-- validation/               # tabular data and helpers for model evaluation
```

## rag_data/

This folder contains materials describing:
- design and assessment of clinical trials;
- regulatory guidance from FDA/EMA/ICH;
- statistical analysis methodologies;
- endpoint classifications and response criteria;
- studies on factors of clinical-program success.

All documents were checked for relevance, authoritative source, and publication date to use only data available at the time of the validation trials.

## How the RAG corpus was assembled

The corpus was hand-curated and includes only authoritative primary sources that are actually used in oncology research and drug registration. Selection steps:

1. **Regulatory documents (FDA, EMA, ICH)**  
   Sourced from official regulator sites; guidance/guideline-level materials. They describe acceptable endpoints, response criteria, statistical principles (controls, analysis types), imaging rules (BICR), and handling of missing data. These form the "skeleton" - the regulatory backbone of trial design.

2. **Scientific reviews (PubMed/PMC)**  
   Review articles explaining endpoint concepts, RECIST/iRECIST principles, effect estimation, and interpretation. They give clear explanations that help the agent extract and contextualize facts.

3. **Industry analytical reports (BIO and others)**  
   Studies on success rates of clinical programs and factors influencing outcomes. They let the agent reason about success probabilities and typical failure causes.

Selection criteria:
- official or published in a recognized scientific source;
- focused on oncology or general clinical statistical principles;
- clear and suitable for automatic analysis;
- no secondary retellings or dubious material.

The resulting RAG corpus is a carefully constructed library that explains how clinical trials are run and assessed, giving the model a "professional context" and improving output quality.

## docs_manifest.csv

Metadata for each document:
- title - name;
- file_path - path inside rag_data/;
- url - source;
- publication_date - publication date;
- source - organization/journal;
- tags - topic areas.

The manifest keeps the corpus composition controlled and reproducible when rebuilding the index.

## rag_index/

Artifacts required for RAG:
- rag.faiss - vector index (FAISS);
- rag_meta.jsonl - metadata for each chunk;
- embed_model.txt - embedding model info;
- stats.json - document and chunk counts.

Document text is extracted, normalized, and chunked (~900 tokens, ~150 overlap). Embeddings are generated with intfloat/e5-base.

## validation/

Contains the tabular validation set and helper files for testing the pipeline.

## Data-processing flow

1. Load and verify documents.  
2. Describe each document in docs_manifest.csv.  
3. Extract text, normalize, chunk.  
4. Generate embeddings and build the FAISS index.  
5. Use the index in the RAG component to fetch relevant context.

## Structure advantages

- Clear separation of raw data, index, and metadata.
- Ability to fully rebuild the RAG layer when documentation updates.
- Transparent, controlled corpus composition.
- Ready for agent integration.

## Adding new documents

1. Drop the PDF into rag/rag_data/.  
2. Add a row to docs_manifest.csv.  
3. Rebuild the index.  
4. Confirm stats.json and rag_meta.jsonl are updated.

## Streamlit MVP agent

A simple agent in the repo root scores a single endpoint with RAG, priors, and flags (red/yellow/green).

### Run
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...             # key for an OpenAI-compatible API
# when using Nebius:
# export OPENAI_BASE_URL="https://api.studio.nebius.ai/v1"
streamlit run app.py
```

### What it does
- Choose an endpoint from the validation set or provide JSON.
- The agent pulls context from `data/rag/rag_index`, applies a prior by phase/endpoint type, adds gentle penalties (small-N survival, single-arm, refractory) and a simple biomarker flag.
- The LLM returns probability, flags, and rationale with citations; the system blends with the prior and shows applied penalties/flags.

### Quality evaluation
- Planned script over `train_labels.csv` (precision/recall/F1) can be run separately; the UI is for manual checks and demos.

### Batch evaluation (script)
```bash
# Example: evaluate 30 random endpoints
conda activate clin-agent  # or your env
export OPENAI_API_KEY=...       # + OPENAI_BASE_URL, OPENAI_MODEL if needed
python eval_agent.py --limit 30 --threshold 0.5 --output data/results/results.jsonl
```
Arguments:
- `--limit` (default 20; -1 = all labels) - number of endpoints to evaluate;
- `--threshold` - decision threshold on blended probability;
- `--seed` - for random sampling;
- `--output` - path for JSONL logs with probabilities/rationales (optional);
- `--model` - override model name (otherwise uses `OPENAI_MODEL`).

Sample logs live in `data/results/`.

### Predictions for test.csv (submission)
```bash
conda activate clin-agent  # or your env
export OPENAI_API_KEY=...       # + OPENAI_BASE_URL, OPENAI_MODEL if needed
python predict_test.py --output data/results/sample_submission.csv --details data/results/test_details.jsonl
```
Arguments:
- `--output` - CSV in format `endpoint_id,endpoint_criterion_met`;
- `--details` - optional JSONL with probabilities/rationales;
- `--threshold` - decision threshold (default 0.5);
- `--limit` - score only the first N test endpoints (for debugging).
