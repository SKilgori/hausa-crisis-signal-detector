# 🔍 Hausa Crisis Signal Detector

**Built by Sadiya Muhammad Kilgori**

A natural language processing tool that classifies Hausa-language text into humanitarian crisis categories to support early warning systems in Northern Nigeria and the Sahel.

---

## Why This Exists

Northern Nigeria is home to over 70 million Hausa speakers and is one of the regions most affected by conflict, displacement, food insecurity, and climate-related disasters. Yet almost all digital humanitarian tools — early warning systems, crisis monitoring platforms, and response coordination tools — operate in English or French.

This project builds the infrastructure to change that: a classifier that can read Hausa text from social media, news sources, or community reports, and flag signals of humanitarian crisis before they escalate.

---

## Crisis Categories

| Category | Hausa | Description |
|---|---|---|
| `conflict` | Rikici / Tashin Hankali | Armed conflict, violence, security incidents |
| `displacement` | Gudun Hijira | People forced to flee their homes |
| `food_insecurity` | Yunwa / Karancin Abinci | Hunger, food shortages, nutrition crises |
| `disease_outbreak` | Annoba / Cutar Yaduwa | Infectious disease or health emergency |
| `flood` | Ambaliyar Ruwa | Flooding or water-related disaster |
| `no_crisis` | Ba Matsala Ba | No crisis signal detected |

---

## Technical Stack

- **Model:** AfriBERTa Large (`castorini/afriberta_large`) — fine-tuned for Hausa crisis classification
- **Framework:** Hugging Face Transformers
- **Training:** Google Colab (free GPU)
- **Two separate deployment targets:**
  - **`app.py`** — Gradio demo, deployed on Hugging Face Spaces. Free, public, for showcasing the model.
  - **`api.py`** — FastAPI service, deployed as a metered commercial API via RapidAPI. Requires a `RAPIDAPI_PROXY_SECRET` and loads the model from a **local checkpoint**, not the public Hub — see the note in "How to Deploy the API" below on why.

---

## Project Structure

```
hausa-crisis-signal-detector/
├── data/
│   └── hausa_crisis_data.csv     # Labeled training data (canonical — see note below)
├── app.py                         # Gradio demo app (Hugging Face Spaces)
├── api.py                         # FastAPI commercial service (RapidAPI)
├── train.py                       # Training script (run in Google Colab)
├── requirements.txt               # Dependencies for app.py / HF Spaces
├── requirements-api.txt           # Dependencies for api.py
└── README.md
```

**Note on the data file:** `data/hausa_crisis_data.csv` is the single canonical dataset (deduplicated — see CHANGELOG below). A stale, truncated 39-row copy previously existed at the repo root and caused `train.py` to silently train on the wrong file when run per the old instructions; that root-level file should be deleted if it still exists in your checkout.

---

## How to Train

1. Open [Google Colab](https://colab.research.google.com)
2. Go to **Runtime > Change Runtime Type > GPU**
3. Clone the repo (or upload it) so `data/hausa_crisis_data.csv` is present at that relative path — `train.py` reads `data/hausa_crisis_data.csv` explicitly, not a bare filename
4. Run `train.py` cell by cell
5. Review the printed per-class classification report before doing anything else — check recall specifically on `conflict` and `disease_outbreak`, where a missed detection matters most
6. Model saves locally to `./hausa_crisis_model_final` by default. Pushing to the Hugging Face Hub is optional (`PUSH_TO_HUB = False` by default) — see the note below on why

---

## How to Deploy the Gradio Demo (Hugging Face Spaces)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload `app.py` and `requirements.txt`
4. Push the trained model to the Hub first (`PUSH_TO_HUB = True` in `train.py`) and update `MODEL_PATH` in `app.py` to match
5. The Space will build and deploy automatically

---

## How to Deploy the API (RapidAPI)

**Important:** `api.py` loads the model from a **local checkpoint path** (`MODEL_PATH` env var, defaults to `./hausa_crisis_model`), not a public Hugging Face repo. This is deliberate — if the fine-tuned weights are published publicly, anyone can download and self-host the model for free, which undermines using it as a paid, metered API. Keep the checkpoint private to your deployment (baked into a Docker image, or a private/gated Hub repo with token auth) rather than a public push.

1. Build a Docker image containing `api.py`, `requirements-api.txt`, and the trained checkpoint from `train.py`'s output directory
2. Deploy to a persistent host (not Colab — Colab's disk is ephemeral)
3. Set the `RAPIDAPI_PROXY_SECRET` environment variable to the secret RapidAPI assigns at listing time
4. List the API on RapidAPI, pointing its base URL at your deployed instance
5. `GET /health` is available unauthenticated for uptime monitoring; `POST /classify` requires the RapidAPI proxy secret

---

## Dataset

The training dataset contains labeled Hausa-language sentences covering all six crisis categories, drawn from contexts relevant to Northern Nigeria and the Lake Chad Basin region. The dataset will be expanded continuously.

**To contribute data:** Open an issue on the GitHub repository with additional labeled examples.

---

## Changelog

- **Fixed:** `TrainingArguments(evaluation_strategy=...)` renamed to `eval_strategy` in current `transformers` versions — training previously failed before it started.
- **Fixed:** `train.py` read a bare `hausa_crisis_data.csv` filename, which resolved to a stale 39-row stub at the repo root instead of the real dataset. Now reads `data/hausa_crisis_data.csv` explicitly.
- **Fixed:** 19 of 219 rows in the dataset were exact-duplicate texts, risking train/test leakage. Deduplicated to 200 rows.
- **Fixed:** `train.py` previously called `push_to_hub()` unconditionally with no login step executed, which would crash. Now optional and off by default.
- **Added:** `api.py` — a separate FastAPI service for commercial API deployment, alongside the existing Gradio demo.

---

## Author

**Sadiya Muhammad Kilgori**
MSc International Affairs and Diplomacy, Ahmadu Bello University
Sokoto, Nigeria

- LinkedIn: [linkedin.com/in/sadiya-muhammad](https://linkedin.com/in/sadiya-muhammad)
- DataCamp: [datacamp.com/portfolio/Kilgori](https://datacamp.com/portfolio/Kilgori)

---

## Citation

If you use this tool in research, please cite:

```
Kilgori, S.M. (2024). Hausa Crisis Signal Detector.
GitHub. https://github.com/SKilgori/hausa-crisis-signal-detector
```

---

## Roadmap

- [ ] Expand dataset to 500+ labeled examples (see note on template-based augmentation risk in `expand_data_v2.py` — structurally-identical templated sentences need careful train/test handling, not just exact-dedup)
- [ ] Add support for mixed Hausa-English (code-switched) text
- [x] Build API endpoint for integration with humanitarian platforms (`api.py`)
- [ ] Add confidence threshold alerts for high-risk signals
- [ ] Partner with Northern Nigeria community radio stations for real-time monitoring
