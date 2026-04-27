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
- **Interface:** Gradio
- **Deployment:** Hugging Face Spaces

---

## Project Structure

```
hausa_crisis_detector/
├── data/
│   └── hausa_crisis_data.csv     # Labeled training data
├── app.py                         # Gradio web app
├── train.py                       # Training script (run in Google Colab)
├── requirements.txt               # Dependencies
└── README.md
```

---

## How to Train

1. Open [Google Colab](https://colab.research.google.com)
2. Go to **Runtime > Change Runtime Type > GPU**
3. Upload `hausa_crisis_data.csv` and `train.py`
4. Run `train.py` cell by cell
5. Push model to your Hugging Face Hub account
6. Update `MODEL_PATH` in `app.py` with your model path

---

## How to Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload `app.py` and `requirements.txt`
4. The Space will build and deploy automatically

---

## Dataset

The training dataset contains labeled Hausa-language sentences covering all six crisis categories, drawn from contexts relevant to Northern Nigeria and the Lake Chad Basin region. The dataset will be expanded continuously.

**To contribute data:** Open an issue on the GitHub repository with additional labeled examples.

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
GitHub. https://github.com/Kilgori/hausa-crisis-signal-detector
```

---

## Roadmap

- [ ] Expand dataset to 500+ labeled examples
- [ ] Add support for mixed Hausa-English (code-switched) text
- [ ] Build API endpoint for integration with humanitarian platforms
- [ ] Add confidence threshold alerts for high-risk signals
- [ ] Partner with Northern Nigeria community radio stations for real-time monitoring
