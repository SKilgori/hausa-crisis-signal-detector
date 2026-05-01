import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================
# HAUSA CRISIS SIGNAL DETECTOR
# Built by Sadiya Muhammad Kilgori
# Classifies Hausa-language text into humanitarian crisis categories
# to support early warning systems in Northern Nigeria
# ============================================================

# After training, replace this with your actual model path:
# MODEL_PATH = "YOUR_USERNAME/hausa-crisis-signal-detector"
# For now we use a placeholder — swap it after training
MODEL_PATH = "Skilgori/hausa-crisis-signal-detector"

# Crisis category descriptions
CATEGORY_INFO = {
    "conflict": {
        "emoji": "⚔️",
        "hausa": "Rikici / Tashin Hankali",
        "description": "Violence, armed conflict, or security incidents"
    },
    "displacement": {
        "emoji": "🏃",
        "hausa": "Gudun Hijira",
        "description": "People forced to flee their homes"
    },
    "food_insecurity": {
        "emoji": "🌾",
        "hausa": "Karancin Abinci / Yunwa",
        "description": "Hunger, food shortages, or nutrition crises"
    },
    "disease_outbreak": {
        "emoji": "🏥",
        "hausa": "Annoba / Cutar Yaduwa",
        "description": "Infectious disease outbreak or health emergency"
    },
    "flood": {
        "emoji": "🌊",
        "hausa": "Ambaliyar Ruwa",
        "description": "Flooding or water-related disaster"
    },
    "no_crisis": {
        "emoji": "✅",
        "hausa": "Ba Matsala Ba",
        "description": "No crisis signal detected"
    }
}

# Load model
print("Loading Hausa Crisis Signal Detector...")
try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        top_k=None
    )
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model not yet available: {e}")
    model_loaded = False


def classify_text(text):
    """Classify Hausa text into crisis categories."""
    if not text or len(text.strip()) < 5:
        return "Please enter some Hausa text to classify.", None

    if not model_loaded:
        return "Model is not yet loaded. Please train and push the model first.", None

    try:
        results = classifier(text)[0]
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

        top_label = results_sorted[0]["label"]
        top_score = results_sorted[0]["score"]

        info = CATEGORY_INFO.get(top_label, {})

        # Build output
        output_lines = []
        output_lines.append(f"## {info.get('emoji', '')} Detected: {info.get('hausa', top_label)}")
        output_lines.append(f"**English:** {info.get('description', '')}")
        output_lines.append(f"**Confidence:** {top_score:.1%}")
        output_lines.append("")
        output_lines.append("### All Scores")

        for r in results_sorted:
            label = r["label"]
            score = r["score"]
            cat = CATEGORY_INFO.get(label, {})
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            output_lines.append(
                f"{cat.get('emoji', '')} **{cat.get('hausa', label)}**: {bar} {score:.1%}"
            )

        return "\n".join(output_lines)

    except Exception as e:
        return f"Error during classification: {str(e)}"


# Example texts
EXAMPLES = [
    ["An kashe mutane da yawa a harin da aka kai a kauyenmu jiya da dare"],
    ["Mun bar gidanmu saboda tsoron hare-haren boko haram, muna gudun hijira yanzu"],
    ["Yara suna mutuwa saboda yunwa, babu abinci a wannan gari"],
    ["Cutar kwalara ta bazu a duk fadin karamar hukuma, asibiti ya cika da marasa lafiya"],
    ["Ambaliyar ruwa ta mamaye gidaje da gonakin manoma, mutane sun rasa komai"],
    ["Yara suna tafi makaranta lafiya lau a yau, malamai sun zo aikin su"],
]

# Build Gradio interface
with gr.Blocks(
    title="Hausa Crisis Signal Detector",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 800px; margin: auto; }
    .title { text-align: center; color: #1a3a6b; }
    .subtitle { text-align: center; color: #555; font-size: 14px; }
    """
) as demo:

    gr.HTML("""
    <div class='title'>
        <h1>🔍 Hausa Crisis Signal Detector</h1>
    </div>
    <div class='subtitle'>
        <p>Classifies Hausa-language text into humanitarian crisis categories to support
        early warning systems in Northern Nigeria and the Sahel.</p>
        <p><em>Built by <strong>Sadiya Muhammad Kilgori</strong> | Powered by AfriBERTa</em></p>
    </div>
    <hr>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            text_input = gr.Textbox(
                label="Enter Hausa text",
                placeholder="Rubuta rubutun Hausa a nan...\n(Enter Hausa text here)",
                lines=5
            )
            submit_btn = gr.Button("Classify / Rarrabawa", variant="primary")
            gr.Markdown("### Try these examples:")
            examples = gr.Examples(
                examples=EXAMPLES,
                inputs=text_input,
                label=""
            )

        with gr.Column(scale=1):
            gr.Markdown("### Result / Sakamakon")
            output = gr.Markdown(
                value="Results will appear here after classification.",
                label="Classification Result"
            )

    submit_btn.click(
        fn=classify_text,
        inputs=text_input,
        outputs=output
    )

    gr.HTML("""
    <hr>
    <div style='text-align:center; color:#888; font-size:12px;'>
        <p>Crisis Categories: Conflict (Rikici) | Displacement (Gudun Hijira) |
        Food Insecurity (Yunwa) | Disease Outbreak (Annoba) | Flood (Ambaliya) | No Crisis</p>
        <p>This tool is intended to support humanitarian early warning systems.
        Always verify signals with ground-level sources.</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
