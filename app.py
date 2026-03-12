import gradio as gr
from model import load_model, predict_sarcasm

# Load model once when app starts
model, tokenizer = load_model()

def inference(text):
    label, prob, uncertainty = predict_sarcasm(text, model, tokenizer)

    return {
        "Prediction": label,
        "Confidence": f"{prob:.4f}",
        "Uncertainty": f"{uncertainty:.4f}"
    }

demo = gr.Interface(
    fn=inference,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter Malayalam sentence here..."
    ),
    outputs="json",
    title="Malayalam Sarcasm Detection with Uncertainty",
    description="Detect sarcasm in Malayalam text using DeBERTa with uncertainty estimation."
)

demo.launch()
