import torch
import pickle
from flask import Flask, render_template, request

# -----------------------------
# 1. Load model safely on CPU
# -----------------------------
model_path = "sentiment_analysis.pkl"

try:
    # Try loading as PyTorch checkpoint
    sentiment_pipeline = torch.load(
        model_path,
        map_location=torch.device('cpu'),  # force CPU
        weights_only=False
    )
    sentiment_pipeline.eval()
    print("Loaded model using torch.load")
except (RuntimeError, pickle.UnpicklingError):
    # Fallback: load as standard Python pickle
    with open(model_path, "rb") as f:
        sentiment_pipeline = pickle.load(f)
    print("Loaded model using pickle.load")

# -----------------------------
# 2. Setup Flask
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form["text"]

        try:
            with torch.no_grad():
                # If model is PyTorch nn.Module
                if isinstance(sentiment_pipeline, torch.nn.Module):
                    # Adjust if your model expects tensors
                    result = sentiment_pipeline(text_input)
                else:
                    # For pickle or Hugging Face-like pipelines
                    result = sentiment_pipeline(text_input)

                # Extract label if result is a list of dicts
                if isinstance(result, list) and "label" in result[0]:
                    sentiment = result[0]["label"]
                else:
                    sentiment = str(result)
        except Exception as e:
            sentiment = f"Error during inference: {e}"

    return render_template("index.html", sentiment=sentiment, text_input=text_input)

# -----------------------------
# 3. Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
