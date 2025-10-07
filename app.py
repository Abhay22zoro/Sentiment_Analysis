import torch
import pickle
from flask import Flask, render_template, request

# -----------------------------
# 1. Load Model on CPU Safely
# -----------------------------
model_path = "sentiment_analysis.pkl"

try:
    # Try to load as PyTorch object (map to CPU)
    with open(model_path, "rb") as f:
        sentiment_pipeline = torch.load(f, map_location=torch.device("cpu"))
    print("✅ Model loaded using torch.load (mapped to CPU).")

except Exception as e:
    print(f"⚠️ torch.load failed ({e}), trying pickle.load...")
    try:
        # Fallback for plain pickle
        with open(model_path, "rb") as f:
            sentiment_pipeline = pickle.load(f)
        print("✅ Model loaded using pickle.load.")
    except Exception as e2:
        raise RuntimeError(f"❌ Failed to load model both ways: {e2}")

# -----------------------------
# 2. Setup Flask
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form.get("text", "")

        try:
            with torch.no_grad():
                # HuggingFace pipeline or torch.nn.Module
                if callable(sentiment_pipeline):
                    result = sentiment_pipeline(text_input)
                else:
                    raise ValueError("Loaded object is not callable.")

                # Handle Hugging Face-like output
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    sentiment = result[0].get("label", "Unknown")
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
