import pickle
from flask import Flask, render_template, request

# -----------------------------
# 1. Load Pickle Model
# -----------------------------
model_path = "sentiment_analysis.pkl"

try:
    with open(model_path, "rb") as f:
        sentiment_pipeline = pickle.load(f)
    print("✅ Model loaded successfully using pickle.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

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
            # For Hugging Face or sklearn pipeline
            result = sentiment_pipeline(text_input)

            # Handle Hugging Face-like output
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                sentiment = result[0].get("label", "Unknown")
            else:
                sentiment = str(result)
        except Exception as e:
            sentiment = f"Error during prediction: {e}"

    return render_template("index.html", sentiment=sentiment, text_input=text_input)

# -----------------------------
# 3. Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
