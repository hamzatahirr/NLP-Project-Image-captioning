import os
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import add
from pickle import load
import numpy as np
from PIL import Image

app = Flask(__name__)

# -----------------------------------
# Load tokenizer & model
# -----------------------------------
TOKENIZER_PATH = "saved/tokenizer.p"
MODEL_WEIGHTS_PATH = "saved/model_1.h5"

tokenizer = load(open(TOKENIZER_PATH, "rb"))
vocab_size = len(tokenizer.word_index) + 1
max_length = 32

# Faster lookup dictionary
index_to_word = {v: k for k, v in tokenizer.word_index.items()}


# -----------------------------------
# Build model architecture
# -----------------------------------
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


model = define_model(vocab_size, max_length)
model.load_weights(MODEL_WEIGHTS_PATH)

# Xception feature extractor
xception_model = Xception(include_top=False, pooling="avg")


# -----------------------------------
# Extract features from uploaded image
# -----------------------------------
def extract_features_from_buffer(image_buffer, model):
    try:
        image = Image.open(BytesIO(image_buffer)).convert("RGB")
    except:
        return None

    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    try:
        feature = model.predict(image)
    except Exception as e:
        print("🔥 Feature extraction error:", e)
        return None

    return feature


# -----------------------------------
# Convert predicted index → word
# -----------------------------------
def word_for_id(integer):
    return index_to_word.get(integer, None)


# -----------------------------------
# Caption generator
# -----------------------------------
def generate_desc(model, tokenizer, photo, max_length):
    in_text = "start"

    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat)
        if word is None:
            break

        in_text += " " + word

        if word == "end":
            break

    return in_text


# -----------------------------------
# ROUTES
# -----------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image file required"}), 400

        file = request.files["image"]
        image_bytes = file.read()

        # Extract features
        photo = extract_features_from_buffer(image_bytes, xception_model)
        if photo is None:
            return jsonify({"error": "Invalid image or processing failure"}), 400

        # Generate caption
        caption = generate_desc(model, tokenizer, photo, max_length)
        caption = caption.replace("start", "").replace("end", "").strip()

        return jsonify({"caption": caption})

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return jsonify({"error": "Server crashed: " + str(e)}), 500


# -----------------------------------
# Server Runner
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
