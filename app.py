import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM,
    Dense, Activation, BatchNormalization, Dropout
)

ALPHABETS = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
NUM_CHARACTERS = len(ALPHABETS) + 1   
WEIGHTS_PATH = "handwriting_model.weights.h5"  

def num_to_name(num_sequence):
    """Map decoded indices to characters; stop at -1 (CTC blank)."""
    name = ""
    for ch in num_sequence:
        if ch == -1:
            break
        name += ALPHABETS[int(ch)]
    return name

def image_processing(img_gray):
    """
    Preprocess image:
      - pad/crop to (64, 256)
      - rotate 90° clockwise
      - normalize to [0,1]
      - reshape to (1, 256, 64, 1)
    """
    if img_gray is None:
        raise ValueError("Invalid image.")

    h, w = img_gray.shape[:2]
    final_img = np.ones((64, 256), dtype=np.float32) * 255.0

    # crop if larger
    if w > 256:
        img_gray = img_gray[:, :256]
        w = 256
    if h > 64:
        img_gray = img_gray[:64, :]
        h = 64

    # place on white canvas
    final_img[:h, :w] = img_gray.astype(np.float32)

    # rotate and normalize
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    final_img = final_img / 255.0

    return final_img.reshape(1, 256, 64, 1).astype(np.float32)

def build_model():
    """Rebuild CRNN (CNN + BiLSTM) architecture."""
    input_data = Input(shape=(256, 64, 1), name='input')

    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.3)(x)

    x = Reshape(target_shape=(64, 1024))(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)

    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)

    x = Dense(NUM_CHARACTERS, kernel_initializer='he_normal')(x)
    y_pred = Activation('softmax', name='softmax')(x)

    return Model(inputs=input_data, outputs=y_pred, name="crnn_ctc_infer")

@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

st.title("Handwriting Recognition App")
st.write("Upload a handwritten image, and the model will try to recognize the text.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img_gray, caption="Uploaded Image", use_container_width=True, channels="GRAY")

    tensor = image_processing(img_gray)
    preds = model.predict(tensor)

    decoded, _ = tf.keras.backend.ctc_decode(
        preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True
    )
    decoded_seq = decoded[0].numpy()[0]
    text = num_to_name(decoded_seq)

    st.subheader("Prediction:")
    st.success(text if text else "[empty prediction]")
