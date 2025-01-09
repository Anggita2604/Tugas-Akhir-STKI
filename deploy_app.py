import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
@st.cache_resource  # Cache model to avoid reloading
def load_model():
    return tf.keras.models.load_model('model_densenet.h5')  # Path ke model Anda

model = load_model()

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Title and description
st.title("Luxury Bag (Gucci) Authentication Tools")
st.write("""
            Aplikasi ini memungkinkan Anda untuk mengunggah gambar tas mewah (Gucci) 
            dan akan memberikan prediksi apakah tas tersebut asli (ORI) atau palsu (KW).
        """)

# Upload image
uploaded_file = st.file_uploader("Unggah gambar tas Anda di sini:", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')  # Pastikan konversi ke RGB
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)

    # Preprocessing
    img_array = np.array(image.resize((256, 256))) / 255.0  # Resize dan normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display results
    st.write("### Hasil Prediksi:")
    st.write(f"**{predicted_class}**")
    st.write(f"**Kepercayaan:** {confidence:.2f}%")
    
    # Display additional messages
    if predicted_class == "ORI":  # Cocokkan langsung sesuai isi class_names.json
        st.success("Selamat, tas Anda terdeteksi sebagai tas Gucci **Asli (Authentic)**!")
    elif predicted_class == "KW":
        st.error("Maaf, tas Anda terdeteksi sebagai tas Gucci **Palsu (KW)**.")
    else:
        st.warning("Hasil prediksi tidak dapat dipastikan.")


