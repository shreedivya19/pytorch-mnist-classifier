import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from src.model import SimpleNN

# Page configuration
st.set_page_config(
    page_title="PyTorch MNIST Digit Classifier",
    page_icon="🧠",
    layout="wide"
)

# Load your trained model
@st.cache_resource
def load_model():
    model = SimpleNN()
    model.load_state_dict(
        torch.load(
            'models/saved_models/mnist_simple_model.pth',
            map_location=torch.device('cpu')
        )
    )
    model.eval()
    return model

def preprocess_image(image_data):
    """Convert drawn image to MNIST format using PIL (no OpenCV)."""
    if image_data is None:
        return None

    # Convert RGBA numpy array to PIL image, convert to grayscale
    pil_img = Image.fromarray(image_data).convert("L")

    # Resize to 28×28 with LANCZOS (replacement for ANTIALIAS)
    pil_img = pil_img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array and normalize to [0,1]
    img = np.array(pil_img).astype("float32") / 255.0

    # Convert to PyTorch tensor and add batch & channel dims
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    return img_tensor

def predict_digit(model, image_tensor):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_digit = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_digit].item()
    return predicted_digit, confidence, probabilities[0]

# Main Streamlit app
def main():
    st.title("🧠 PyTorch MNIST Digit Classifier")
    st.markdown("### Draw a digit (0-9) and watch AI predict it in real time!")

    # Model information expander
    with st.expander("📊 Model Information"):
        st.write("- **Architecture**: Simple Feedforward Neural Network")
        st.write("- **Parameters**: 101,770")
        st.write("- **Test Accuracy**: 97.81%")
        st.write("- **Training Time**: ~10 seconds")

    # Split layout into two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✏️ Draw Here")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas"
        )
        if st.button("🗑️ Clear Canvas"):
            st.rerun()

    with col2:
        st.markdown("#### 🔮 Prediction Results")
        model = load_model()

        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
            image_tensor = preprocess_image(canvas_result.image_data)
            if image_tensor is not None:
                predicted_digit, confidence, all_probs = predict_digit(model, image_tensor)

                st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                st.markdown(f"### Confidence: **{confidence:.2%}**")

                st.markdown("#### 📊 Probability Distribution")
                prob_data = {
                    'Digit': list(range(10)),
                    'Probability': [float(p) for p in all_probs]
                }
                st.bar_chart(prob_data, x='Digit', y='Probability')

                st.markdown("#### 🖼️ Processed Image (28×28)")
                processed_img = image_tensor.squeeze().numpy()
                st.image(processed_img, width=150, use_container_width=False)
        else:
            st.info("👆 Draw a digit on the canvas to see predictions!")

if __name__ == "__main__":
    main()
