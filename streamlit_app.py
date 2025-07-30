import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
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
    model.load_state_dict(torch.load('models/saved_models/mnist_simple_model.pth', 
                                   map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_data):
    """Convert drawn image to MNIST format"""
    if image_data is None:
        return None
    
    # Convert to grayscale
    img = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Convert to tensor and add batch dimension
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

# Main app
def main():
    st.title("🧠 PyTorch MNIST Digit Classifier")
    st.markdown("### Draw a digit (0-9) and watch AI predict it in real-time!")
    
    # Model info
    with st.expander("📊 Model Information"):
        st.write("- **Architecture**: Simple Feedforward Neural Network")
        st.write("- **Parameters**: 101,770")
        st.write("- **Test Accuracy**: 97.81%")
        st.write("- **Training Time**: ~10 seconds")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ✏️ Draw Here")
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",  # Transparent
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            background_image=None,
            update_streamlit=True,
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Clear button
        if st.button("🗑️ Clear Canvas"):
            st.rerun()
    
    with col2:
        st.markdown("#### 🔮 Prediction Results")
        
        # Load model
        model = load_model()
        
        # Make prediction if something is drawn
        if canvas_result.image_data is not None:
            # Check if anything is drawn
            if np.any(canvas_result.image_data[:, :, 3] > 0):  # Alpha channel
                # Preprocess image
                image_tensor = preprocess_image(canvas_result.image_data)
                
                if image_tensor is not None:
                    # Make prediction
                    predicted_digit, confidence, all_probs = predict_digit(model, image_tensor)
                    
                    # Display prediction
                    st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                    st.markdown(f"### Confidence: **{confidence:.2%}**")
                    
                    # Show probability distribution
                    st.markdown("#### 📊 Probability Distribution")
                    
                    prob_data = {
                        'Digit': list(range(10)),
                        'Probability': [prob.item() for prob in all_probs]
                    }
                    
                    st.bar_chart(prob_data, x='Digit', y='Probability')
                    
                    # Show processed image
                    st.markdown("#### 🖼️ Processed Image (28×28)")
                    processed_img = image_tensor.squeeze().numpy()
                    st.image(processed_img, width=150, use_container_width=False)
            else:
                st.info("👆 Draw a digit on the canvas to see predictions!")
        else:
            st.info("👆 Draw a digit on the canvas to see predictions!")

if __name__ == "__main__":
    main()
