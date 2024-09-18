import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import json

# Constants
NUM_CLASSES = 102
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def predict(model, image_tensor, class_to_idx):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class[predicted.item()]
    probability = probability[predicted].item()
    
    return predicted_class, probability

# Load model and class_to_idx
model = load_model("models/resnet50_flowers.pth")
with open("class_to_idx.json", 'r') as f:
    class_to_idx = json.load(f)

st.title("Flower Classification with ResNet50")

st.write("This app uses a ResNet50 model trained on the Oxford Flowers dataset to classify flower images.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image_tensor = process_image(image)
    predicted_class, probability = predict(model, image_tensor, class_to_idx)
    
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Probability: {probability:.4f}")

st.write("## About the Project")
st.write("""
This project demonstrates the use of transfer learning with a ResNet50 model to classify 102 different types of flowers.
The model was trained on the Oxford Flowers dataset and achieved [insert your accuracy] accuracy on the validation set.

Key features of the project:
- Transfer learning using a pre-trained ResNet50 model
- Fine-tuning on the Oxford Flowers dataset
- Data augmentation techniques for improved generalization
- [Add any other key features of your project]

The code for this project is available on [insert your GitHub link if applicable].
""")