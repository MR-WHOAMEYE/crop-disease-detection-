# 🌾 Crop Disease Detection System

A deep learning-based web application for detecting diseases in crop leaves using ResNet50 convolutional neural networks. The system supports multiple crops including Banana, Rice, and Sugarcane, providing instant disease predictions with detailed treatment recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Multi-Crop Support**: Detect diseases in Banana, Rice, and Sugarcane plants
- **Auto-Detection**: Automatically identifies the crop type from the uploaded image
- **Real-time Predictions**: Instant disease detection with confidence scores
- **Treatment Recommendations**: Provides detailed treatment suggestions for detected diseases
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface
- **Deep Learning Models**: Powered by ResNet50 pre-trained on ImageNet
- **Image Analysis**: Green area detection and image quality assessment

## 🌱 Supported Crops and Diseases

### Banana
- Cordana (Cordana leaf spot)
- Healthy
- Pestalotiopsis
- Sigatoka (Yellow/Black Sigatoka)

### Rice
- Bacterial Leaf Blight
- Brown Spot
- Healthy
- Leaf Blast
- Leaf Scald
- Narrow Brown Spot

### Sugarcane
- Healthy
- Mosaic
- Red Rot
- Rust
- Yellow

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- Streamlit
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Matplotlib (for training)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MR-WHOAMEYE/crop-disease-detection-.git
   cd crop-disease-detection-
   ```

2. **Install required dependencies**
   ```bash
   pip install tensorflow streamlit opencv-python numpy pillow matplotlib
   ```

3. **Verify model files**
   Ensure the following pre-trained model files are present in the root directory:
   - `banana_resnet50.keras`
   - `rice_resnet50.keras`
   - `Sugarcane_resnet50.keras`

## 💻 Usage

### Running the Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

**How to use:**
1. Open the web application
2. Choose a crop type (or use Auto Detect)
3. Upload a clear image of a crop leaf (JPG, JPEG, or PNG)
4. Click "🔬 Analyze Leaf" to get predictions
5. View the detected disease, confidence score, and treatment recommendations

### Training Models

To train a new model or retrain existing models:

1. **Prepare your dataset**
   - Organize images in the following structure:
     ```
     dataset/
     ├── [CropName]/
     │   ├── train/
     │   │   ├── Disease1/
     │   │   ├── Disease2/
     │   │   └── ...
     │   └── validation/
     │       ├── Disease1/
     │       ├── Disease2/
     │       └── ...
     ```

2. **Configure the training script**
   - Edit `main.py` to set the correct paths and parameters
   - Update `BASE_PATH` to point to your dataset location
   - Modify `DATASET_CONFIG` for the crop you want to train

3. **Run the training script**
   ```bash
   python main.py
   ```

The training process uses a two-phase approach:
- **Phase 1**: Transfer learning with frozen base model (30 epochs)
- **Phase 2**: Fine-tuning with unfrozen layers (up to 100 epochs total)

Training features:
- Data augmentation for better generalization
- Model checkpointing to save the best model
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Per-class performance evaluation

## 📁 Project Structure

```
crop-disease-detection-/
├── app.py                      # Streamlit web application
├── main.py                     # Model training script
├── banana_resnet50.keras       # Pre-trained Banana disease model
├── rice_resnet50.keras         # Pre-trained Rice disease model
├── Sugarcane_resnet50.keras    # Pre-trained Sugarcane disease model
└── README.md                   # Project documentation
```

## 🧠 Model Architecture

The system uses ResNet50 (Residual Network with 50 layers) as the base architecture:

- **Input**: 224x224 RGB images
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Custom Head**: 
  - Global Average Pooling
  - Batch Normalization
  - Dropout layers (0.4 and 0.3)
  - Dense layer (256 units, ReLU activation)
  - Output layer (Softmax activation)
- **Regularization**: L2 regularization and dropout to prevent overfitting

## 🔬 Image Preprocessing

All input images undergo the following preprocessing:
1. Conversion to RGB format (if needed)
2. Resizing to 224x224 pixels
3. Normalization (pixel values scaled to 0-1)
4. Batch dimension addition for model input

## 📊 Performance

The models are trained with:
- **Batch Size**: 16
- **Initial Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy
- **Data Augmentation**: Rotation, shifting, zooming, flipping, brightness adjustment

## 🎯 Key Features of the Web App

- **Auto Crop Detection**: Analyzes the image with all models and selects the most confident prediction
- **Confidence Visualization**: Progress bars showing prediction confidence
- **All Predictions View**: Expandable section showing probabilities for all disease classes
- **Green Area Detection**: Analyzes the percentage of green leaf area in the image
- **Responsive Design**: Clean, modern interface with custom styling
- **Disease Information**: Detailed descriptions and treatment recommendations for each disease

## 🛠️ Customization

### Adding New Crops

To add support for a new crop:

1. Train a new model using `main.py` with your crop dataset
2. Add the model path to `MODEL_PATHS` in `app.py`
3. Add disease classes to `DISEASE_CLASSES` dictionary
4. Add disease information to `DISEASE_INFO` dictionary

### Modifying Disease Information

Edit the `DISEASE_INFO` dictionary in `app.py` to update:
- Disease descriptions
- Treatment recommendations

## 📝 Notes

- For best results, use clear, well-lit images of individual leaves
- Images should show disease symptoms clearly
- Avoid blurry or low-resolution images
- The green area detection helps validate that the image contains plant material

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.



## 🙏 Acknowledgments

- ResNet50 architecture by Microsoft Research
- TensorFlow and Keras teams for the deep learning framework
- Streamlit for the web application framework
- The agricultural research community for disease classification knowledge

## 📧 Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository.

---

**Note**: The pre-trained models included in this repository are trained on specific datasets. For production use, consider training models on your own comprehensive datasets that represent the conditions in your target environment.
