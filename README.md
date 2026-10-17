# 🔷 Shape Recognition — CNN with PyTorch

A computer vision project that uses a Convolutional Neural Network (CNN) to classify 10 types of geometric shapes in real time using a webcam.

The project combines **PyTorch** for deep learning and **OpenCV** for image processing and contour detection.

## ✨ Features

- 🧠 Custom CNN built with PyTorch
- 🔟 Classification of 10 geometric shape classes
- 📷 Real-time webcam recognition
- 🟩 Bounding boxes around detected shapes
- 🏷️ Predicted class labels and confidence scores
- ⚡ GPU acceleration with CUDA when available
- 📊 Model evaluation using accuracy and a classification report

## 🎯 Supported Classes

The model recognizes the following geometric shapes:

- Circle
- Diamond
- Hexagon
- Oval
- Pentagon
- Rectangle
- Square
- Star
- Trapezoid
- Triangle

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network training and inference |
| Torchvision | Image datasets and transformations |
| OpenCV | Webcam capture and contour detection |
| NumPy | Image processing |
| Pillow | Image conversion and preprocessing |
| scikit-learn | Model evaluation and classification report |

## 📁 Project Structure

```text
.
├── models/
│   └── shape_CNN_10type_irregular.pth
├── train/
│   └── dataset.py
│   └── model.py
│   └── train.py
├── predict.py
├── webcam.py
└── README.md
```

The dataset and trained model are stored locally and may need to be downloaded or generated separately.

Link to dataset: https://drive.google.com/file/d/10Pqym9d8kqRp8dlyNali-dsqdUMAiAR0/view?usp=drive_link

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/on1tch/shape_identifier_cnn.git
cd shape_identifier_cnn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchvision opencv-python numpy pillow scikit-learn
```

For CUDA acceleration, install a PyTorch build compatible with your NVIDIA GPU and CUDA environment. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Prepare the dataset

Make sure your dataset is organized by class, following the structure shown above.

The project uses `torchvision.datasets.ImageFolder`, which assigns class indices based on alphabetical directory names.

### 5. Train the model

```bash
python train/train.py
```

The training script trains the CNN, evaluates it on validation and test sets, prints a classification report, and saves the model weights to:

```text
models/shape_CNN_10type_irregular.pth
```

If you already have a trained checkpoint, you can skip training.

### 6. Run webcam recognition

```bash
python webcam.py
```

The application opens the webcam and displays bounding boxes, predicted class names, and confidence scores.

Press **Q** to close the application.

## 🧠 Model

The project uses a custom convolutional neural network implemented in `model.py`.

The model receives an image tensor and produces a score for each of the 10 classes. A softmax function converts these scores into class probabilities for displaying the prediction confidence.

The webcam pipeline consists of:

1. Capture a frame using OpenCV.
2. Convert the frame to grayscale and apply thresholding.
3. Detect external contours.
4. Extract individual candidate shapes.
5. Resize and normalize each crop using the evaluation transform.
6. Run inference with the trained CNN.
7. Draw bounding boxes and predicted labels.

**Note:** The current detection pipeline relies on contour detection and works best with distinct shapes on a contrasting, relatively plain background. It is not a general-purpose object detector and may struggle with overlapping shapes, complex backgrounds, or poor lighting.

## 📈 Evaluation

The training script reports:

- Training loss and accuracy
- Validation loss and accuracy
- Test loss and accuracy
- Precision, recall, and F1-score for each class

Add your actual test accuracy here after training:

```text
Test Accuracy: XX.XX%
```

Replace the placeholder with the measured result from your experiment.

## 🔮 Future Improvements

- [ ] Improve contour detection under different lighting conditions
- [ ] Support overlapping and touching shapes
- [ ] Optimize inference for multiple objects
- [ ] Add FPS measurement
- [ ] Display the top-3 predictions
- [ ] Export the model to ONNX
- [ ] Add a graphical user interface
- [ ] Experiment with more complex CNN architectures

## 📄 License

Choose a license for the project before publishing it for reuse. For example, the MIT License allows broad reuse with attribution.

---

**Built with Python, PyTorch, and OpenCV.**
