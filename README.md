# Facial Emotion Recognition System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Advanced facial emotion recognition system using **Histogram of Oriented Gradients (HOG)** features and **Support Vector Machine (SVM)** classifier. This project detects six basic emotions (Anger, Fear, Happy, Sad, Surprise) plus Neutral expressions from facial images.

### Key Features

- ✅ **High Accuracy**: Achieved 89.9% accuracy on combined JAFFE+CK dataset
- 🎯 **HOG Feature Extraction**: Robust gradient-based feature extraction
- 🤖 **SVM Classification**: RBF kernel for non-linear decision boundaries
- 📊 **Data Augmentation**: Improved performance through image augmentation
- 🔍 **Face Detection**: Haar Cascade for reliable face localization
- 📈 **Comprehensive Evaluation**: Confusion matrices and detailed metrics

## Technical Approach

### Pipeline Architecture

1. **Face Detection** → Haar Cascade with padded cropping
2. **Preprocessing** → CLAHE + Histogram Equalization
3. **Feature Extraction** → HOG (9 orientations, 16×16 cells, 2×2 blocks)
4. **Classification** → SVM with RBF kernel (C=12, gamma='scale')

### Why HOG + SVM?

- **Lightweight**: Minimal computational requirements
- **Interpretable**: Clear decision boundaries
- **Effective**: Works well with limited datasets
- **Robust**: Captures edge orientations crucial for facial expressions

## Results

| Dataset | Before Augmentation | After Augmentation |
|---------|-------------------|-------------------|
| JAFFE   | 76.6%            | 78.2%             |
| CK+     | 62.9%            | 65.0%             |
| **Combined** | **87.5%**   | **89.9%**         |

### Performance Highlights

- **Best Recognized**: Happy, Surprise, Anger
- **Challenging**: Neutral vs Sad, Fear vs Surprise
- **Improvement**: +2-3% with data augmentation

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Jasmeen1331/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from facial_emotion_recognition import FacialEmotionRecognizer

# Initialize
fer = FacialEmotionRecognizer()

# Train on your dataset
X_train, y_train = fer.load_dataset('path/to/train')
X_test, y_test = fer.load_dataset('path/to/test')

fer.train(X_train, y_train)

# Evaluate
accuracy = fer.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}%")

# Predict single image
emotion = fer.predict_image('path/to/image.jpg')
print(f"Detected Emotion: {emotion}")
```

## Datasets

### JAFFE (Japanese Female Facial Expression)
- 213 grayscale images (256×256)
- 10 Japanese female subjects
- 7 emotions with controlled lighting

### CK+ (Cohn-Kanade Extended)
- Peak expression frames from 123 subjects
- Diverse ages and backgrounds
- More challenging with varied lighting

## Project Structure

```
Facial-Emotion-Recognition/
├── facial_emotion_recognition.py   # Main implementation
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── LICENSE                         # MIT License
└── .gitignore                     # Git ignore rules
```

## Technical Details

### HOG Parameters
- **Orientations**: 9 bins
- **Pixels per cell**: 16×16
- **Cells per block**: 2×2
- **Block normalization**: L2-Hys

### SVM Hyperparameters
- **Kernel**: RBF (Radial Basis Function)
- **C**: 12 (regularization parameter)
- **Gamma**: 'scale' (kernel coefficient)

### Data Augmentation
- Horizontal flipping
- Rotation (±15°)
- Scaling variations

## Future Enhancements

- [ ] Deep learning comparison (CNN, Vision Transformers)
- [ ] Real-time webcam emotion detection
- [ ] Multi-face detection and tracking
- [ ] Additional emotion categories
- [ ] Cross-dataset validation
- [ ] Model deployment as REST API

## Author

**Jasmeen** (Student ID: 21236862)
- Final Year Computer Science Student
- University of Lancashire

## References

1. Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. IEEE CVPR.
2. Ekman, P. (1992). *An Argument for Basic Emotions*. Cognition & Emotion.
3. Lucey, P., et al. (2010). *The Extended Cohn-Kanade Dataset (CK+)*. IEEE CVPRW.
4. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- JAFFE and CK+ dataset creators
- OpenCV and scikit-learn communities
- University of Lancashire CS Department

---

⭐ **Star this repository if you find it helpful!**
