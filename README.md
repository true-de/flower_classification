# Flower Classification Project

## Overview
This project is a deep learning-based flower classification system that identifies five types of flowers: daisy, lavender, lotus, sunflower, and tulip. It includes a training script (`train.py`), a Streamlit web application (`app.py`) for predictions, and various model files and metrics.

## Features
- Train a CNN model using TensorFlow and Keras.
- Interactive web app for uploading images and getting classifications with confidence scores.
- Visualization of training history, confusion matrix, and class confidence.
- Evaluation metrics saved in JSON format.

## Project Structure
- `train.py`: Script to train the model.
- `app.py`: Streamlit app for flower classification.
- `model1.keras`: Trained model file.
- `training_history.json`: Training metrics.
- `model_metrics.json`: Evaluation metrics.
- `training_history.png`, `confusion_matrix.png`, `class_confidence.png`: Visualization images.
- `flower/flower_images/`: Dataset directory with subfolders for each flower class.

## Installation
1. Clone the repository.
2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training the Model
Run:
```bash
python train.py
```

### Running the App
Run:
```bash
streamlit run app.py
```
Upload an image to classify it.

## Requirements
The `requirements.txt` file lists all necessary packages.

## Future Work
Expand to more flower types and integrate with mobile apps.

## License
MIT License.
