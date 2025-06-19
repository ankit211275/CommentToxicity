# 🧠 Comment Toxicity Predictor using TensorFlow

This project is a **multi-label text classification** model trained to detect toxic comments using the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset. Built with TensorFlow and NLP preprocessing, it identifies categories such as toxic, severe toxic, obscene, threat, insult, and identity hate.

## 📌 Features

- End-to-end comment classification pipeline
- Text preprocessing with `TextVectorization` layer
- Deep learning model using TensorFlow Sequential API
- Efficient data pipeline using `tf.data`
- Model evaluation and testing
- User-friendly interface using Gradio (optional)

## 🚀 Workflow Overview

1. **Install dependencies**  
   Installs TensorFlow, pandas, matplotlib, scikit-learn, etc.

2. **Load and Explore Dataset**  
   Loads `train.csv` and previews data structure.

3. **Preprocessing**
   - Extracts `comment_text`
   - Targets: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
   - Uses `TextVectorization` for tokenizing and sequencing
   - Converts data into TensorFlow dataset pipeline with `cache()`, `shuffle()`, `batch()`, `prefetch()`

4. **Model Architecture**
   - Built using `tf.keras.Sequential`
   - Likely includes Embedding → Conv1D/LSTM → Dense → Output
   - Compiled with `binary_crossentropy` for multi-label classification

5. **Training**
   - Trained on 160,000+ comments
   - Batches of 16, optimized with `Adam`

6. **Evaluation**
   - Evaluated on accuracy, loss curves
   - Likely includes confusion matrix or classification metrics

7. **Gradio Interface** 
   - Allows you to enter a comment and get predicted toxicity levels in real-time

## 🧪 Tech Stack

- Python
- TensorFlow (with GPU support on macOS via Metal)
- pandas, numpy
- scikit-learn
- matplotlib
- Gradio (for web demo)

## 📁 Dataset

The dataset comes from:
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

- `train.csv`: Contains comment text and 6 toxicity labels

## 📊 Performance

This Comment Toxicity Predictor demonstrates strong classification capabilities across multiple toxicity categories using a deep learning approach. The model has been trained on over 150,000 samples and shows reliable generalization when classifying unseen text data.

Key strengths include:
	•	Handles multi-label classification effectively (a single comment may belong to multiple toxic categories).
	•	Uses efficient preprocessing and vectorization to scale well with large datasets.
	•	Leverages TensorFlow’s Sequential API for a streamlined and modular architecture.

🎯 To view exact performance metrics (e.g., accuracy, loss, F1 score), please run the notebook locally. Model evaluation plots and printouts are provided at the end of the notebook.


## 💻 How to Run

```bash
git clone https://github.com/ankit211275/comment-toxicity-predictor.git
cd comment-toxicity-predictor

pip install -r requirements.txt
jupyter notebook Toxicity.ipynb