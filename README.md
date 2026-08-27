# IMDb Sentiment Analysis with Bidirectional LSTM

A natural language processing project that classifies IMDb movie reviews as **positive** or **negative** using a Bidirectional LSTM neural network built with PyTorch.

## Overview

The project demonstrates an end-to-end NLP workflow:

**Movie Reviews → Preprocessing → Tokenization → Vocabulary → Sequence Padding → Embeddings → Bi-LSTM → Sentiment Prediction**

The final model achieved approximately **75% test accuracy** on unseen reviews.

## Model

The classifier uses:

- Word embedding layer
- Bidirectional LSTM
- Fully connected output layer
- Sigmoid activation for binary classification

**Training setup**

- Optimizer: Adam
- Loss: Binary Cross Entropy
- Batch size: 64
- Epochs: 12

## Dataset

IMDb movie reviews for binary sentiment classification.

- Training set: **24,000 reviews**
- Test set: **6,000 reviews**
- Classes: **Positive / Negative**

## Results

**Test Accuracy: ~75%**

The notebook also includes model evaluation and a function for testing the trained model on custom movie reviews.

The result is presented as-is rather than overstating performance; the project focuses on demonstrating the complete NLP and deep-learning workflow.

## Technologies

Python · PyTorch · Hugging Face Datasets · NumPy · Pandas · Scikit-learn · Matplotlib

## Project Structure

```text
imdb-sentiment-analysis-lstm/
├── README.md
├── imdb_sentiment_analysis_lstm.ipynb
├── requirements.txt
├── .gitignore
└── results/
    └── model_results.txt
```

## Run the Project

```bash
pip install -r requirements.txt
```

Then open `imdb_sentiment_analysis_lstm.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run the cells from top to bottom.

## Key Skills Demonstrated

- Natural language processing
- Text preprocessing and tokenization
- Sequence modeling
- Word embeddings
- Bidirectional LSTMs
- Binary classification
- Model evaluation
- PyTorch

## Future Improvements

Possible next steps would be hyperparameter tuning, pretrained embeddings, stronger regularization, and comparison with transformer-based models.
