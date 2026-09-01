# IMDb Sentiment Analysis with Bidirectional LSTM

A PyTorch NLP project that classifies IMDb movie reviews as **positive** or **negative** using a Bidirectional LSTM.

**Test accuracy: 79.6%**

## Results

### Training Loss

The training loss decreased from **0.6501** in epoch 1 to **0.0102** in epoch 12.

![Training loss across 12 epochs](images/training-loss.png)

### Test Performance

| Metric | Negative | Positive |
|---|---:|---:|
| Precision | 0.75 | 0.86 |
| Recall | 0.88 | 0.71 |
| F1-score | 0.81 | 0.78 |

**Accuracy:** 79.63% on 6,000 test reviews.

### Confusion Matrix

![Confusion matrix showing model predictions](images/confusion-matrix.png)

The model correctly classified **4,778 of 6,000** test reviews.

## Approach

1. Lowercase and tokenize reviews
2. Build a vocabulary from the training data
3. Convert reviews to integer sequences
4. Truncate/pad sequences to 300 tokens
5. Learn word representations with an embedding layer
6. Classify reviews with a Bidirectional LSTM

### Model

- Embedding dimension: 100
- LSTM hidden dimension: 128
- Bidirectional LSTM
- Adam optimizer
- Binary Cross Entropy loss
- Batch size: 64
- 12 training epochs

## Dataset

The IMDb dataset contains 25,000 labeled training reviews and 25,000 labeled test reviews. This project uses a shuffled subset of **24,000 training reviews** and **6,000 test reviews**.

## Custom Predictions

The notebook also includes a simple prediction function for new reviews. Example outputs from the notebook:

- Positive review → **0.9948**
- Negative review → **0.0013**

## Tech Stack

`Python` `PyTorch` `NLP` `Scikit-learn` `Hugging Face Datasets` `NumPy` `Matplotlib`

## Files

- `imdb_sentiment_analysis_lstm.ipynb` — complete analysis and model
- `images/` — key evaluation visualizations
- `results/model_results.txt` — final metrics
- `requirements.txt` — Python dependencies

## Run

```bash
pip install -r requirements.txt
```

Open the notebook in Jupyter or Google Colab and run the cells from top to bottom.
