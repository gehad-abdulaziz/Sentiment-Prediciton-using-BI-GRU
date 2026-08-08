# IMDB Sentiment Classification with Bidirectional GRU + GloVe

A sentiment classifier for movie reviews, built on the IMDB 50k dataset. The main goal of this project was to actually apply what I had studied about GRU networks and word embeddings, not just read about them, and to compare a few design choices instead of settling for whatever worked first.

Final test accuracy: **88.34%**

## Why this architecture

**Embeddings.** I used pretrained GloVe vectors (300 dimensions) instead of training the embedding layer from scratch. GloVe builds its vectors from global word co-occurrence statistics across the whole corpus, which is a different approach from something like Word2Vec that mostly looks at local context windows. With a vocabulary capped at 20,000 words, GloVe covered 19,902 of them, about 99.5%, so almost nothing was left with a random initialization.

**GRU over LSTM.** GRU has fewer gates and fewer parameters than LSTM, so it trains faster while still handling sequential dependencies reasonably well. For a binary sentiment task on review-length text, that tradeoff made sense, the extra modeling capacity of LSTM wasn't worth the added training cost here.

**Bidirectional.** A single-direction GRU only sees a word's past context. Wrapping it as bidirectional lets the model read the sentence forward and backward and combine both, which matters for sentiment since the meaning of a phrase can depend on what comes after it, not just what came before (e.g. "wasn't as bad as I expected").

**Two stacked BiGRU layers**, 128 units then 64 units. First layer returns full sequences so the second layer can process the representation further, then the second layer condenses that into a single vector for classification.

## Trainable vs frozen embeddings

This was the main experiment in the project.

I first let the GloVe embeddings fine-tune during training (`trainable=True`). Training accuracy climbed to around 97.5%, but validation accuracy plateaued near 89% and the gap kept widening, i.e. classic overfitting. The model was memorizing training-specific patterns instead of generalizing.

I then froze the embedding layer (`trainable=False`), so only the GRU and dense layers were learning. Validation accuracy dropped slightly, but the train/val gap shrank a lot and training was more stable. I went with the frozen version for the final model since the small accuracy tradeoff was worth the better generalization.

## Pipeline

```
Raw reviews
    -> lowercase, strip HTML tags, remove non-alphanumeric chars (keep apostrophes)
    -> train/test split (80/20, stratified) BEFORE fitting the tokenizer, to avoid leakage
    -> Keras Tokenizer, vocab size 20,000, OOV token
    -> pad/truncate to 200 tokens
    -> GloVe 300d embedding matrix (frozen)
    -> SpatialDropout1D(0.2)
    -> Bidirectional GRU(128, return_sequences=True, dropout=0.2)
    -> Bidirectional GRU(64, dropout=0.2)
    -> Dense(64, relu)
    -> Dropout(0.3)
    -> Dense(1, sigmoid)
```

Trained with Adam, binary crossentropy, batch size 64, up to 15 epochs with early stopping on `val_loss` (patience 3) and a learning rate reduction on plateau. It stopped after 9 epochs.

## Results

```
Test Accuracy: 88.34%

              precision    recall  f1-score   support
    Negative       0.90      0.87      0.88      5000
    Positive       0.87      0.90      0.89      5000
    accuracy                           0.88     10000
```

Confusion matrix:

<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/42de042d-09f8-4000-8d54-f8f44ec734b6" />

Training and validation curves:

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/de284942-c936-4fd9-96b7-b5f8f717909e" />

The loss curves show training loss dropping steadily below validation loss after a few epochs, which is expected with the frozen embeddings still leaving some capacity for the GRU layers to overfit. Early stopping caught this before it went much further.

## Manual test cases

I ran a few reviews through the trained model by hand after evaluation, mostly to see how it handled clear cases versus trickier phrasing.

| Review | Prediction | P(positive) |
|---|---|---|
| "This movie was absolutely amazing. The story was interesting, the acting was excellent, and I really enjoyed every minute of it." | Positive | 0.9950 |
| "i don't like or hate this movie" | Negative | 0.3550 |
| "i don't like this movie" | Negative | 0.3400 |
| "i don't like this movie ! it's amazing" | Positive | 0.9780 |

The last two are the interesting ones. "i don't like this movie" is picked up correctly as negative, but the model's confidence isn't very high (0.34), meaning it's not fully separating negation from the surrounding words with much certainty. Once "amazing" is added at the end despite the negative opener, the model flips hard to positive (0.978), which suggests it's leaning more on strong sentiment-carrying words than on parsing the negation structure of the sentence. That's a reasonable limitation for a word-embedding-based sequence model without attention, and it's a good pointer for future work.


## Project structure

```
.
├── sentiment-bigru.ipynb     # full notebook: preprocessing, training, evaluation
└── README.md
```

## Dataset

[IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and [GloVe 6B 300d](https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt) embeddings, both from Kaggle.
