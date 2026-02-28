# A1: The goal is to identify and categorize sexism (Direct, Judgemental, Reported) in tweets (severe class imbalance and linguistic distinctions) using the EXIST 2023 dataset.

Designed and implemented two custom recurrent models:

Baseline Bi-LSTM
Trainable embedding layer with GloVe (100d)
Bidirectional LSTM (128 units)
Dropout regularization
Softmax classification layer

Stacked Bi-LSTM
Two Bi-LSTM layers for deeper semantic modeling
Intermediate ReLU dense layer
Contextual data augmentation (+50% training data using BERT word substitution)

Conducted multi-seed experiments and applied Early Stopping based on Macro-F1 to reduce overfitting.
