# A1: The goal is to identify and categorize sexism (Direct, Judgemental, Reported) in tweets (severe class imbalance and linguistic distinctions) using the EXIST 2023 dataset.

Designed and implemented two custom recurrent models:

Baseline Bi-LSTM: Trainable embedding layer with GloVe (100d), Bidirectional LSTM (128 units), Dropout regularization, Softmax classification layer

Stacked Bi-LSTM: Two Bi-LSTM layers for deeper semantic modeling, Intermediate ReLU dense layer, Contextual data augmentation 

Transformer Fine-Tuning (Hugging Face): Fine-tuned a pretrained twitter-roberta-base model, replaced binary head with custom 4-class classifier, used weighted cross-entropy loss to address class imbalance, and contextual embeddings instead of static word vectors

Conducted multi-seed experiments and applied Early Stopping based on Macro-F1 to reduce overfitting

# A2: Evaluated the effectiveness of LLMs for multi-class sexism detection (not-sexist, derogation, animosity, threats, prejudiced), focusing on Prompt-based Learning, in low-resource, zero-shot, and few-shot settings. 

Designed and optimized task-specific prompts for zero-shot and few-shot (k=8) classification, leveraging in-context learning to provide the models with representative semantic examples without additional training

Used Mistral-7B-Instruct-v0.3 and DeepSeek-R1-Distill-Qwen-7B models and implemented a scalable inference pipeline for batch processing of LLM responses, including raw output parsing and validation of reasoning chains (for DeepSeek-R1)

Conducted a comparative analysis between the two models, assessing their robustness and generalization capabilities in complex sentiment analysis

Performed a deep-dive into model failure modes, identifying Mistral's over-reliance on toxic keywords and DeepSeek's struggles with mixed sentiment signals despite its internal reasoning mechanisms
