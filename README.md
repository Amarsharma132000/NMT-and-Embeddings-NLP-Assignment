Deep Learning for NLP: Sanskrit-English Representation & Translation
Author: Amar Sharma | Environment: PyTorch, Hugging Face, Google Colab (T4 GPU)

This repository contains two advanced Deep Learning pipelines developed to solve complex Natural Language Processing (NLP) tasks for Sanskrit, a low-resource and morphologically highly complex language.

Rather than relying on basic academic architectures (like vanilla RNNs or LSTMs), these projects demonstrate a production-grade approach to Deep Learning: leveraging State-of-the-Art (SOTA) foundation models, optimizing for strict hardware constraints (15GB VRAM), and engineering inference pipelines for maximum speed and reproducibility.

📌 Project 1: Cross-Lingual Semantic Sentence Embeddings
Objective: Generate semantically aligned, highly compressed (128D) sentence embeddings for Sanskrit-English parallel text to maximize cosine similarity.

Approach & SOTA Architecture
Foundation Model: Utilized paraphrase-multilingual-mpnet-base-v2, a SOTA multilingual sentence transformer, establishing a strong zero-shot cross-lingual baseline.

Siamese Fine-Tuning: Fine-tuned the architecture using Multiple Negatives Ranking Loss (MNRL). This contrastive learning approach pulls parallel Sanskrit-English pairs closer in the vector space while pushing apart non-matching pairs within the batch.

Production & Optimization Engineering
Mathematical Dimensionality Reduction: The deployment constraint required low-dimensional embeddings. Instead of retraining a smaller model, I applied a fitted PCA projection to compress the embeddings from 768 to 128 dimensions, retaining maximum explained variance while drastically cutting memory footprint.

Deterministic Inference: Engineered the pipeline to save the exact PCA transformation matrix (pca_transform.joblib) alongside the model weights. This guarantees that production inference dynamically maps unseen data into the exact same 128-dimensional latent space used during training.

Vector Normalization: Applied strict L2-normalization to all output vectors, allowing downstream applications to use computationally cheap dot-products instead of full cosine similarity calculations.

📌 Project 2: Neural Machine Translation (Sanskrit to English)
Objective: Build a highly accurate Seq2Seq NMT system capable of parsing complex Sanskrit Sandhi (compound words) and translating them into fluent English.

Approach & SOTA Architecture
Foundation Model: Leveraged Meta's No Language Left Behind (NLLB-200-distilled-600M). NLLB was specifically chosen over standard T5/BART because its pre-training corpus natively supports Sanskrit, preventing catastrophic forgetting of the source syntax.

Parameter-Efficient Fine-Tuning (PEFT): Fine-tuning a 600M parameter model on a single Colab T4 is traditionally impossible due to Out-Of-Memory (OOM) errors. I bypassed this by implementing Low-Rank Adaptation (LoRA, r=32), freezing the base model and training less than 1% of the total parameters.

Production & Optimization Engineering
Zero-Latency Inference: During the deployment/inference phase, passing tensors through separate LoRA adapters introduces latency. I utilized peft_model.merge_and_unload() to physically fuse the trained LoRA weights back into the base model's standard matrices. This resulted in a Trainable Parameters: 0 state during evaluation, achieving blazing-fast inference speeds of ~0.12 seconds/sentence.

VRAM Management: Enabled gradient checkpointing during the training loop. This trades a minor increase in computation time for a massive reduction in memory footprint, allowing large-batch training on edge/free-tier hardware.

Robust Data Augmentation: Implemented custom source-side noise injection (stochastic word dropout and adjacent word swapping). This acts as aggressive regularization, forcing the model to learn broader semantic context rather than memorizing small, low-resource training datasets.

Metric Maturity: Evaluated the model using both rigid n-gram matching (BLEU) and LLM-based semantic overlap (BERTScore). This dual-metric approach successfully quantified the model's performance even when exposed to severe "Domain Shift" (e.g., testing on the highly poetic Bhagavad Gita after training on technical software tutorials).
