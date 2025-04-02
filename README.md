# Comparative-analysis-of-LLMs-for-NLP-Tasks

Overview

This repository presents the complete implementation of my MSc Data Science dissertation at City, University of London. The project involves a comparative evaluation of three advanced Large Language Models (LLMs) — LLama 7B, Mistral 7B, and GPT-3.5 — across two prominent NLP tasks: Sentiment Analysis and Question Answering (QA). The study also investigates the impact of fine-tuning, embeddings, and prompting strategies on model performance.

Objectives

Evaluate LLaMA 7B, Mistral 7B, and GPT-3.5 on sentiment analysis and QA tasks.

Fine-tune Mistral 7B and LLaMA 7B using domain-specific datasets.

Analyze GPT-3.5 embeddings and their integration into traditional ML pipelines.

Compare the performance using metrics such as Accuracy, F1-score, ROUGE, and EM.


Dataset

1. Sentiment Analysis

Name: XED (Multilingual Emotion Dataset by Helsinki NLP)

Language: English

Size: ~17,500 examples

Labels: Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust, Neutral

Link: https://github.com/Helsinki-NLP/XED

2. Question Answering

Source: Kaggle QA Dataset

Size: 825 QA pairs across articles

Preprocessing: Retained columns - ArticleTitle, Question, Answer

Link: https://www.kaggle.com/datasets/rtatman/questionanswer-dataset



Technologies and Tools

Languages: Python

Libraries: Hugging Face Transformers, PyTorch, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Platforms: Google Colab (A100 GPU), OpenAI API, Hugging Face Model Hub

Models: LLaMA 7B, Mistral 7B, GPT-3.5

Methodology

Preprocessing

Text cleaning and normalization

Emotion label encoding for XED dataset

Removal of irrelevant columns and nulls in QA dataset

Fine-Tuning (Mistral 7B & LLaMA 7B)

Few-shot prompting used to guide sentiment/QA outputs

Limited to 2000 samples due to compute constraints

Fine-tuned models uploaded to Hugging Face for reproducibility

GPT-3.5 (via OpenAI API)

Utilized GPT-3.5-turbo for sentiment and QA via structured prompts

Also used text-embedding-ada-002 for embedding-based classification (Scikit-learn models)

Notebooks and Code

Notebook

Description

Preprocessing.ipynb

Data cleaning and label encoding

EDA_FINAL.ipynb

Visualizations of data distributions

Sentiment_Mistral_finetunning.ipynb

Fine-tuning Mistral 7B on XED

Mistral_finetunning_QnA.ipynb

Fine-tuning Mistral 7B for QA

GPT.ipynb

Prompt-based GPT-3.5 results for both tasks

Llama_Dissertation_code.ipynb

LLaMA 7B sentiment analysis via few-shot prompts

Evaluation Metrics

Sentiment Analysis: Accuracy, F1-score (macro)

QA: Exact Match (EM), ROUGE

Embedding Comparison: Confusion matrices, traditional ML classifiers with/without embeddings

Results Summary (Simulated from Final Report)

Task

Model

Metric

Score

Sentiment Analysis

Mistral 7B (FT)

F1-score

92.3%

Sentiment Analysis

GPT-3.5

F1-score

78.1%

QA

Mistral 7B (FT)

EM

83.6%

QA

GPT-3.5

EM

71.9%

Sentiment (Embed)

GPT-3.5 + SVM

Accuracy

85.2%

Installation

pip install -r requirements.txt

Use the latest versions of:

transformers

openai

pandas

scikit-learn

matplotlib

seaborn



How to Run

Clone the repository

Upload the datasets to data/

Launch each notebook from notebooks/

To reproduce OpenAI results, insert your API key where required



Acknowledgements

Prof. Pranava Madhyastha (Supervisor)

Hugging Face for model hosting

OpenAI for GPT-3.5 access

Helsinki-NLP for the XED Dataset

Kaggle for QA Dataset



Author: Kunj PatelEmail: kunjp1230@gmail.comLinkedIn: kunj-patel-7b1374189



