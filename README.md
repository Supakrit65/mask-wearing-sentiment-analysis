# Mask-Wearing Sentiment Analysis

![GitHub](https://img.shields.io/github/license/Supakrit65/mask-wearing-sentiment-analysis)

## Overview

This repository contains the code and data for the project conducted during my internship at the Social Computing Laboratory, NAIST. The project, titled  **“Comparative Analysis of Mask-Wearing Stance and Emotional Intensity in Thai and USA YouTube Comments during the COVID-19 Pandemic,"** aims to analyze public sentiment and emotional intensity towards mask-wearing. This is achieved using advanced NLP techniques and fine-tuning state-of-the-art models.

## Contents

- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [License](#license)

## Introduction

The COVID-19 pandemic has sparked widespread debate about mask-wearing and the emotions tied to it across different groups and platforms. This study investigates the stance on mask-wearing and the emotional intensity in YouTube comments from Thai and USA viewers of popular news videos. By fine-tuning the GEMMA 2 model and using the EmoLLama-chat-7b model, we analyze the data to uncover insights into public sentiment and emotional responses.

## Directory Structure

```
mask-wearing-sentiment-analysis/
├── LICENSE
├── README.md
├── Supakrit_Cooperative_Education_Report_Github.pdf # Full Project Report
├── gemma_fine_tune
│   ├── gemma_2
│   │   ├── gemma2_train.ipynb  # Notebook for training the GEMMA 2 model
│   │   ├── gemma2_validate.ipynb  # Notebook for validating the GEMMA 2 model
|   |   └── [Fine-tuned GEMMA 2 Model](https://huggingface.co/Supakrit65/gemma-2-9b-mask-wearing-stance)
│   └── smm4h_task_2_dataset
│       ├── alpaca_format
│       │   ├── stance_test.csv  # Test set formatted for Alpaca
│       │   └── stance_train.csv  # Training set formatted for Alpaca
│       └── raw
│           ├── test_smm4h.tsv  # Raw test dataset from SMM4H Task 2
│           └── train_smm4h.tsv  # Raw training dataset from SMM4H Task 2
├── stance_emotion_labelling
|   ├── [Emollama-7b](https://huggingface.co/lzw1008/Emollama-7b)
│   ├── emotion_labelling.py  # Script for labeling emotions in YouTube comments
│   └── mask_stance_labelling.py  # Script for labeling stances in YouTube comments
└── youtube_comments
    ├── code
    │   ├── analysis
    │   │   ├── correlation
    │   │   │   ├── WHO-COVID-19-global-data.csv  # WHO global COVID-19 data for correlation analysis
    │   │   │   ├── correlation_analysis.ipynb  # Notebook for performing correlation analysis
    │   │   │   ├── stance_emotion_th_mask.csv  # Stance and emotion data for Thai comments
    │   │   │   └── stance_emotion_us_mask.csv  # Stance and emotion data for US comments
    │   │   └── main
    │   │       ├── youtube_comment_stance_emotion_analysis.ipynb  # Main analysis notebook for YouTube comment stance and emotion
    │   ├── fetcher
    │   │   ├── fetch_comments_th.ipynb  # Notebook for fetching Thai YouTube comments
    │   │   └── fetch_comments_us.ipynb  # Notebook for fetching US YouTube comments
    │   └── preprocess
    │       ├── process_comments.ipynb  # Notebook for preprocessing YouTube comments
    │       └── thai_to_english_translation_gemini.ipynb  # Notebook for translating Thai comments to English using Gemini
    └── datasets
        ├── raw
        │   ├── th_mask_unlb.csv  # Unlabeled Thai comments dataset
        │   └── us_mask_unlb.csv  # Unlabeled US comments dataset
        └── stance_emotion_version
            ├── stance_emotion_th_mask.csv  # Labeled stance and emotion dataset for Thai comments
            └── stance_emotion_us_mask.csv  # Labeled stance and emotion dataset for US comments

```

## Installation

To clone the repository and set up the environment, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/mask-wearing-sentiment-analysis.git
   cd mask-wearing-sentiment-analysis
   ```

2.	Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate 
    ```

3. Install the required dependencies:
    ```sh
    # Install the primary requirements first
    pip install -r requirements.txt

    # Install unsloth
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip uninstall unsloth -y
    pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/unslothai/unsloth.git
    ```

## License
This project is licensed under the UNLICENSE.