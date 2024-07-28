# Mask-Wearing Sentiment Analysis

![GitHub](https://img.shields.io/github/license/Supakrit65/mask-wearing-sentiment-analysis)

## Overview

This repository contains the code and data for the project conducted during my internship at the Social Computing Laboratory, NAIST. The project, titled  **“Comparative Analysis of Mask-Wearing Stance and Emotional Intensity in Thai and USA YouTube Comments during the COVID-19 Pandemic,"** aims to analyze public sentiment and emotional intensity towards mask-wearing. This is achieved using advanced NLP techniques and fine-tuning state-of-the-art models.

## Contents

- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

The COVID-19 pandemic has sparked widespread debate about mask-wearing and the emotions tied to it across different groups and platforms. This study investigates the stance on mask-wearing and the emotional intensity in YouTube comments from Thai and USA viewers of popular news videos. By fine-tuning the GEMMA 2 model and using the EmoLLama-chat-7b model, we analyze the data to uncover insights into public sentiment and emotional responses.

## Directory Structure

```
mask-wearing-sentiment-analysis/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_fine_tuning.ipynb
│   ├── 04_emotion_analysis.ipynb
│   └── 05_results_analysis.ipynb
├── src/
│   ├── data_preparation.py
│   ├── eda.py
│   ├── model_fine_tuning.py
│   ├── emotion_analysis.py
│   ├── results_analysis.py
│   └── utils.py
├── models/
│   ├── GEMMA_2/
│   ├── EmoLLama_chat_7b/
├── youtube_dataset/
│   ├── code/
│   │   ├── fetcher/
│   │   │   ├── fetch_comments_th.ipynb
│   │   │   ├── fetch_comments_us.ipynb
│   │   ├── preprocess/
│   │   │   ├── process_comments.ipynb
│   │   │   ├── thai_to_english_translation_gemini.ipynb
│   └── datasets/
│       ├── th_mask_unlb.csv
│       ├── us_mask_unlb.csv
└── references.bib
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
    pip install -r requirements.txt
    ```

## Usage

## License
This project is licensed under the UNLICENSE.