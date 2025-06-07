# SMS Spam Classification

A machine learning project for classifying SMS messages as spam or ham (not spam) using Naive Bayes and Random Forest models.

---

## Table of Contents

- [SMS Spam Classification](#sms-spam-classification)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Dataset](#dataset)
  - [Data Preparation](#data-preparation)
  - [Models](#models)
    - [Naive Bayes](#naive-bayes)
    - [Random Forest](#random-forest)
  - [Results](#results)
  - [How to Run](#how-to-run)
  - [Repository and Plots](#repository-and-plots)

---

## Project Description

The goal of this project is to develop effective models to classify SMS messages as spam or ham. The implemented models are:  
- Multinomial Naive Bayes  
- Random Forest  

The project demonstrates text data preprocessing, feature extraction, hyperparameter tuning, and model evaluation.

---

## Dataset

The project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains 5572 SMS messages labeled as `ham` (normal) or `spam`.

---

## Data Preparation

- Converted all text to lowercase  
- Removed special characters and numbers  
- Removed extra spaces  
- Tokenized and vectorized text using `TfidfVectorizer`  
- Split data into training (80%) and testing (20%) sets with stratified sampling

---

## Models

### Naive Bayes

- Multinomial Naive Bayes with default parameters (`alpha=1.0`)  
- Vectorization parameters: `min_df=1`, `stop_words='english'`, `ngram_range=(1,1)`

### Random Forest

- Number of trees: 100  
- Maximum tree depth: 20  
- Class weighting: `balanced` to handle imbalanced classes  
- Hyperparameters tuned using GridSearchCV

---

## Results

| Metric    | Naive Bayes | Random Forest |
|-----------|-------------|---------------|
| Accuracy  | 0.9686      | 0.9758        |
| Precision | 1.0000      | 1.0000        |
| Recall    | 0.7651      | 0.8188        |
| F1-score  | 0.8669      | 0.9004        |

Random Forest achieved slightly better performance but is more computationally intensive. Naive Bayes is faster and simpler, making it suitable for real-time systems.

---

## How to Run

1. Clone this repository  
2. Install dependencies (e.g., scikit-learn, pandas, matplotlib)  
3. Run the notebook or scripts to preprocess data, train models, and evaluate results  
4. Plots and evaluation metrics will be saved in the `results/` folder

---

## Repository and Plots

The full code, including data preprocessing, model training, evaluation, and visualization scripts, is available in this repository. All generated plots (confusion matrices, metric comparisons) are also included.

---