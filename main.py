import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Wczytanie danych
df = pd.read_csv("dataset/spam.csv", encoding='latin-1')

print(df)