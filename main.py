import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Wczytanie danych
df = pd.read_csv("dataset/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Zamiana etykiet na wartości binarne
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Proste czyszczenie tekstu (małe litery)
df['message'] = df['message'].str.lower()

# Podział na cechy i etykiety
X = df['message']
y = df['label_num']

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Wektoryzacja tekstu na podstawie TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print(f"Train samples: {X_train_tfidf.shape}")
print(f"Test samples: {X_test_tfidf.shape}")