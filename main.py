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


# print(f"Train samples: {X_train_tfidf.shape}")
# print(f"Test samples: {X_test_tfidf.shape}")

# --- MODEL 1: Naive Bayes ---

# 8. Inicjalizacja i trening
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)

# 9. Predykcje i metryki
y_pred_nb = nb_model.predict(X_test_tfidf)

print("Naive Bayes results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_nb):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_nb):.4f}")
print()

# --- MODEL 2: Random Forest ---

# 10. Parametry do GridSearch
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

best_rf = grid_search.best_estimator_

# 11. Predykcje i metryki RF
y_pred_rf = best_rf.predict(X_test_tfidf)

print("Random Forest results:")
print(f"Best params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_rf):.4f}")

# 12. Raport klasyfikacji (dla RF)
print("\nClassification report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['ham', 'spam']))
