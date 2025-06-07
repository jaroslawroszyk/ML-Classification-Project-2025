import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Wczytanie danych
df = pd.read_csv("dataset/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Zamiana etykiet na wartości binarne
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Proste czyszczenie tekstu (małe litery)
df['message'] = df['message'].str.lower()

# --- EDA: WYKRESY ---
# Countplot
fig = plt.figure()
fig.canvas.manager.set_window_title("Wykres liczby wiadomości")
sns.countplot(data=df, x='label')
plt.title("Liczba wiadomości (ham vs spam)")
plt.xlabel("Typ wiadomości")
plt.ylabel("Liczba")
plt.tight_layout()
plt.show()

# Histogram długości wiadomości
df['length'] = df['message'].apply(len)
fig = plt.figure(figsize=(10, 6))
fig.canvas.manager.set_window_title("Rozkład długości wiadomości")
sns.histplot(data=df, x='length', hue='label', bins=50, kde=True, element='step')
plt.title("Rozkład długości wiadomości")
plt.xlabel("Długość wiadomości (znaki)")
plt.ylabel("Liczba")
plt.tight_layout()
plt.show()

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

# --- MODEL 1: Naive Bayes ---
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

print("Naive Bayes results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_nb):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_nb):.4f}")
print()

# --- MODEL 2: Random Forest ---
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_tfidf)

print("Random Forest results:")
print(f"Best params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_rf):.4f}")

print("\nClassification report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['ham', 'spam']))

# --- EWALUACJA: MACIERZE POMYŁEK ---
# Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Macierz pomyłek - Naive Bayes")
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['ham', 'spam'])
disp_nb.plot(ax=ax)
plt.title("Macierz pomyłek - Naive Bayes")
plt.tight_layout()
plt.show()

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Macierz pomyłek - Random Forest")
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['ham', 'spam'])
disp_rf.plot(ax=ax)
plt.title("Macierz pomyłek - Random Forest")
plt.tight_layout()
plt.show()

# --- PORÓWNANIE MODELI ---
metrics = {
    'Model': ['Naive Bayes', 'Random Forest'],
    'Precision': [precision_score(y_test, y_pred_nb), precision_score(y_test, y_pred_rf)],
    'Recall': [recall_score(y_test, y_pred_nb), recall_score(y_test, y_pred_rf)],
    'F1-score': [f1_score(y_test, y_pred_nb), f1_score(y_test, y_pred_rf)],
}
metrics_df = pd.DataFrame(metrics)
fig = plt.figure(figsize=(8, 6))
fig.canvas.manager.set_window_title("Porównanie metryk modeli")
metrics_df.set_index('Model')[['Precision', 'Recall', 'F1-score']].plot(
    kind='bar', colormap='viridis', ax=plt.gca()
)
plt.title("Porównanie metryk modeli")
plt.ylabel("Wartość")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
