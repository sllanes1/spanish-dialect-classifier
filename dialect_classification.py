import os
import glob
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load clean text in
def load_sentences(folder, label):
    files = glob.glob(os.path.join(folder, "*.txt"))
    sentences = []
    labels = []

    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if len(line.split()) >= 4: # keep lines at least 4 words
                    sentences.append(line.lower()) #lower case
                    labels.append(label)   # label of mx or es
    return sentences, labels

mx_sentences, mx_labels = load_sentences("data/clean/mx", "mx")
es_sentences, es_labels = load_sentences("data/clean/es", "es")

texts = mx_sentences + es_sentences
labels = mx_labels + es_labels

print(f"Loaded {len(texts)} total sentences")

# Train the test split 80/20
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Tf Idf with n-grams (works best!)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),     # we are using unigrams, bigrams, trigrams
    min_df=3,
    max_features=50000 # limits to the 50k most important features
)

X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)

# The logistic regression classifier itself
clf = LogisticRegression(max_iter=4000)
clf.fit(X_train_vec, y_train)

# Predicting and showing what happens!
pred = clf.predict(X_test_vec)
prob = clf.predict_proba(X_test_vec)

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# How it should look based off an example data point
example = "yo quiero coger el autobús"
vec = vectorizer.transform([example])
print("\nExample:", example)
print("Predicted dialect:", clf.predict(vec)[0])
print("Probabilities:", clf.predict_proba(vec))

print("\n===============================================")
print("   Bienvenidos to the Spanish Dialect Classifier  ")
print("---------------------------------------------------")
print("This model predicts whether a sentence is:")
print("    Mexican Spanish (mx)")
print("    European Spanish (es)")
print("---------------------------------------------------")
print("Type a sentence to classify it.")
print("Type 'quit' to exit :) ")
print("===============================================\n")

while True:
    user_input = input("Enter a Spanish sentence: ")

    if user_input.lower().strip() in ["quit", "exit"]:
        confirm1 = input("\nAre you sure? (yes/no): ").lower().strip()
        if confirm1 in ["yes", "y"]:
            confirm2 = input("\nAre you REALLY really sure?? (yes/no): ").lower().strip()
            if confirm2 in ["yes", "y"]:
                print("\nare you really...fine. bye. 🙄")
                break
            else:
                print("\nokayyy, staying in the classifier!\n")
                continue
        else:
            print("\nokayyy, staying in the classifier!\n")
            continue

    vec = vectorizer.transform([user_input])
    pred = clf.predict(vec)[0]
    prob = clf.predict_proba(vec)[0]

    # Make sure we pick the right index for mx/es
    classes = list(clf.classes_)   # e.g. ['es', 'mx']
    mx_prob = prob[classes.index("mx")]
    es_prob = prob[classes.index("es")]

    print(f"\nPredicted dialect: {pred.upper()}")
    print(f"   MX probability: {mx_prob:.3f}")
    print(f"   ES probability: {es_prob:.3f}\n")