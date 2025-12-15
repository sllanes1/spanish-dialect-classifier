import os
import glob
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from test_sentences import mx_obvious, es_obvious, neutral_mx, neutral_es

# Words to remove so the model doesn't "cheat" using country/place names
BLOCKED_WORDS = [
    "méxico", "mexico", "mexicano", "mexicana", "mexicanos", "mexicanas",
    "españa", "espana", "español", "espanol", "española", "espanola", "españoles", "espanoles", "españolas", "espanolas",
    "madrid"
]

def remove_blocked_words(text: str) -> str:
    # remove whole-word matches only (case-insensitive)
    # removes "mexico" but not "mexicano" unless you add it explicitly
    for w in BLOCKED_WORDS:
        text = re.sub(rf"\b{re.escape(w)}\b", "", text, flags=re.IGNORECASE)
    # normalize extra spaces
    return re.sub(r"\s+", " ", text).strip()

# now loads clean text in (1 file = 1 example)
def load_sentences(folder, label):
    files = glob.glob(os.path.join(folder, "*.txt"))
    sentences, labels = [], []

    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            text = file.read().strip().replace("\n", " ").lower()
            text = remove_blocked_words(text)

            if len(text.split()) >= 4:
                sentences.append(text)
                labels.append(label)

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
    analyzer="char_wb",
    ngram_range=(4,6),
    min_df=2,
    max_features=30000
)

X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)

# The logistic regression classifier itself
clf = LogisticRegression(max_iter=4000)
clf.fit(X_train_vec, y_train)

# Predicting and showing what happens!
pred = clf.predict(X_test_vec)
prob = clf.predict_proba(X_test_vec)

print("\n============ Logistic Regression Results ============")
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\n", classification_report(y_test, pred))

feature_names = np.array(vectorizer.get_feature_names_out())
coef = clf.coef_[0]
classes = clf.classes_  # ['es','mx'] usually

# coef > 0 corresponds to classes[1] (mx) in sklearn binary LR
top_mx = feature_names[np.argsort(coef)[-20:]][::-1]
top_es = feature_names[np.argsort(coef)[:20]]

print("\nTop MX char-ngrams:", top_mx)
print("\nTop ES char-ngrams:", top_es)

def run_quick_tests():
    test_sets = [
        ("MX_OBVIOUS", mx_obvious, "mx"),
        ("ES_OBVIOUS", es_obvious, "es"),
        ("NEUTRAL_MX", neutral_mx, "mx"),
        ("NEUTRAL_ES", neutral_es, "es"),
    ]

    classes = list(clf.classes_)  # e.g. ['es', 'mx']

    print("\n=== QUICK TEST SUITE ===")
    for name, sentences, gold in test_sets:
        correct = 0
        print(f"\n{name} (expected: {gold.upper()})")
        for s in sentences:
            vec = vectorizer.transform([s.lower()])
            pred = clf.predict(vec)[0]
            probs = clf.predict_proba(vec)[0]
            mx_prob = probs[classes.index("mx")]
            es_prob = probs[classes.index("es")]

            is_correct = (pred == gold)
            correct += int(is_correct)

            print(f"- {s}")
            print(f"  Pred: {pred.upper()} | MX={mx_prob:.3f} ES={es_prob:.3f} | {'🟢' if is_correct else '🔴'}")

        print(f"{name} accuracy: {correct}/{len(sentences)} = {correct/len(sentences):.2f}")

# running tests once before interactive mode
run_quick_tests()

# interactive demo

print("\n=====================================================")
print("    Bienvenidos to the Spanish Dialect Classifier  ")
print("   above we tested our models on sentences but you ")
print("               should try it yourself :)           ")
print("-----------------------------------------------------")
print("This model predicts whether a sentence is:")
print("     Mexican Spanish (mx)")
print("     European Spanish (es)")
print("-----------------------------------------------------")
print("Type a sentence to classify it.")
print("Type 'quit' to exit :) ")
print("=====================================================\n")


def ask_yes_no(prompt: str) -> bool:
    # returns true for yes, false for no and then keeps asking until valid input
    while True:
        ans = input(prompt).strip().lower()
        if ans in {"yes", "y"}:
            return True
        if ans in {"no", "n"}:
            return False
        print("Please type 'yes' or 'no'.")


def has_min_words(text: str, min_words: int = 4) -> bool:
    # checks if text has at least min_words words
    words = [w for w in text.strip().split() if w]
    return len(words) >= min_words


while True:
    user_input = input("Enter a Spanish sentence: ").strip()

    if user_input.lower() in {"quit", "exit"}:
        if ask_yes_no("\nAre you sure? (yes/no): "):
            if ask_yes_no("\nAre you REALLY really sure?? (yes/no): "):
                print("\nare you really...fine. bye. 🙄")
                break
            else:
                print("\nokayyy, staying in the classifier!\n")
        else:
            print("\nokayyy, staying in the classifier!\n")
        continue

    # Enforce minimum length (more than 3 words)
    if not has_min_words(user_input, min_words=4):
        print("Please enter a sentence with at least 4 words (more than 3). Try again!\n")
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