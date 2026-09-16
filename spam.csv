import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred) * 100

print("Model Accuracy:", round(accuracy, 2), "%")


# Function to check a new message
def check_spam(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    print("\nMessage:", message)
    print("Prediction:", prediction)


# Example messages
check_spam("Congratulations, you won 100000 dollars")
check_spam("Hey, are we meeting today?")
