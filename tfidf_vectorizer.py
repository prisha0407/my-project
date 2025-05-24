import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_excel("Customs_Document_Annotations_All.xlsx")

df['text'] = df.fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1)

le = LabelEncoder()
df['label'] = le.fit_transform(df['Document Type'])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))


joblib.dump(model, "document_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


joblib.dump(le, "label_encoder.pkl")

print("✅ Classifier and TF-IDF vectorizer saved!")
