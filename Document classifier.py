
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

file_path = "Customs_Document_Annotations_All.xlsx" 
df = pd.read_excel(file_path)

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
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

new_text = ["Exporter: Alpha Exports Pvt. Ltd. Total weight: 350 kg"]
new_vec = vectorizer.transform(new_text)
predicted_label = model.predict(new_vec)
print("Predicted Document Type:", le.inverse_transform(predicted_label)[0])
import joblib

joblib.dump(model, "document_classifier_model.pkl")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

