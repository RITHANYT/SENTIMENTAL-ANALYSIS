# %%
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tkinter
from tkinter import Tk, filedialog
import joblib
import pickle



# %%
df1 = pd.read_csv('C:\\Users\\tnrit\\OneDrive\\my web\\pppp\\Restaurant_Reviews.tsv', delimiter='\t')
df2 = pd.read_csv('C:\\Users\\tnrit\\OneDrive\\my web\\pppp\\scraped_df.csv')
df2 = df2.drop('sentiment', axis=1)
df2 = df2.drop('Unnamed: 0', axis=1)
df2 = df2.rename(columns={'Reviews': 'Review'})
df2 = df2.rename(columns={'hugging_face_label': 'Liked'})
df = pd.concat([df1, df2], ignore_index=True)
#display(df)

# %%
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# %%
df['Liked'] = df['Liked'].astype(int)  # Convert 'Liked' column to integer type
df['Review'] = df['Review'].str.lower()
df['Review'] = df['Review'].str.replace('[^\w\s]', '')
df.head(5)

# %%
df['Review'].fillna('', inplace=True)
df['Review'] = df['Review'].astype(str)
df['Review'] = df['Review'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
df['Review'] = df['Review'].apply(lambda x: [word for word in x if word not in stop_words])
lemmatizer = WordNetLemmatizer()
df['Review'] = df['Review'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['Review'] = df['Review'].apply(lambda x: ' '.join(x))

# %%
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Review'].astype(str))
y = df['Liked']

# %%
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Liked'], test_size=0.2, random_state=42)

# %%
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

# %%
classifier = SVC()
classifier.fit(X_train_tfidf, y_train)

# %%
X_test_tfidf = tfidf.transform(X_test)
y_pred = classifier.predict(X_test_tfidf)

# %%
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# %%
joblib.dump(classifier, 'trained_model.pkl')

# %%
classifier = joblib.load('trained_model.pkl')

# %%
# Create Tkinter root window
root = Tk()
root.withdraw()

# Open file dialog to choose a file
file_path = filedialog.askopenfilename()

# %%

with open(file_path, 'r') as file:
    reviews = file.readlines()

# %%
X_test = tfidf.transform(reviews)
y_pred = classifier.predict(X_test)

# %%
positive_percentage = (y_pred == 1).mean() * 100
negative_percentage = (y_pred == 0).mean() * 100

print(f"Positive reviews: {positive_percentage:.2f}%")
print(f"Negative reviews: {negative_percentage:.2f}%")

