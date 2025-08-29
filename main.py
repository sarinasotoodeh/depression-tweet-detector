# Depression Detection in text using ML

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

file = open('cleaned.obj', 'rb')
df = pickle.load(file)

print(df.shape[0])  # number of the rows
print(df.isna().sum())  # number of the empty rows

x = df['text']
y = df['class']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print(logreg.score(x_test, y_test, sample_weight=None))
print(df.head(20))

text = ['is your name suicide? cause I think about u every night']
text = vectorizer.transform(text)
print(logreg.predict(text))
