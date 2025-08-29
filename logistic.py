# Depression Detection in text using ML

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

file = open("cleaned.obj",'rb')
df = pickle.load(file)


x = df['text']
y = df['class']

vectorizer = TfidfVectorizer(ngram_range=(3,3))
X = vectorizer.fit_transform(x)
X1 = vectorizer.fit(x)
pickle.dump(X1, open('vectorizer.pkl', 'wb'))

x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print(logreg.score(x_test, y_test, sample_weight=None))

pickle.dump(logreg, open("logistic1.pkl", "wb"))