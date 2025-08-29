from tkinter import *
import pickle
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

wordnetLem = WordNetLemmatizer()
def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()


def remove_punc(text):
    for char in string.punctuation:
        text = text.replace(char, ' ')
    return text


def lem(text):
    tokens = nltk.word_tokenize(text)
    text = ''
    for i in tokens:
        i = wordnetLem.lemmatize(i, pos='v')
        text = text + ' ' + i
    return text


stop = stopwords.words('english')
stop_removes = ['ain', 'aren', "isn't", 'mightn', 'no', 'nor', 'not', 'don', "don't", 'ain', 'aren', "aren't",
                    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                    'haven', "haven't", 'isn', "isn't"'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                    'wouldn', "wouldn't"]



log = pickle.load(open("logistic.pkl", "rb"))
vec = pickle.load(open('vectorizer.pkl', 'rb'))

root = Tk()
root.geometry('400x360')
root.title("Depression detection using ML")
root.resizable(False, False)
root['bg']='#ddffe7'

title = Label(root, text="Depression detection using ML", font=('Times', 20,), bg='#ddffe7', fg= '#167d7f')
title.place(x=35, y=50)

text = Label(root, text="Enter your comment: ", font=('Times', 15,'bold'), bg='#ddffe7', fg= '#167d7f')
text.place(x=20, y=120)

in_put = Entry(root, width=25, bg='#98d7c2', font=('bold', 15))
in_put.place(x=20, y=170)

can = Canvas(root, bg='#167d7f', height=50, width=330)
can.place(x=20, y=200)
result = Label(root, text="", font=('Times', 25,), bg='#167d7f', fg= '#ddffe7')
result.place(x=23, y=207)

def func():
    result.config(text="")
    text = in_put.get()
    text = text_cleaner(remove_punc(lem(text)))
    stop_ = [item for item in stop if item not in stop_removes]
    text = [word for word in text.split() if word not in stop_]
    cleaned = ""
    for i in text:
        cleaned = cleaned + i + " "

    cleaned = vec.transform([cleaned])
    if log.predict(cleaned) == ['suicide']:
        result.config(text= "suicidal", bg='#167d7f')
    else:
        result.config(text="not suicidal", bg='#167d7f')


button = Button(root, text='enter', command=func, font=('Times', 15,'bold'), bg='#ddffe7', fg= '#167d7f',
                activebackground='#167d7f', activeforeground='#ddffe7')
button.place(x = 20, y= 280)
root.mainloop()
