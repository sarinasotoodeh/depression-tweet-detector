import csv
import pickle
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


def slang_abbreviation_translator(user_string):
    user_string = user_string.split(' ')
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        file_name = 'slang.txt'
        # File Access mode [Read Mode]
        accessMode = 'r'
        with open(file_name, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return ' '.join(user_string)


stop_removes = ['ain', 'aren', "isn't", 'mightn', 'no', 'nor', 'not', 'don', "don't", 'ain''aren', "aren't", 'couldn',
                "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                "haven't", 'isn', "isn't"'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

if __name__ == '__main__':
    filehandler = open(b"cleaned.obj", "wb")

    df = pd.read_csv('train.csv')  # reading the dataset
    print(df.shape[0])  # number of the rows

    stop = stopwords.words('english')
    stop = [item for item in stop if item not in stop_removes]

    df = df.dropna()  # removal of the empty rows
    df = df.drop_duplicates(subset='text', keep='first')  # removal of the duplicated rows
    df = df.applymap(lambda x: x.lower() if type(x) == str else x)  # turn all the words to lower case

    df['text'] = df['text'].apply(remove_punc)  # remove punctuations

    wordnetLem = WordNetLemmatizer()
    df['text'] = df['text'].apply(lem)  # Lemmatization
    df['text'] = df['text'].apply(text_cleaner)  # Noise Removal
    df['text'] = df['text'].apply(slang_abbreviation_translator)  # slang and abbreviation

    pickle.dump(df, filehandler)
