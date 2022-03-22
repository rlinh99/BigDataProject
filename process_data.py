from nltk import SnowballStemmer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import data_loader
from nltk.corpus import stopwords
import textblob as TextBlob
import nltk
from bs4 import BeautifulSoup
import re

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def convert_comment(entries):
    converted = []
    for entry in entries:
        result = TextBlob.TextBlob(entry)
        converted.append(result.sentiment.polarity)
    return converted

def review_to_words(raw_review):
    stemmer = SnowballStemmer('english')
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = review_text.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stopwords]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(stemming_words))


def get_processed_data():
    train_features = ['drugName', 'condition', 'reviewComment', 'usefulCount', 'sideEffects']
    train = data_loader.train_data
    train = train.dropna(how='any', axis=0)
    test = data_loader.test_data
    X_train_raw, y_train_raw = train[train_features], train['rating']
    X_test_raw, y_test_raw = test[train_features], test['rating']
    print("Data Loaded.")

    # clean review
    # remove stemmer
    reviews = X_train_raw['reviewComment']
    reviews_t = X_test_raw['reviewComment']

    X_train_raw['clean_comment'] = reviews.apply(review_to_words)
    X_test_raw['clean_comment'] = reviews_t.apply(review_to_words)

    clean_reviews = X_train_raw['clean_comment']
    clean_reviews_t = X_test_raw['clean_comment']

    # clean word count
    X_train_raw['word_count'] = clean_reviews.apply(lambda x: len(str(x).split()))
    X_test_raw['word_count'] = clean_reviews_t.apply(lambda x: len(str(x).split()))

    # handle comment value
    X_train_raw['comment_val'] = convert_comment(reviews)
    X_test_raw['comment_val'] = convert_comment(reviews_t)

    X_train_raw['clean_comment_val'] = convert_comment(clean_reviews)
    X_test_raw['clean_comment_val'] = convert_comment(clean_reviews_t)

    # categorical data
    le = LabelEncoder()
    le.fit(X_train_raw['sideEffects'])

    X_train_raw['sideEffects_val'] = le.transform(X_train_raw['sideEffects'])
    X_test_raw['sideEffects_val'] = le.transform(X_test_raw['sideEffects'])

    # le2.fit(X_train_raw['drugName'])
    # X_train_raw['drugName_val'] = le2.transform(X_train_raw['drugName'])
    # X_test_raw['drugName_val'] = le2.transform(X_test_raw['drugName'])
    eva_labels = ['sideEffects_val', 'comment_val', 'clean_comment_val', 'word_count']
    X_train = X_train_raw[eva_labels].to_numpy()
    X_train = preprocessing.scale(X_train)

    X_test = X_test_raw[eva_labels].to_numpy()
    X_test = preprocessing.scale(X_test)

    y_train = y_train_raw.to_numpy()
    y_test = y_test_raw.to_numpy()

    print("Data Handling Done.")
    return X_train, y_train, X_test, y_test, len(eva_labels)
