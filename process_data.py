from nltk import SnowballStemmer
from sklearn import preprocessing
import sklearn.preprocessing as sp
import data_loader
from nltk.corpus import stopwords
import textblob as TextBlob
import nltk
import re

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))


def convert_comment_polarity(entries):
    converted = []
    for entry in entries:
        result = TextBlob.TextBlob(entry)
        converted.append(result.sentiment.polarity)
    return converted


def clean_review(raw_review):
    stemmer = SnowballStemmer('english')
    # Remove Numerical with regex
    letters = re.sub('[^a-zA-Z]', ' ', raw_review)
    # To Lower
    # words = letters.lower().split()
    words = raw_review.lower().split()
    # Remove stop words, push meaning words into a list
    meaningful_words = [w for w in words if not w in stopwords]
    # Extract Stem words, using english standard
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # Join words back to a clean review
    return ' '.join(meaningful_words)
    # return ' '.join(stemming_words)


def get_processed_data():
    train_features = ['drugName', 'condition', 'reviewComment', 'usefulCount', 'sideEffects']
    train = data_loader.train_data
    NA_row = train.isnull().sum()
    train = train.dropna(how='any', axis=0)
    test = data_loader.test_data
    NA_row_test = test.isnull().sum()
    to_fill_data = data_loader.to_fill_data
    # test = test.dropna(how='any', axis=0)

    X_train_raw, y_train_raw = train[train_features], train['rating']
    X_test_raw, y_test_raw = test[train_features], test['rating']
    x_to_fill_raw = to_fill_data[train_features]
    print(NA_row)
    print(NA_row_test)
    print("Data Loaded.")

    reviews = X_train_raw['reviewComment']
    reviews_t = X_test_raw['reviewComment']
    reviews_to_fill = x_to_fill_raw['reviewComment']

    X_train_raw['clean_comment'] = reviews.apply(clean_review)
    X_test_raw['clean_comment'] = reviews_t.apply(clean_review)
    x_to_fill_raw['clean_comment'] = reviews_to_fill.apply(clean_review)

    clean_reviews = X_train_raw['clean_comment']
    clean_reviews_t = X_test_raw['clean_comment']
    clean_reviews_to_fill = x_to_fill_raw['clean_comment']

    # clean word count
    X_train_raw['word_count'] = clean_reviews.apply(lambda x: len(str(x).split()))
    X_test_raw['word_count'] = clean_reviews_t.apply(lambda x: len(str(x).split()))
    x_to_fill_raw['word_count'] = clean_reviews_to_fill.apply(lambda x: len(str(x).split()))

    # handle comment value
    X_train_raw['comment_val'] = convert_comment_polarity(reviews)
    X_test_raw['comment_val'] = convert_comment_polarity(reviews_t)
    x_to_fill_raw['comment_val'] = convert_comment_polarity(reviews_to_fill)

    X_train_raw['clean_comment_val'] = convert_comment_polarity(clean_reviews)
    X_test_raw['clean_comment_val'] = convert_comment_polarity(clean_reviews_t)
    x_to_fill_raw['clean_comment_val'] = convert_comment_polarity(clean_reviews_to_fill)


    X_train_raw['scaled_useful_count'] = preprocessing.scale(X_train_raw['usefulCount'])
    X_test_raw['scaled_useful_count'] = preprocessing.scale(X_test_raw['usefulCount'])
    x_to_fill_raw['scaled_useful_count'] = preprocessing.scale(x_to_fill_raw['usefulCount'])

    labels_to_cat = ['sideEffects']
    for label in labels_to_cat:
        # categorical data
        le = sp.LabelEncoder()
        le.fit(X_train_raw[label])
        to_train = le.transform(X_train_raw[label])
        to_test = le.transform(X_test_raw[label])
        to_fill = le.transform(x_to_fill_raw[label])
        X_train_raw[f'{label}_val'] = to_train
        X_test_raw[f'{label}_val'] = to_test
        x_to_fill_raw[f'{label}_val'] = to_fill

    eva_labels = ['comment_val', 'clean_comment_val', 'sideEffects_val']
    # eva_labels = ['comment_val', 'clean_comment_val','sideEffects_val']

    X_train = X_train_raw[eva_labels].to_numpy()
    X_train = preprocessing.scale(X_train)

    X_test = X_test_raw[eva_labels].to_numpy()
    X_test = preprocessing.scale(X_test)

    x_fill = x_to_fill_raw[eva_labels].to_numpy()
    x_fill = preprocessing.scale(x_fill)


    y_train = y_train_raw.to_numpy()
    y_test = y_test_raw.to_numpy()

    print("Data Handling Done.")
    return X_train, y_train, X_test, y_test, len(eva_labels), x_fill
