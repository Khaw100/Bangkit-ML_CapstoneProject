import random
import re
import string
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import optimizers


df = pd.read_csv('ReviewsEN.csv')
# Global Variables

EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"
MAX_EXAMPLES = len(df)
TRAINING_SPLIT = 0.8

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('ReviewsEN.csv')

def replace_lower_data (data):
    data['sentiment'] = data['sentiment'].replace([1], 2)
    data['sentiment'] = data['sentiment'].replace([0], 1)
    data['sentiment'] = data['sentiment'].replace([-1], 0)
    data['reviews'] = data['reviews'].apply(str.lower)
    return data
def remove_punctuation(text):
    # Remove punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation
def removed_punctuation(data):
    removedPunctuation_text = []
    for i in range(len(df)):
        removedPunctuation_text.append(remove_punctuation(data['reviews'][i]))
    return data
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        lemmatized_words.append(lemma)
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text
def lemmatized_data(data):
    temp =  []
    for i in range(len(df)):
        temp.append(lemmatize_text(df['reviews'][i]))
    data['reviews'] = temp
    return data
def remove_numbers(text):
    cleaned_text = re.sub(r'\d+', '', text)
    return cleaned_text
def removed_numbers(data):
    data['reviews'] = data['reviews'].apply(remove_numbers)
    return data
def random_sampling(data):
    random.seed(42)

    # Get the indices of the DataFrame
    indices = data.index.tolist()

    # Perform random sampling on the indices
    selected_indices = random.sample(indices, MAX_EXAMPLES)

    # Select the corresponding sentences and labels based on the sampled indices
    sentences = data.loc[selected_indices, 'reviews']
    labels = data.loc[selected_indices, 'sentiment']
    return sentences, labels, data
def train_val_split(sentences, labels, training_split):
    ### START CODE HERE

    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = int(len(sentences)*training_split)

    # Split the sentences and labels into train/validation splits
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    test_sentences = sentences[train_size:]
    test_labels = labels[train_size:]

    ### END CODE HERE

    return train_sentences, test_sentences, train_labels, test_labels
def read_stopwords():
    my_file = open("stopwords.txt", "r")

    data = my_file.read()

    stopwords_data = data.split("\n")
    my_file.close()
    stop_words_d = set(stopwords.words('english'))
    stop_words_list = list(stop_words_d)
    stopwords_data = stopwords_data + stop_words_list
    return stopwords_data
def remove_stopwords(sentence, stopwords_data):
    stopwords = stopwords_data + ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    numbers_stopwords = ["1", "2", "3", "4","5","6","7","8","9","10",
                         "one", "two","three","four","five","098"]
    more_words = [ "hw","won't","lpu","weren't","mr","mcq","shes", "shes","india","in","hes","shes","me", "dr", "nlandu", "ko","it","1st", "omr", "ha", "upto","ca", "soo", "cd", "ive","po","cse", "chem", "un","of",
                  "mte", "omr","mte's","ca's","ete's","jnv","ip","sir","its","wks","prob","python","java","lattc","ol","ived","elsewhere", "mother","wouldnt","car",
                  "si", "sat","we","home","hot","god","ice","money's","money","even","about","thats", "wks", "thurs", "months", "sir", "go", "jnv", "ip", "today", "today's", "linux", "github",
                  "lt", "ums", "superb", "at", "cgpa","ques", "brain's", "mcqs", "ve", "say", "pc", "viva", "after", "before", "draw", "asst", "only", "rich", "never", "went", "pcs", "gk", "one's",
                  "co", "duty", "gona", "attendnce","same", "that's", "hahahah", "ad's", "university's", "relly", "build", "cricket", "said", "hall", "profs", "guy's", "can", "along", "archieve", "bag",
                  "part", "master", "push", "or", "add", "were", "virginia","human", "bless", "clean", "count", "onlineopen", "ounce", "brushing", "zero", "mail", "fys", "lowell", "stets", "untill", "until",
                  "prep", "appears", "giulia", "yuk", "memo", "ton", "110q", "unit", "80","re","by","order","fob", "sit", "from","art", "org", "4d", "3d", "cinema", "iii", "cal", "both", "sundays", "todays", "ad",
                  "yoursel","yourself", "kiss", "it'll", "obayani's", "anal", "pgs", "csci", "hw", "more", "able", "lecturer", "lecturer's", "student", "stundet's", "it", "want", "you","he's", "she's"]
    more =  [
    'a', 'about', 'above', 'after', 'again', 'against', "ain't", 'all', 'am', 'an', 'and', 'any', 'are', 'as', "'", ",", ".", "'s",
    'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'by', 'doing', 'down', 'during', 'each', 
    'few','for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll',
    'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll',
    'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most',
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s',
    'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'we', 'we\'d', 'we\'ll', 'we\'re','we\'ve', 'were', 'weren\'t', 
    'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who',
    'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re',
    'you\'ve', 'your', 'yours', 'yourself', 'yourselves']

    final_stopwords = stopwords + numbers_stopwords + more_words + more

    words = sentence.split()
    tempWords = []
    for i in words:
        if i not in final_stopwords:
            tempWords.append(i)
            sentence = ' '.join(tempWords)


    return sentence
# vectorizer = TfidfVectorizer(stop_words='english')
# response = vectorizer.fit_transform(df['reviews'])
def removed_stopwords(data, sw_data):
    cKomen = []
    for i in range(len(data)):
        cKomen.append(remove_stopwords(data['reviews'][i], sw_data))
    data['reviews'] = cKomen
    return data
def fit_tokenizer(sentences, OOV_TOKEN):
    tokenizer = Tokenizer(oov_token = OOV_TOKEN)
    tokenizer.fit_on_texts(sentences)
    return tokenizer
def seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_trunc_sequences = pad_sequences(sequences, maxlen= maxlen, padding = padding, truncating = truncating)
    return pad_trunc_sequences

    my_file = open("stopwords.txt", "r")

    data = my_file.read()

    stopwords_data = data.split("\n")
    my_file.close()
    stop_words_d = set(stopwords.words('english'))
    stop_words_list = list(stop_words_d)
    stopwords_data = stopwords_data + stop_words_list
    return stopwords_data

data = replace_lower_data(df)
data = removed_punctuation(data)
data = lemmatized_data(data)
data = removed_numbers(data)
sentences, labels, data = random_sampling(data)
train_sentences, test_sentences, train_labels, test_labels = train_val_split(sentences, labels, TRAINING_SPLIT)
sw_data = read_stopwords()
data = removed_stopwords(data, sw_data)
tokenizer = fit_tokenizer(train_sentences, OOV_TOKEN)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)
train_pad_trunc_seq = seq_pad_and_trunc(train_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)
val_pad_trunc_seq = seq_pad_and_trunc(test_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)
train_labels = np.array(train_labels)
val_labels = np.array(test_labels)



