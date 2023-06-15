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
# from tensorflow import keras
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import optimizers

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('ReviewsEN.csv')

# Replace values in pandas DataFrame.
df['sentiment'] = df['sentiment'].replace([1], 2)
df['sentiment'] = df['sentiment'].replace([0], 1)
df['sentiment'] = df['sentiment'].replace([-1], 0)

"""## Case Folding"""

# Apply lower function
df['reviews'] = df['reviews'].apply(str.lower)


"""## Hyper Parameter"""

# Global Variables

EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"
MAX_EXAMPLES = len(df)
TRAINING_SPLIT = 0.8

"""## Remove Punctuation"""


def remove_punctuation(text):
    # Remove punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation

removedPunctuation_text = []
for i in range(len(df)):
  removedPunctuation_text.append(remove_punctuation(df['reviews'][i]))

df['reviews'] = removedPunctuation_text
"""## Lemmatizing"""

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        lemmatized_words.append(lemma)
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

temp =  []
for i in range(len(df)):
  temp.append(lemmatize_text(df['reviews'][i]))
df['reviews'] = temp

"""## Convert Numbers"""

def remove_numbers(text):
    cleaned_text = re.sub(r'\d+', '', text)
    return cleaned_text

df['reviews'] = df['reviews'].apply(remove_numbers)
"""## Random Sampling"""

random.seed(42)

# Get the indices of the DataFrame
indices = df.index.tolist()

# Perform random sampling on the indices
selected_indices = random.sample(indices, MAX_EXAMPLES)

# Select the corresponding sentences and labels based on the sampled indices
sentences = df.loc[selected_indices, 'reviews']
labels = df.loc[selected_indices, 'sentiment']

"""# Training - Validation Split"""

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

train_sentences, test_sentences, train_labels, test_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

"""# Tokenization & Stopwords - Sequences, Truncating, and Padding"""

"""#### Stopwords"""

my_file = open("stopwords.txt", "r")

data = my_file.read()

stopwords_data = data.split("\n")
my_file.close()

def remove_stopwords(sentence, data):
    # List of stopwords

    stopwords = data + ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    numbers_stopwords = ["1", "2", "3", "4","5","6","7","8","9","10",
                         "one", "two","three","four","five","098"]
    more_words = [ "hw","won't","lpu","weren't","mr","mcq","shes", "shes","india","in","hes","shes","me", "dr", "nlandu", "ko","it","1st", "omr", "ha", "upto","ca", "soo", "cd", "ive","po","cse", "chem", "un","of",
                  "mte", "omr","mte's","ca's","ete's","jnv","ip","sir","its","wks","prob","python","java","lattc","ol","ived","elsewhere", "mother","wouldnt","car",
                  "si", "sat","we","home","hot","god","ice","money's","money","even","about","thats", "wks", "thurs", "months", "sir", "go", "jnv", "ip", "today", "today's", "linux", "github","doe"
                  "lt", "ums", "superb", "at", "cgpa","ques", "brain's", "mcqs", "ve", "say", "pc", "viva", "after", "before", "draw", "asst", "only", "rich", "never", "went", "pcs", "gk", "one's",
                  "co", "duty", "gona", "attendnce","same", "that's", "hahahah", "ad's", "university's", "relly", "build", "cricket", "said", "hall", "profs", "guy's", "can", "along", "archieve", "bag",
                  "part", "master", "push", "or", "add", "were", "virginia","human", "bless", "clean", "count", "onlineopen", "ounce", "brushing", "zero", "mail", "fys", "lowell", "stets", "untill", "until",
                  "prep", "appears", "giulia", "yuk", "memo", "ton", "110q", "unit", "80","re","by","order","fob", "sit", "from","art", "org", "4d", "3d", "cinema", "iii", "cal", "both", "sundays", "todays", "ad",
                  "yoursel","yourself", "kiss", "it'll", "obayani's", "anal", "pgs", "csci", "hw", "more", "able", "lecturer", "lecturer's", "student", "stundet's", "it", "want", "you","he's", "she's"]
    more =  [
    'a', 'about', 'above', 'after', 'again', 'against', "ain't", 'all', 'am', 'an', 'and', 'any', 'are', 'as',
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

stop_words = set(stopwords.words('english'))
stop_words_list = list(stop_words)
stopwords_data = stopwords_data + stop_words_list


vectorizer = TfidfVectorizer(stop_words='english')
response = vectorizer.fit_transform(df['reviews'])

cKomen = []
for i in range(len(df)):
  cKomen.append(remove_stopwords(df['reviews'][i], stopwords_data))

df['reviews'] = cKomen

def fit_tokenizer(sentences, oov_token):
    tokenizer = Tokenizer(oov_token = OOV_TOKEN)
    tokenizer.fit_on_texts(sentences)
    return tokenizer

# Test your function
tokenizer = fit_tokenizer(train_sentences, OOV_TOKEN)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)
def seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_trunc_sequences = pad_sequences(sequences, maxlen= maxlen, padding = padding, truncating = truncating)
    return pad_trunc_sequences

train_pad_trunc_seq = seq_pad_and_trunc(train_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)
val_pad_trunc_seq = seq_pad_and_trunc(test_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)

train_labels = np.array(train_labels)
val_labels = np.array(test_labels)