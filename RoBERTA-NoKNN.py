from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, \
    confusion_matrix, average_precision_score, multilabel_confusion_matrix
import joblib
from sklearn.datasets import load_digits

path = "training-Obama-Romney-tweets.xlsx"
stopwords = ["able", "about", "across", "after", "all", "almost", "also", "am", "among",
             "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by",
             "can", "cannot", "could", "dear", "did", "do", "does", "either", "else",
             "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he",
             "her", "hers", "him", "his", "how", "however", "if", "in", "into", "is", "it",
             "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most",
             "must", "my", "neither", "no", "nor", "of", "off", "often", "on",
             "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she",
             "should", "since", "so", "some", "than", "that", "the", "their", "them",
             "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us",
             "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who",
             "whom", "why", "will", "with", "would", "yet", "you", "your"]

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # using a given text and a contraction map, this function handls contractions by expanding them to the complete form

    global CONTRACTION_MAP
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def cleaning_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)", text) if s])
    return text
f = open('Evaluation_Report.txt','a+')
filename = 'second_classifier.joblib.pkl'

def calculate_metric(labels, preds, y_pred, y_true, model):
    f.write("---------------Twitter-Roberta-Base-Sentiment---------------------\n")
    f.write("accuracy: {:.3f}\n".format((labels == preds).mean()))
    f.write("Average precision probability: {:.3f}\n".format(precision_score(labels, preds, average='macro')))
    f.write("precision: {:.3f}\n".format(average_precision_score(y_true, y_pred, average='macro')))
    f.write("recall: {:.3f}\n".format(recall_score(labels, preds, average='macro')))
    # print("recall with probability: {:.3f}".format(recall_score(y_true, y_pred, average='macro')))
    f.write("f1: {:.3f}\n".format(f1_score(labels, preds, average='macro')))
    # print("f1 with probability: {:.3f}".format(f1_score(y_true, y_pred, average='macro')))
    f.write("ROC score- One over Rest: {:.3f}\n".format(roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')))
    if (labels == preds).mean() > 0.58:
        # save the classifier
        _ = joblib.dump(model, filename, compress=9)


cols = [1, 2, 3, 4]
df= pd.read_excel(path, usecols=cols)
df = df.rename({'Unnamed: 4': 'Class'}, axis=1)

print(len(df))

df = df.drop(index = df.loc[df['Class'] == 'irrelevant'].index)
df = df.drop(index = df.loc[df['Class'] == 'irrevelant'].index)
df1 = df.drop(index = df.loc[df['Class'] == 2].index)
df1 = df1.drop(index = df1.loc[df1['Class'] == '2'].index)
# print(df1.loc[df1['Class'] == 2])
df2 = df1.dropna()
print('Positive', round(df['Class'].value_counts()[1] / len(df) * 100, 2),
      '% of the dataset')
print('Negative', round(df['Class'].value_counts()[-1] / len(df) * 100, 2),
      '% of the dataset')
print('Neutral', round(df['Class'].value_counts()[0] / len(df) * 100, 2),
      '% of the dataset')
df3 = df2.drop(columns=['date', 'time'])
df4 = df3.dropna()
df4['Anootated tweet'] = df4['Anootated tweet'].apply(lambda x:cleaning_text(x))
df4['Anootated tweet'] = df4['Anootated tweet'].apply(lambda x:expand_contractions(x))
X_train, X_test= train_test_split(df4, test_size=0.2)
# print(X_test.loc[X_test['Class'] == 2])

y = X_test['Class']
y = y.astype(int)
y = y.tolist()
y = list(filter((2).__ne__, y))


X = X_test.drop('Class', axis=1)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
#model = joblib.load(filename)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
labels = []
results = []
results_prob = []

dict = {0:-1,
        1: 0,
        2: 1,
              }
for tweet in X['Anootated tweet']:
    encoded_input = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = np.argmax(scores)
    results.append(dict[sentiment])
    results_prob.append(scores)
    # f.write(str(dict[sentiment]) + '\n')
    # output = open('Obama.txt', 'w')
    # output.write(str(dict[sentiment]) + '\n')

# output.close()

y_true = np.array(y)
y_pred = np.array(results)
class_0 = np.zeros((len(y)))
class_0[np.where(y_true == 0)] = 1
class_1 = np.zeros((len(y)))
class_1[np.where(y_true == 1)] = 1
class_min1 = np.zeros((len(y)))
class_min1[np.where(y_true == -1)] = 1
y_true2 = np.column_stack((class_min1, class_0, class_1))
y_pred2 = np.array(results_prob).reshape((len(y), 3))
calculate_metric(y_true, y_pred, y_pred2, y_true2, model)
f.write(classification_report(y, results, digits=3))
f.write(str("---------------------"*20))
f.close()
# output = open('Obama.txt', 'w')

# output.write(y_pred)
