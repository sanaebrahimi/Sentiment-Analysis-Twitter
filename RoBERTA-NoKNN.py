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
    '''using a given text and a contraction map, this function handls contractions by expanding them to the complete form
            :param text(string), contraction_mapping(dict)
            :return expanded text(string)
    '''
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
    '''This function cleans the text data from special characters, punctuation, numbers and stop words
        :param text(string)
        :return text(string)
        '''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)", text) if s])
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text

f = open('Evaluation_Report.txt','a+')
filename = 'norob_classifier.joblib.pkl'

def calculate_metric(labels, preds, y_pred, y_true, model):
    '''Calculates evaluation metrics and writes them in the file. Save the model in a pkl file.
    :param true labels and predicted labels, (n, 1) vectors, 
        predicted labels and true labels, (n, 3) matrices, the trained model.
    '''
    f.write("---------------Twitter-Roberta-Base-Sentiment---------------------\n")
    f.write("accuracy: {:.3f}\n".format((labels == preds).mean()))
    f.write("Average precision probability: {:.3f}\n".format(precision_score(labels, preds, average='macro')))
    f.write("precision: {:.3f}\n".format(average_precision_score(y_true, y_pred, average='macro')))
    f.write("recall: {:.3f}\n".format(recall_score(labels, preds, average='macro')))
    f.write("f1: {:.3f}\n".format(f1_score(labels, preds, average='macro')))
    # print("f1 with probability: {:.3f}".format(f1_score(y_true, y_pred, average='macro')))
    f.write("ROC score- One over Rest: {:.3f}\n".format(roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')))
    # if (labels == preds).mean() > 0.58:
        # save the classifier
    _ = joblib.dump(model, filename, compress=9)


def cleaning_dataframe(path):
    '''read data from excel sheet, create and clean a data frame containing tweets and their labels
        :param a path to .xlsx file
        :return dataframe
        '''
    cols = [1, 2, 3, 4]
    df= pd.read_excel(path, usecols = cols)
    df = df.rename({'Unnamed: 4': 'Class'}, axis=1)
    df = df.drop(index = df.loc[df['Class'] == 'irrelevant'].index)
    df = df.drop(index = df.loc[df['Class'] == 'irrevelant'].index)
    df1 = df.drop(index = df.loc[df['Class'] == 2].index)
    df1 = df1.drop(index = df1.loc[df1['Class'] == '2'].index)
    df2 = df1.dropna()
    df3 = df2.drop(columns=['date', 'time'])
    df4 = df3.dropna()
    df4['Anootated tweet'] = df4['Anootated tweet'].apply(lambda x:cleaning_text(x))
    df4['Anootated tweet'] = df4['Anootated tweet'].apply(lambda x:expand_contractions(x))
    df4['Class'] = df4['Class'].astype(int)
    
    return df4



def prediction(model, tokenizer, X, y):
    ''' Predict the labels for a given input using the pre-trained roberta model
    and the tokenizer. Also writes the classification report into a file.
    :param model(roberta), tokenizer, dataframe of test tweets, dataframe of true labels
    '''
    prediction = []
    prediction_prob = []
    labels = {0: -1,
            1: 0,
            2: 1}
  
    for tweet in X['Anootated tweet']:
        encoded_input = tokenizer(tweet, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment = np.argmax(scores)
        label = labels[sentiment]
        prediction.append(label)
        scores = np.zeros((1,3))
        scores[0,sentiment] = 1
        prediction_prob.append(scores)
        
    


    y_true = np.array(y)
    n = len(y)
    y_pred = np.array(prediction)
    class_0 = np.zeros(n)
    class_0[np.where(y_true == 0)] = 1
    class_1 = np.zeros(n)
    class_1[np.where(y_true == 1)] = 1
    class_min1 = np.zeros(n)
    class_min1[np.where(y_true == -1)] = 1
    y_true2 = np.column_stack((class_min1, class_0, class_1))
    y_pred2 = np.array(prediction_prob)


    calculate_metric(y_true, y_pred, y_pred2, y_true2, model)
    f.write(classification_report(y, prediction, digits=3))
    confusion_mat = confusion_matrix(y_true, prediction)
    f.write("Confiusion matrix:  0 true  1        2 \n")
    f.write("      0 predicted   {}      {}       {}\n".format(confusion_mat[0][0],confusion_mat[0][1],confusion_mat[0][2]))
    f.write("      1             {}      {}       {}\n".format(confusion_mat[1][0],confusion_mat[1][1], confusion_mat[1][2]))
    f.write("      2             {}      {}       {}\n".format(confusion_mat[2][0],confusion_mat[2][1], confusion_mat[2][2]))
    right = confusion_mat[0][0]+confusion_mat[1][1]+confusion_mat[2][2]
    wrong = confusion_mat[0][1]+confusion_mat[0][2]+confusion_mat[1][0]+confusion_mat[1][2]+confusion_mat[2][0]+confusion_mat[2][1]
    f.write("      right     |      wrong   \n")
    f.write("       {}       |      {}      \n".format(right, wrong))
    
    f.write("  Classification accuracy :  {}\n".format(right/(right+wrong)))
    f.write(str("---------------------"*20))
    f.close()


if __name__ == '__main__':
    
    
    df2 = cleaning_dataframe(path)
    X = df2.drop('Class', axis=1)
    y = df2['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y = y.astype(int)
    y = y.tolist()
    y = list(filter((2).__ne__, y))
    
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    #model = joblib.load(filename)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    prediction(model,tokenizer,X_test, y_test)

