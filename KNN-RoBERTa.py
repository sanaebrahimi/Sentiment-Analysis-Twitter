from sklearn.ensemble import BaggingClassifier
from transformers import AutoTokenizer, AutoModel, TFAutoModel
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.decomposition import PCA
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, \
    confusion_matrix, average_precision_score, multilabel_confusion_matrix


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
path = "training-Obama-Romney-tweets.xlsx"
f = open('Eval_Repo.txt', 'a+')
filename = 'first_classifier.joblib.pkl'

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
    text = re.sub('\[.*?@\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords])
    text = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)", text) if s])
    return text

def calculate_metric(labels, preds, y_pred, y_true):
    '''Calculates evaluation metrics '''
    print("accuracy: {:.3f}".format((labels == preds).mean()))
    print("precision without probability: {:.3f}".format(precision_score(labels, preds, average='macro')))
    print("precision with probability: {:.3f}".format(average_precision_score(y_true, y_pred, average='macro')))
    print("recall without probability: {:.3f}".format(recall_score(labels, preds, average='macro')))
    # print("recall with probability: {:.3f}".format(recall_score(y_true, y_pred, average='macro')))
    print("f1 without probability: {:.3f}".format(f1_score(labels, preds, average='macro')))
    # print("f1 with probability: {:.3f}".format(f1_score(y_true, y_pred, average='macro')))
    print("ROC score: {:.3f}".format(roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')))

def cleaning_dataframe(path):
    '''read data from excel sheet, create and clean a data frame containing tweets and their labels
    :param a path to .xlsx file
    :return dataframe
    '''
    cols = [1, 2, 3, 4]
    df= pd.read_excel(path, usecols=cols)
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


skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)





def get_embedding(text, model):
    '''This function creates a vector that represents each tweet by a vector
    :param A list containing a single tweet: text(list), pretrained roberta model
    :return feature vector: numpy array
    '''
    encoded_input = tokenizer(text[0], return_tensors='pt')
    features = model(**encoded_input)
    features = features[0].detach().cpu().numpy()
    features_mean = np.mean(features[0], axis=0)
    return features_mean


def Classifier(classifier, model, X_train, y_train, X_test, y_test):
    '''This function trains and finds the best classifier and saves the model into a .pkl file
     :param classifier, model(roberta), dataframe of train tweets, dataframe of train labels,
     dataframe of test tweets, dataframe of test labels
     :return predicted labels
     '''

    accuracy= []
    precision= []
    recall = []
    f1 = []
    t0 = time.time()
    pca = PCA(n_components=60)
    for train, test in skf.split(X_train, y_train):

        embedded_vector = []
        test_vector = []
        for tweet in X_train[train]:
            features_mean = get_embedding(tweet, model)
            embedded_vector.append(features_mean)
        EV1 = pca.fit_transform(embedded_vector)
        # print("Total variance explained:", np.around(np.cumsum(pca.explained_variance_ratio_), decimals=3))
        m = classifier.fit(EV1, y_train[train])
        # picking the best estimator
        best = classifier.best_estimator_
        for tweet_test in X_train[test]:
            encoded_test = get_embedding(tweet_test, model)
            test_vector.append(encoded_test)
        EV2 = pca.fit_transform(test_vector)
        prediction = best.predict(EV2)

        accuracy.append(accuracy_score(prediction, y_train[test]))
        precision.append(precision_score(y_train[test], prediction, average='macro'))
        recall.append(recall_score(y_train[test], prediction, average='macro'))
        f1.append(f1_score(y_train[test], prediction, average='macro'))
    t1 = time.time()

    # final evaluation is the average of running the model on all the folds

    f.write('---' * 10+ 'K Neighbors Classifier' +'---' * 10)
    f.write("The entire process of training and testing took {:.3f} s\n".format(t1-t0))
    # f.write("accuracy: {:.3f}\n".format(np.mean(accuracy)))
    # f.write("precision: {:.3f}\n".format(np.mean(precision)))
    # f.write("recall: {:.3f}\n".format(np.mean(recall)))
    # f.write("f1: {:.3f}\n".format(np.mean(f1)))
    f.write("Best hyper parameters to use: {}\n".format(classifier.best_params_))


    embedded_test = []
    for test in X_test:
        encoded_test = get_embedding(test, model)
        embedded_test.append(encoded_test)
    test_data = pca.fit_transform(embedded_test)
    prediction = best.predict(test_data)


    # save the classifier
    _ = joblib.dump(best, filename, compress=9)
    return prediction


def display(confusion, prediction_knn, original_ytest):
    n = len(original_ytest)
    y_pred = np.array(prediction_knn).reshape((n, 3))
    original_ytest = np.array(original_ytest)
    y_true = np.zeros((n, 3))
    y_true[np.where(original_ytest == 0), 0] = 1
    y_true[np.where(original_ytest == 1), 1] = 1
    y_true[np.where(original_ytest == -1), 2] = 1
    f.write("Confiusion matrix: -1 true  0        1 \n")
    f.write("     -1 predicted   {}      {}       {}\n".format(confusion[0][0], confusion[0][1], confusion[0][2]))
    f.write("      0             {}      {}       {}\n".format(confusion[1][0], confusion[1][1], confusion[1][2]))
    f.write("      1             {}      {}       {}\n".format(confusion[2][0], confusion[2][1],
                                                               confusion[2][2]))
    right_classified = confusion[0][0] + confusion[1][1] + confusion[2][2]
    wrong_classified = confusion[0][1] + confusion[0][2] + confusion[1][0] + confusion[1][2] + confusion[2][0] + \
                       confusion[2][1]

    f.write("      right     |      wrong   \n")
    f.write("       {}       |      {}      \n".format(right_classified, wrong_classified))
    f.write("  Classification accuracy :  {}\n".format(right_classified / (right_classified + wrong_classified)))
    f.write("ROC score- One over Rest: {:.3f}\n".format(
       roc_auc_score( y_true, y_pred, multi_class='ovr', average='macro')))
    f.close()
if __name__ == '__main__':
    df4 = cleaning_dataframe(path)
    for train_index, test_index in skf.split(df4, df4['Class']):
        original_Xtrain, original_Xtest = df4.iloc[train_index], df4.iloc[test_index]
        original_ytrain, original_ytest = df4['Class'].iloc[train_index], df4['Class'].iloc[test_index]

    original_Xtrain = original_Xtrain.drop('Class', axis=1)
    original_Xtest = original_Xtest.drop('Class', axis=1)

    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    original_ytrain = original_ytrain.astype(int)
    original_ytest = original_ytest.astype(int)

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    #to load the trained model from pickle file
    # best_model = joblib.load(filename)
    knears_param = {'base_estimator__n_neighbors': [20, 50, 150],'base_estimator__algorithm' : ['auto'], 'base_estimator__metric': ['cosine'], 'n_estimators':[10,20,50]}
    knn = GridSearchCV(BaggingClassifier(KNeighborsClassifier()), knears_param)
    prediction_knn = Classifier(knn, model,  original_Xtrain , original_ytrain , original_Xtest, original_ytest)
    f.write(classification_report(original_ytest, prediction_knn, digits=3))
    confusion_mat = confusion_matrix(original_ytest, prediction_knn)
    display(confusion_mat, prediction_knn, original_ytest)





# X_test = cleaning_dataframe("final-testData-no-label-Obama-tweets.xlsx")
# X_test = cleaning_dataframe("final-testData-no-label-Romney-tweets.xlsx")
# X_test = X_test.values
# print(X_test)
# X_test = np.array(X_test)
# pca = PCA(n_components=50)
# final = []
# for tweet in X_test:
#     # print(tweet)
#     encoded_test = get_embedding(tweet[0], model)
#     final.append(encoded_test)
# test_data = pca.fit_transform(final)
# smote_prediction = best_model.predict(test_data)
# # print(best_model.score(test_data))
# # f.write(classification_report(original_ytest, prediction_knn, digits=3))
# # _ = confusion_matrix(original_ytest, prediction_knn)
# output = open('Obama.txt', 'w')
# # output = open('Obama.txt', 'w')








