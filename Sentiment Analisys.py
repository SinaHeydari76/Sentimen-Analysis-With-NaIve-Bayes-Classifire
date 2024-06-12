import os  # Importing the os module for interacting with the operating system
import pandas as pd  # Importing pandas for data manipulation and analysis
import numpy as np  # Importing numpy for numerical operations
import spacy  # Importing spacy for natural language processing
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from spacy_cleaner import Cleaner  # Importing Cleaner from spacy_cleaner for text cleaning
from spacy_cleaner.processing import mutators, removers  # Importing mutators and removers for text processing
from collections import defaultdict  # Importing defaultdict for default dictionary functionality
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, recall_score, accuracy_score  # Importing metrics for evaluation
from sklearn.feature_extraction.text import CountVectorizer  # Importing CountVectorizer for text vectorization
from sklearn.naive_bayes import MultinomialNB  # Importing MultinomialNB for naive bayes classification

class Preprocess:
    
    def __init__(self):

        self.language_model = spacy.load("en_core_web_sm")  # Loading the spacy language model
        
        # Initializing directories for test and train data
        self.directory1 = "test"
        self.directory2 = "train"
        
        # Initializing lists to hold reviews and their corresponding sentiments
        self.reviews = []
        self.sentiments = []
        
        self.cleaned_reviews = []

        self.cleanerpipeline = Cleaner(self.language_model,
                                       removers.remove_email_token,    
                                       removers.remove_stopword_token,
                                       removers.remove_punctuation_token,
                                       removers.remove_number_token,
                                       mutators.mutate_lemma_token)

    def preprocess(self):
        # Processing the test data
        for label1 in ['pos', 'neg']:  # Iterating through 'pos' and 'neg' directories
            self.sentiment = 1 if label1 == 'pos' else 0  # Assigning sentiment value (1 for positive, 0 for negative)
            path1 = os.path.join(self.directory1, label1)  # Constructing the path to the directory
            
            # Iterating through each file in the directory
            for file_name in os.listdir(path1):
                # Opening the file and reading its content
                with open(os.path.join(path1, file_name), 'r', encoding='utf-8') as file:
                    x1 = file.read()
                    x1 = x1.replace("<br />", ' ') \
                            .replace('?', ' ') \
                            .replace('!', ' ') \
                            .replace('.', ' ') \
                            .replace(':', ' ') \
                            .replace(';', ' ') \
                            .replace('\r', '') \
                            .replace('\n', ' ') \

                    self.reviews.append(x1)  # Appending the review text to the reviews list
                    self.sentiments.append(self.sentiment)  # Appending the sentiment value to the sentiments list
        
        # Processing the train data in the same manner as test data
        for label2 in ['pos', 'neg']:  # Iterating through 'pos' and 'neg' directories
            self.sentiment = 1 if label2 == 'pos' else 0  # Assigning sentiment value (1 for positive, 0 for negative)
            path2 = os.path.join(self.directory2, label2)  # Constructing the path to the directory
            
            # Iterating through each file in the directory
            for file_name in os.listdir(path2):
                # Opening the file and reading its content
                with open(os.path.join(path2, file_name), 'r', encoding='utf-8') as file:
                    x2 = file.read()
                    x2 = x2.replace("<br />", ' ') \
                            .replace('?', ' ') \
                            .replace('!', ' ') \
                            .replace('.', ' ') \
                            .replace(':', ' ') \
                            .replace(';', ' ') \
                            .replace('\r', '') \
                            .replace('\n', ' ') \

                    self.reviews.append(x2)  # Appending the review text to the reviews list
                    self.sentiments.append(self.sentiment)  # Appending the sentiment value to the sentiments list

        
        # Cleaning the reviews
        for i in range(0, len(self.reviews)):
            self.reviews[i] = self.reviews[i].lower()  # Converting to lowercase
            cleaned = self.cleanerpipeline.clean([self.reviews[i]])  # Cleaning the review
            self.cleaned_reviews.append(cleaned[0])  # Appending the cleaned review
            print("review " + str(i) + " from " + str(len(self.reviews)) + " cleaned")  # Printing the progress

    def create_dataframe(self, first_row, second_row):
        # Creating a two column pandas DataFrame from the input
        self.dataframe = pd.DataFrame({'review': first_row, 'sentiment': second_row})
        
        return self.dataframe  # Returning the dataframe as output of the method
    
    def dataframe_to_csv(self, dataframe, csv_name):
        # Converting a pandas DataFrame to .csv file in the same directory with no indices for rows
        dataframe.to_csv(csv_name + '.csv', index=False)

class Data:
    
    def __init__(self, csv_file_name):
        self.dataframe = pd.read_csv(csv_file_name + '.csv')  # Reading the CSV file into a DataFrame
        shuffled = self.dataframe.to_numpy()  # Converting the DataFrame to a numpy array
        np.random.shuffle(shuffled)  # Shuffling the data
        self.data_matrix = shuffled  # Storing the shuffled data
    
class NaiveBayes:
    def __init__(self):
        self.priors = {}  # Initializing prior probabilities
        self.likelihoods = {}  # Initializing likelihoods
        self.vocab = set()  # Initializing vocabulary set
        self.class_counts = {}  # Initializing class counts
        self.total_words = 0  # Initializing total word count

    def fit(self, X, y):
        # Calculating prior probabilities
        self.class_counts = defaultdict(int)  # Initializing class counts with default int
        total_docs = len(y)  # Counting the total number of documents
        
        for label in y:
            self.class_counts[label] += 1  # Counting the number of documents per class
        
        self.priors = {label: count / total_docs for label, count in self.class_counts.items()}  # Calculating prior probabilities
        
        # Calculating likelihoods
        word_counts = {label: defaultdict(int) for label in self.class_counts}  # Initializing word counts per class
        total_words = {label: 0 for label in self.class_counts}  # Initializing total word counts per class
        
        for doc, label in zip(X, y):
            words = doc.split()  # Splitting the document into words
            for word in words:
                self.vocab.add(word)  # Adding the word to the vocabulary
                word_counts[label][word] += 1  # Counting the word per class
                total_words[label] += 1  # Counting the total words per class
        
        self.likelihoods = {label: {} for label in self.class_counts}  # Initializing likelihoods
        
        for label in self.class_counts:
            for word in self.vocab:
                # Using Laplace smoothing
                self.likelihoods[label][word] = (word_counts[label][word] + 1) / (total_words[label] + len(self.vocab))
        
        self.total_words = sum(total_words.values())  # Summing the total words across all classes

    def predict(self, X):
        results = []
        for doc in X:
            words = doc.split()  # Splitting the document into words
            class_probs = {}
            
            for label in self.class_counts:
                class_probs[label] = np.log(self.priors[label])  # Initializing the log probability with the prior
                for word in words:
                    if word in self.likelihoods[label]:
                        class_probs[label] += np.log(self.likelihoods[label][word])  # Adding the log likelihood
                    else:
                        class_probs[label] += np.log(1 / (self.total_words + len(self.vocab)))  # Handling unseen words
            
            results.append(max(class_probs, key=class_probs.get))  # Predicting the class with the highest probability
        return results

    def predict_proba(self, X):
            results = []
            for doc in X:
                words = doc.split()
                class_probs = {}
                
                for label in self.class_counts:
                    class_probs[label] = np.log(self.priors[label])
                    for word in words:
                        if word in self.likelihoods[label]:
                            class_probs[label] += np.log(self.likelihoods[label][word])
                        else:
                            class_probs[label] += np.log(1 / (self.total_words + len(self.vocab)))
    
                total_prob = sum(np.exp(class_probs[label]) for label in self.class_counts)
                if total_prob == 0:
                    total_prob = 1e-10  # Handle zero total probability
                probs = {label: np.exp(class_probs[label]) / total_prob for label in self.class_counts}
                results.append(probs)
            return results



def get_user_review(cleanerpipeline):
    user_review = input("Please enter your review: ")
    cleaned_review = user_review.lower()
    cleaned_review = cleanerpipeline.clean([cleaned_review])
    return cleaned_review[0]

def nb():
    evaluate(nb_classifier, X_test, y_test)


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))
    print("Recall Score:", recall_score(y_test, predictions))

    # Calculate probabilities for ROC curve
    probabilities = model.predict_proba(X_test)
    probs = [p[1] for p in probabilities]  # Extract positive class probabilities

    # Handling NaN values
    probs = np.nan_to_num(probs, nan=1e-10)

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def sknb():
    sklearn_predictions = sklearn_nb.predict(X_test_vec)
    accuracy = accuracy_score(y_test, sklearn_predictions)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, sklearn_predictions))
    print("F1 Score:", f1_score(y_test, sklearn_predictions))
    print("Recall Score:", recall_score(y_test, sklearn_predictions))

    # Calculate probabilities for ROC curve
    sklearn_probs = sklearn_nb.predict_proba(X_test_vec)[:, 1]  # Extract positive class probabilities

    fpr, tpr, _ = roc_curve(y_test, sklearn_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Sklearn ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


prepare = Preprocess()
# prepare.preprocess()
# df = prepare.create_dataframe(prepare.cleaned_reviews, prepare.sentiments)
# prepare.dataframe_to_csv(df, 'cleaned_data')

# Loading the preprocessed data from the CSV file
data = pd.read_csv('cleaned_data.csv')
print('THE CSV FILE LOADED')

# Shuffling the data to randomize the order of samples
data = data.sample(frac=1).reset_index(drop=True)

# Splitting the data into features (X) and labels (y)
X = data['review'].values
y = data['sentiment'].values

# Splitting the data into training and test sets (80% training, 20% test)
split_point = int(0.8 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Training the Naive Bayes classifier
nb_classifier = NaiveBayes()
nb_classifier.fit(X_train, y_train)

# Sklearn Naive Bayes for comparison
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
sklearn_nb = MultinomialNB()
sklearn_nb.fit(X_train_vec, y_train)

# Main loop
go = True
while True:
    if go:
        print('PROGRAM INITIALIZED')
        
        # Checking class distribution in the training and test sets
        print("Class distribution in training set:\n----------------------------------------------\n")
        print(pd.Series(y_train).value_counts())
        print('\n')
        print("Class distribution in test set:\n----------------------------------------------\n")
        print(pd.Series(y_test).value_counts())
        
    else:
        break
    
    while True:
        x = input('''
                     0 ==> exit 
                     1 ==> self implemented NaiveBayes classifier performance
                     2 ==> SKlearn NaiveBayes classifier performance
				     3 ==> Get sentiment of a user review\n''')
        
        if x == '0':
            print('PROGRAM CLOSED')
            x = ''
            go = False
            break

        elif x == '1':
            print('''Self Implemented NaiveBayes Classifier Result:\n
                  ----------------------------------------------\n''')    
            nb()
            x = ''

        elif x == '2':
            print('''SKlearn Classifier Result:\n
                  ----------------------------------------------\n''')
            sknb()
            x = ''
        
        elif x == '3':
            user_review = get_user_review(prepare.cleanerpipeline)
            sentiment = nb_classifier.predict([user_review])[0]
            print("Predicted sentiment for the user review:", "Positive" if sentiment == 1 else "Negative")
            x = ''
            
        else:
            print('!!!!! WRONG INPUT !!!!!')
