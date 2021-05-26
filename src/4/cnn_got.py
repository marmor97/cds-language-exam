# importing modules and packages
# system tools
import os
import sys
import argparse
sys.path.append(os.path.join(".."))
from contextlib import redirect_stdout

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

import matplotlib as plt
# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Embedding, # Layers
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam # Two optimizers this time - Adam works very efficiently
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer # Working with text CNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

class cnn_classifier():
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args["filename"], usecols=['Sentence', 'Season', 'Episode', 'Name'])
        
    def preprocessing(self):   
        '''
        The preprocessing function performs various transformations to the data
        1. Data is balanced to have an equal amount of label classes
        2. Data is split into x and y 
        3. Data is further split into train and test values
        '''
        
        print("[INFO] Preprocessing data... ")

        if self.args['preprocess']=='collapse_character':
        # Grouping by character    
            self.data = self.data.groupby(['Season','Episode','Name'])['Sentence'].apply('.'.join).reset_index()
        
        elif self.args['preprocess']=='add_character':
            self.data['Sentence'] = self.data['Name']+ ' ' +self.data['Sentence']

       # I'm interested in seeing how many sentences there are from each season
        n_sentences = []
        for val in set(self.data['Season']):
            length = len(self.data['Sentence'].loc[self.data['Season'] == val])
            n_sentences.append(length) 

# I can see there is a different amount of sentences in each season, which might affect the classification. So I am saving all lengths and choose the minimum as n in the balance function to have a distribution that isn't skewed
            
        # Balancing data to not bias classifier
        balanced_data = clf.balance(self.data, label = "Season", n=min(n_sentences))
        
        # Extracting numbers from seasons
        balanced_data['Season']=balanced_data['Season'].str.extract('(\d+)')
        balanced_data['Season']=pd.to_numeric(balanced_data['Season'])

        # Splitting up to x features and y from the balanced data
        x = balanced_data['Sentence'].values
        y = balanced_data['Season'].values
        
        # Splitting into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y, # Labels
                                                            test_size=0.25, 
                                                            random_state=42,
                                                            stratify=y) # This should keep an equal amount of labels in each set - keeps the original distribution, which is equal.
        # Creating dummy variables from y-values
        lb = preprocessing.LabelBinarizer()
        self.y_train = lb.fit_transform(y_train)
        self.y_test = lb.fit_transform(y_test)
        
        # Changing to strings
        X_train = X_train.astype(str)
        X_test = X_test.astype(str)
        
        return X_train, X_test
        
        
    def tokenization(self, X_train, X_test):
        '''
        Tokenization of data
        
            Input: X train and test values,  y train and test values 
        '''
        print("[INFO] Tokenizing data... ")
        
        # initialize tokenizer
        tokenizer = Tokenizer(num_words=5000) # Create it
        # fit to training data
        tokenizer.fit_on_texts(X_train) # Vocabulary of words fitted with training data

        # tokenized training and test data
        X_train_toks = tokenizer.texts_to_sequences(X_train) # Here they become numbers
        X_test_toks = tokenizer.texts_to_sequences(X_test)
        
        return tokenizer, X_train, X_train_toks, X_test_toks
    
    def pad_embed_model(self, tokenizer, X_train, X_train_toks, X_test_toks):
        '''
        Padding data, creating embedding matrix and vocabulary size, and defining Convolutional Neural Network.
            Input: tokenizer, tokenized X train and X test values
        '''
        print("[INFO] Padding tokens, creating embeddings matrix and defining model... ")
       
        # Max length of doc 
        maxlen = max([len(x) for x in X_train]) #max length of sentences

        # pad training data to maxlen
        self.X_train_pad = pad_sequences(X_train_toks, 
                                    padding='post', # put padding at the end of the sequences. 
                                    # sequences can be padded "pre" or "post"
                                    maxlen= maxlen)
        # pad testing data to maxlen
        self.X_test_pad = pad_sequences(X_test_toks, 
                                   padding='post', 
                                   maxlen= maxlen)
        
        
        embedding_matrix = clf.create_embedding_matrix(os.path.join("..","..","data","4",f"glove.6B.{self.args['embed_dim']}d.txt"), # Comes from  utils 
                                           tokenizer.word_index, 
                                           self.args['embed_dim'])

        
        
        # overall vocabulary size
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        l2 = L2(0.001)

        # Intialize sequential model
        self.model = Sequential()

        
        # Add embedding layer that converts the numerical representations of the sentences into a dense, embedded representation
        self.model.add(Embedding(input_dim = vocab_size,
                        output_dim = self.args['embed_dim'],
                        input_length = maxlen))    

        # Add convolutional layer
        self.model.add(Conv1D(256, 1,
                     activation = 'relu'
                        , kernel_regularizer = l2
                        )) # L2 regularization 

        # Global max pooling
        self.model.add(GlobalMaxPool1D())

        # Add dense layer
        self.model.add(Dense(128, activation = 'relu'
                        ,kernel_regularizer = l2
                       ))

        self.model.add(Dropout(0.2))

        # Add dense layer with 8 nodes; one for each season 
        self.model.add(Dense(8,
                    activation = 'softmax')) # we use softmax because it is a categorical classification problem

        # Compile model
        self.model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 0.00001), 
                  metrics = ['accuracy'])

        # Saving model summary in path defined from commandline (default is out)
        with open(os.path.join(self.args['outpath'], f'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
        
        # Fitting model
        self.history = self.model.fit(self.X_train_pad, self.y_train,
                    epochs=self.args["epochs"],
                    verbose=False,
                    validation_data=(self.X_test_pad, self.y_test),
                    batch_size=10
                   )
        
    def evaluation(self):
        '''
        Evaluation of model - plots learning curves and save accuracies
        '''
        print("[INFO] Evaluating model...")
        # Evaluate 
        # Print performance for each set
        loss_train, accuracy_train = self.model.evaluate(self.X_train_pad, self.y_train)
        loss_test, accuracy_test = self.model.evaluate(self.X_test_pad, self.y_test)
        print(pd.DataFrame({'accuracy_train': [accuracy_train], 'accuracy_test': [accuracy_test]})) # accuracies
        
        # Classification report
        predictions = self.model.predict(self.X_test_pad)
        report = pd.DataFrame(metrics.classification_report(self.y_test.argmax(axis=1), predictions.argmax(axis=1), output_dict = True))
        report.to_csv(os.path.join(self.args['outpath'], "cnn_classification_report.csv"))
        print(f"[INFO] report is saved in {self.args['outpath']} as cnn_classification_report.csv")
        
        # Plot of learning curves
        savepath = os.path.join(self.args['outpath'], "cnn_learning_curve.png")
        clf.plot_history(self.history, # model history
                         epochs = self.args["epochs"], # n epochs
                         savepath = savepath) # path to save
        print(f"[INFO] Plot of learning curve is saved in {self.args['outpath']} as cnn_learning_curve.png")

def main():
    
    # Argparse
    ap = argparse.ArgumentParser(description="[INFO] Argumets CNN on Game of Trones data")

    
    ap.add_argument("-p", 
                "--preprocess", 
                required=False, 
                choices = ["add_character", "collapse_character"],
                type=str, 
                default= "add_character", 
                help="preprocess method of sentences: add_character adds name of character in every sentence, collapse_character groups by character and episode and combine all sentences in one row per character")    
    
    ap.add_argument("-f", 
                "--filename", 
                required=False, 
                type=str, 
                default= os.path.join("..","..","data","4","Game_of_Thrones_Script.csv"), 
                help="str, file name and location") 
    
    ap.add_argument("-e", 
                    "--epochs", 
                required=False, 
                type=int, 
                default= 60, 
                help="int, number of epochs")   
    
        
    ap.add_argument("-o", 
                "--outpath", 
                required=False, 
                type=str, 
                default= os.path.join("..","out"), 
                help="str, output location")

    ap.add_argument("-ed", 
                "--embed_dim", 
                required=False, 
                type=int,
                choices = [50, 100, 200, 300],
                default=50, 
                help="int, glove embeddings size") 
    
    args = vars(ap.parse_args())

    
    # Define class 
    cnn_classifier_got = cnn_classifier(args)
    
    X_train, X_test = cnn_classifier_got.preprocessing()
    
    tokenizer, X_train, X_train_toks, X_test_toks = cnn_classifier_got.tokenization(X_train, 
                                    X_test)
    
    cnn_classifier_got.pad_embed_model(tokenizer, X_train, X_train_toks, X_test_toks)
    
    cnn_classifier_got.evaluation()
    
if __name__=="__main__":
    main()

