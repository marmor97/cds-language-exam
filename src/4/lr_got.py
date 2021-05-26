# importing modules and packages
# system tools
import os
import sys
import argparse
sys.path.append(os.path.join("..", ".."))
from contextlib import redirect_stdout

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# matplotlib
import matplotlib.pyplot as plt

class lr_classifier():
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args["filename"])

        
    def preprocessing(self):
        '''
        The preprocessing function performs various transformations to the data
        1. Data is balanced to have an equal amount of label classes
        2. Data is split into x and y 
        3. Data is further split into train and test values
        4. X features are vectorized
        '''
        
        print("[INFO] Preprocessing Game of Thrones data...")
        # I'm interested in seeing how many sentences there are from each season
        n_sentences = []

        for val in set(self.data['Season']):
            length = len(self.data['Sentence'].loc[self.data['Season'] == val])
            n_sentences.append(length) # I can see there is a different amount of sentences in each season, which might affect the classification. So I am saving all lengths and choose the minimum as n in the balance function to have a distribution that isn't skewed
            # Balancing data to not bias classifier
        balanced_data = clf.balance(self.data, label = "Season", n=min(n_sentences))
        
        # Splitting up to x features and y from the balanced data
        x = balanced_data['Sentence'].values
        y = np.array(balanced_data['Season'].str.extract('(\d+)')).ravel()# Extracting only numbers to have cleaner output - ravel to make it a row-vector instead of column vector
        self.y = [int(numeric_string) for numeric_string in y] # Integer
        
        
        # Splitting into train and test sets
        # I am only attributing  "self"  to y because these are finished being preprocessed. self.X features are defined in vectorization 
        X_train, X_test, self.y_train, self.y_test = train_test_split(x, # Creating two lists - sentences is an array
                                                    self.y, # Labels 
                                                    test_size=0.25, 
                                                    random_state=42,
                                                    stratify=self.y) # This should keep an equal amount of labels in each set - keeps the original distribution, which is equal.
        # .fit_transform(X) = learn feature names + .transform(X)
        # Vectorization
        print("[INFO] Vectorizing text...")
        vectorizer = CountVectorizer()

        # Fitting the vectorizer to our data 
        # Transform to traning featues
        self.X_train_feats = vectorizer.fit_transform(X_train)
        #... then we do it for our test data
        self.X_test_feats = vectorizer.transform(X_test) 
        # Create a list of the feature names. 
        feature_names = vectorizer.get_feature_names()
        
        # Vectorize full dataset
        self.X_vect = vectorizer.fit_transform(x)

        
    def model(self):
        '''
        Function that fit a Logistic Regression to countvectorized X and y features and generates predictions
        '''
        print("[INFO] Defining logistic regression model...")

        # Basic logistic regression
        classifier = LogisticRegression(random_state=42).fit(self.X_train_feats, self.y_train)
        self.y_pred = classifier.predict(self.X_test_feats)
        
        
    def evaluation(self):
        '''
        Evaluation function that saves classification report and learning curve in defined paths. The learning curve is made from a 10-fold cross-validation of the the entired dataset 
        '''
        print("[INFO] Evaluating logistic regression model...")

        # Evaluation
        classifier_metrics = pd.DataFrame(metrics.classification_report(self.y_test, 
                                                                        self.y_pred, 
                                                                        output_dict = True))
        
        print(classifier_metrics)   
        classifier_metrics.to_csv(os.path.join(self.args['outpath'], "lr_classification_report.csv"))
        
        
def main():
    # Argparse
    ap = argparse.ArgumentParser(description="[INFO] LR classifier arguments") 
    
    ap.add_argument("-f", 
                "--filename", 
                required=False, 
                type=str, 
                default= os.path.join("..","..", "data", "4", "Game_of_Thrones_Script.csv"), 
                help="str, file name and location")    
    
    ap.add_argument("-o", 
                "--outpath", 
                required=False, 
                type=str, 
                default= os.path.join("..","..", "out","4"), 
                help="str, output location")   

    
    args = vars(ap.parse_args())

    
    # Define class 
    lr_classifier_got = lr_classifier(args)
    
    lr_classifier_got.preprocessing()
    
    lr_classifier_got.model()
    
    lr_classifier_got.evaluation()
    
if __name__=="__main__":
    main()
    
