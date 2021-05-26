# system tools
import os
import sys
import argparse
sys.path.append(os.path.join("..", ".."))
import random

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# data munging tools
import pandas as pd
import numpy as np
import utils.classifier_utils_reddit as clf

# Machine learning stuff
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns


class reddit_lr():
    """This is a class for performing a Logistic regression on a Kaggle dataset containing depressed and suicide sub-reddits.
    """
    def __init__(self, args):
        self.args = args
       # Load file
        data = pd.read_csv(self.args["filename"], dtype={"text": "string", "class": "string"})
        data = data.dropna()

        # Balance and split
        data_balanced = clf.balance(data) # This function uses the same as in class and balances the data based on label - I have changed it to use "class" insteat of "label" and it takes 10000 of each class as default
        self.texts = data_balanced["text"] # Assigning to separate objects
        self.labels = data_balanced["class"]


    # Vectorization
    def vectorization(self):
        """Function for vectorizing input text.
        """
        print("[INFO] Vectorizing data...")
        vectorizer = TfidfVectorizer(ngram_range = (1,4), # As our input we are taking n-grams / combinations of tokens which appear together in the data. One word is a uni-gram. Here were are using unigrams,  bi-grams, trigrams and fourgrams (?) as input.
                                     lowercase = True, # Telling the vectorizer that we want to take the input data annd make all lowercase
                                     max_df = 0.95, # Filtering based on how common / infrequent a word is - max_df = maximum number of documents that the word is allowed to appear in. If it appears in more than 95 % of the documents - remove it
                                     min_df = 0.05, # Below 5 % occurences are also removed
                                     max_features = 500) # Keep only top 500 features - most informative features
        # Applying it to our texts
        self.X_vect = vectorizer.fit_transform(self.texts) # Full dataset
        
        return vectorizer

    def classify(self, vectorizer):
        """Function for making classification and getting results
            
            Input: vectorizer: TF-IDF or other type vectorizer 
        """

                # Taking only train sample from entire datasaet and testing values on this
        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(self.X_vect, 
                                                            self.labels, 
                                                            test_size=0.2, 
                                                            random_state=0)
        if self.args["parameters"] == "gridsearch":
            print("[INFO] Performing parameter grid search...")

            # Set seed
            random.seed(2021)
            
            # Initialise the default model, here given the name 'classifier'
            pipe = Pipeline([('classifier' , LogisticRegression())])

            # Set tunable parameters for grid search
            penalties = ['none','l1', 'l2'] # different regularization parameters
            C = [1.0, 0.1, 0.01]     # different regularization 'strengths'
            solvers = ['liblinear', 'sag', 'saga', 'lbfgs']  # different solvers - lbfgs is default, sag and saga are good for large ds and liblinear for small

            # Create parameter grid (a Python dictionary)
            parameters = dict(classifier__penalty = penalties,  # notice how we use the name 'classifier'
                              classifier__C = C,
                              classifier__solver = solvers)
            
            score = "f1_macro"
            print(f"# Tuning hyper-parameters for {score}...")
            print()
            # Initialise Gridsearch with predefined parameters
            grid_classifier = GridSearchCV(pipe, 
                               parameters, 
                               scoring= score, 
                               cv=10,# use 10-fold cross-validation
                               error_score=0.0) # Not stopping although parameters cannot be combined
            # Fit
            grid_classifier.fit(X_train, y_train)
            
            # Print best results on training data
            print("Best parameters set found on training data:")

            # add new lines to separate rows
            print()
            print(grid_classifier.best_params_)

            # Assigning best parameters
            C = grid_classifier.best_params_.get('classifier__C')
            solver = grid_classifier.best_params_.get('classifier__solver')
            penalty = grid_classifier.best_params_.get('classifier__penalty')
            
            self.classifier = LogisticRegression(C = C,
                                    solver = solver, 
                                    penalty = penalty, 
                                    random_state=2021).fit(X_train, y_train) # Random state to reproduce results
            
        elif self.args["parameters"] == "default":
            print("[INFO] Using default classifier...")
            self.classifier = LogisticRegression(random_state=2021).fit(X_train, y_train) # Random state to reproduce results
                        
            
        # Predictions
        print("[INFO] Classification metrics using simple 80/20% split...")
        y_pred = self.classifier.predict(X_test)

        # Metrics
        out_conf = os.path.join(self.args['outpath'], f"{self.args['parameters']}_model_confusion_matrix.png")
        
        clf.plot_cm(y_test, y_pred, normalized = True, outpath = out_conf)
        
        classifier_metrics = metrics.classification_report(y_test, y_pred)
        print(classifier_metrics)

        # Most important features
        print("The most informative features for classifying suicidal reddits and depressive reddits are:")
        print()
        clf.show_features(vectorizer, y_train, self.classifier, n=20)

        print("[INFO] Assessing cross-validated scores of model...")
        self.cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 2021) # Use this to create 10 different samples of train and test set
        self.cv_results = cross_validate(self.classifier, # Cross validate uses the classfier and cross-validation method specified to get accuracies and f-scores
                       self.X_vect, 
                       self.labels,
                       scoring=["accuracy", "f1_macro"], 
                       cv=self.cv,
                       return_estimator=True)
        

    def save_results(self):
        """Function for printing evaluation metrics and saving learning curve plot.
        """
        print("[INFO] Saving results...")

        print(pd.DataFrame(self.cv_results)) # Printing information of each of the folds to the terminal
        summary = f"The mean f1-score in a 10-fold cv is {np.mean(self.cv_results['test_f1_macro']).round(3)} with a std of {np.std(self.cv_results['test_f1_macro']).round(3)}" # Creating a str object with mean accuracy and f-score and sd accuracy and f-score
        print(summary)
        with open(os.path.join(self.args['outpath'], f"{self.args['parameters']}_model_summary_metrics.txt"), "w", encoding = "utf-8") as file: # Save as .txt file
            file.write(summary)
        
        title = "Learning Curve for model discriminating between suicide and depression subreddits (Logistic Regression)"
        

def main():
    ap = argparse.ArgumentParser(description="[INFO] class made to run logistic regression on Kaggle data set with suicide and depression subreddits") 
    
    ap.add_argument("-f", 
                "--filename", 
                required=False, 
                type=str, 
                default= os.path.join("..", "..", "data", "3", "Suicide_Detection.csv"), 
                help="str, filename for dataset name and location")   

    ap.add_argument("-p", 
                "--parameters", 
                required=False, 
                choices = ["gridsearch", "default"],
                type=str, 
                default= "gridsearch", 
                help="str, gridsearch or default parameters")   
    
    ap.add_argument("-o", 
                "--outpath", 
                required=False, 
                type=str, 
                default= os.path.join("..", "..","out","3"), 
                help="str, filename for dataset name and location") 
    
    args = vars(ap.parse_args())
    
    
    # Making a lr_reddit object
    reddit = reddit_lr(args = args)
    vectorizer = reddit.vectorization()
    reddit.classify(vectorizer)
    reddit.save_results()
    
if __name__=="__main__":
    main()        
        