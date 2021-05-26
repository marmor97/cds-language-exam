'''
Description: This script calculate the sentiment score for every headline in a dataset of over a million headlines taken from the Australian news source ABC. This is done using the spaCyTextBlob approach. It saves a plot of sentiment over time with a 1-week rolling average and a plot of sentiment over time with a 1-month rolling average.
'''

# Packages
# Path operations
import os 

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Importing pandas to make changes in dataframe
import pandas as pd

# NLP tools
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Plotting
import matplotlib.pyplot as plt

# Commandline arguments
import argparse


class headline_polarity:
    
    def __init__(self, args):
        self.args = args # Assigns args to use it throughout the script

        '''
        Load data and change data formats
        '''
        print('[INFO] Loading and prepocessing data...')

        # Defining path and csv
        self.ds = pd.read_csv(args['path'])

        # Sample data
        if self.args['sample'] is not None:
            self.ds = self.ds.sample(self.args['sample'])

        # Change date format to year-month-day
        self.ds["publish_date"]= pd.to_datetime(self.ds.publish_date, format="%Y%m%d")
        



    # Achieving polarity
    def polarity(self):
        '''
        Function that calculates text polarity with SpacyTextBlob
        '''
        print('[INFO] Calculating polarity...')
        # Initialising spaCy
        nlp = spacy.load('en_core_web_sm')
        # Spacy text blob
        spacy_text_blob = SpacyTextBlob()
        nlp.add_pipe(spacy_text_blob)

        # Initialising Vader
        import nltk
        nltk.download('vader_lexicon')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sentiment = SentimentIntensityAnalyzer()

    
        # Empty array
        polarity_scores = []
        polarity_scores_vader = []
        
        # Looping through headline in dataset and appending polarities to the array defined above
        for doc in nlp.pipe(self.ds["headline_text"]):
            polarity_scores.append(doc._.sentiment.polarity)

        # Assigning polarity to the dataset
        self.ds["polarity_TextBlob"] = polarity_scores

        # Vader polarity
        for headline in self.ds["headline_text"]:
            polarity_scores_vader.append(sentiment.polarity_scores(headline).get('compound'))
            
        self.ds["polarity_Vader"] = polarity_scores_vader
        
    # Plot function 
    def plot_polarity(self, save = True):
        '''
        Function that plots text polarity with a weekly and monthly rolling mean
        '''

        # Splitting argument up
        rolling_means = self.args['rolling_means'].split(sep=" ")
        print(f'[INFO] Creating plot of polarity with rolling means of {rolling_means[0]} and {rolling_means[1]}...')

        # Changing the color map to Paired instead of Viridis
        plt.rcParams["image.cmap"] = "Paired"
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)    

        # Grouping dataset by date and taking mean polarity within 7 and 30 days
        pol_1 = self.ds[["publish_date", "polarity_TextBlob"]].groupby("publish_date", as_index = False).mean("polarity_TextBlob").rolling(int(rolling_means[0])).mean()

        pol_2 = self.ds[["publish_date", "polarity_TextBlob"]].groupby("publish_date", as_index = False).mean("polarity_TextBlob").rolling(int(rolling_means[1])).mean()

        pol_3 = self.ds[["publish_date", "polarity_Vader"]].groupby("publish_date", as_index = False).mean("polarity_Vader").rolling(int(rolling_means[0])).mean()

        pol_4 = self.ds[["publish_date", "polarity_Vader"]].groupby("publish_date", as_index = False).mean("polarity_Vader").rolling(int(rolling_means[1])).mean()
        
        # Plotting
        linewidth=0.5
        plt.figure(figsize=(15, 10))
        plt.plot(pol_1,  label = 'Weekly TextBlob', linewidth=linewidth)
        plt.plot(pol_2,  label = 'Monthly TextBlob',linewidth=linewidth)
        plt.plot(pol_3,  label = 'Weekly Vader',linewidth=linewidth)
        plt.plot(pol_4,  label = 'Monthly Vader',linewidth=linewidth)
        
        plt.title('Headline polarity scores')
        plt.xlabel('Date')
        plt.ylabel('Polarity score')
        plt.legend()

        # Save figure
        if save == True:
            plt.savefig(os.path.join(self.args['outpath'], 'polarity_plot.png'))

        plt.show()
    
    
def main():
    
    ap = argparse.ArgumentParser(description = "[INFO] calculating sentiments") # Defining an argument parse
    
    ap.add_argument("--path", "-p",
                   default = os.path.join("..","..", "data", "1","abcnews-date-text.csv"), help = "str, path to data")
    
    ap.add_argument("--sample", "-s",
                   default = 10000, help = "sample of data to perform sentiment analysis on, if nothing is specified, if nothing is specified, the entire data set is used")
    
    ap.add_argument("--rolling_means", "-r",
                   default = "7 30", 
                   help = "str, timeframes to create mean over")
    
    ap.add_argument("--outpath", "-o",
                   default = os.path.join("..","..", "out", "1"),
                   help = "str, output path")
    
    args = vars(ap.parse_args()) # Adding them together

    # Code to execute to calculate polarity and plot it 
    headline_polarity_calculator = headline_polarity(args)
    headline_polarity_calculator.polarity()
    headline_polarity_calculator.plot_polarity()

# Define behaviour when called from command line
if __name__=="__main__":
    main()
