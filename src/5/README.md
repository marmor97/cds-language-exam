## Final project 
### Assignment description: 
 
This project aims at distinguishing the topics and themes occuring in five religious and philosophical texts using Topic Modelling with Gensim. 

We have downloaded a data set containing 5 religious and philosophical texts taken from the online book archive Project Gutenberg (https://www.gutenberg.org/). The data set can be downloaded here: https://www.kaggle.com/tentotheminus9/religious-and-philosophical-texts. The 5 books are: 
- The King James Bible (filename pg10.txt)
- The Quran (filename pg2800.txt)
- The Book of Mormon (filename pg17.txt)
- The Gospel of Buddha (filename 35895-0.txt)
- Meditations, by Marcus Aurelius (filename pg2680.txt)

## Collaboration
This project was made in collaboration with Frida Hæstrup. The work has been equally split. Apart from being the other brain of the idea of the pipeline and participating in project discussions, Frida contributed with modifications in bigram, trigram, corpus and dictionary functions as well as parameter tuning and plotting. Methods and results were also written in collaboration. 

## Data
Before you can access the data, you will need to create a kaggle API following the instructions here https://github.com/Kaggle/kaggle-api#:~:text=To%20use%20the%20Kaggle%20API,file%20containing%20your%20API%20credentials. When you have created an API and have a kaggle.json file, upload the file to worker02 if you are working there and move to the place the .json file is saved. When you are there you can execute the commands below:

```
mkdir ~/.kaggle/ # New folder 
mv kaggle.json ~/.kaggle/ # Move kaggle.json to this folder - required to activate api
```
Now you can download the data directly from your terminal by moving to the folder containing my repository and download the data by typing:

```
# When you are at the 'start' of my repository (i.e. cds-language-exam) you can type

cd data

mkdir 5

cd 5 # Changing directory 

kaggle datasets download -d  tentotheminus9/religious-and-philosophical-texts # Download data

unzip religious-and-philosophical-texts.zip # Unzip data
```

If you want to manually download the data, find it here: https://www.kaggle.com/tentotheminus9/religious-and-philosophical-texts

N.B.: If you do not download the data at this location, please change the input path in the commandline argument so that the script knows where to find the input data.

## Run script
### Commandline arguments
```
- "-f", "--filename", required=False, type=str, default= "../data/pg10.txt", help="str, filename for txt file"
                             
- "-o", "--outpath", required=False, type=str, default= os.path.join("..","out"), help="str, folder for output files"
    
- "-m", "--metric", required=False, default = "coherence", choices = ["coherence", "perplexity"],type=str, help="str, method to approximate number of topics with"

- "-n", "--num_topics", required=False, type=int, default = 5, help="int or none, number of topics to model"
```
To run the script, please type the following in the terminal:

```
cd src/5 

python3 religious_topics.py # Default will choose bible - change filename to run other religious text

```

## Methods
Some preprocessing steps were performed to prepare the data for analysis. First, we removed the beginning of the books containing an introduction to Project Gutenberg to only keep the content of the actual books. Then digits, most punctuation, and stop-words were removed. The text was split into individual sentences so that each sentence corresponds to a document in our Topic Model. We then created bigrams and trigrams based on words frequently appearing together, and performed lemmatization and pos-tagging using spaCy (Honnibal & Montani, 2017). Based on ‘part-of-speech’-tagging, we filtered words to keep only nouns and adjectives. As spaCy has a maximum of 1,000,000 characters per doc, we used the first 1,000,000 characters of each book. Lastly, a dictionary with an integer value id for each word and a corpus with a ‘bag-of-words'-model for all documents are returned. This corpus can then be fed into our model. 

Using gensim, we developed a topic-model where multiple parameters needed to be defined such as number of topics to cluster words and documents into and alpha and beta values as prior probabilities of words and topics. We made both a manual selection and automatic approximation of the number of topics possible; one could either write the number of topics or let the script calculate coherence and perplexity scores and select the number of topics with either highest coherence score or lowest perplexity score. Tuning hyperparameters of alpha and beta involved testing low (0.001), medium (1) and high (10) values of each metric and their combinations (following http://ethen8181.github.io/machine-learning/clustering/topic_model/LDA.html#:~:text=Understanding%20the%20role%20of%20LDA%20model%20hyperparameters,-Finally%2C%20we'll&text=alpha%20is%20a%20parameter%20that,word%20weights%20in%20each%20topic.). While this did not seem to yield any significant changes in performance, beta was kept at a fixed, automatic prior and alpha at a fixed, asymmetric prior. In addition, different values for iterations, passes, and chunk sizes were tested in order to balance convergence and speed of topic distribution updates, and to see how this influenced the topics (https://towardsdatascience.com/6-tips-to-optimize-an-nlp-topic-model-for-interpretability-20742f3047e2). 

From the resulting topic-term and document-topic probabilities obtained from the model, we extracted most probable words for each topic across the documents. To further inspect the importance and distribution of weights across words in the topics, we plotted every keyword’s overall frequency in the corpus combined with its weight in a specific topic. 

## Results

All results can be found in ```out/5```. Here, I will only highlight a subset of the results being the topics found in the Bible. Below, we see a plot of the weights and occurences of keywords in each topic.


<p align="left">
    <img src="../../out/3/pg10_word_weight_occurrence.png" alt="Logo" width="400" height="600">
  <p>

Within this plot, there are multiple things to consider. Firstly, we see that the word occurrences and weights align in most cases. Only a few times such as in topic 4 and 'son', the word occurs much more than it is weighted whereas with 'daughter' the opposite pattern is seen. I a couple of topics such as 2 and 3, the first keywords are highly occurring and important and the rest of the words have very small weights. This could indicate that the robustness of this topic isn't high when it only relies on a single word. 
      
An overall impression of the topics is that they are difficult to grasp and do not tap into specific themes. Although some words from similar categories occur (Topic 0: earth, place, land) it also contains other, diverging words (good, brother, thing). This is the case for most topics.

Also, a few general issues arise when attempting to extract meaningful information from religious texts using topic-modelling. In our case, a corpus was made up of only one text/book meaning that we had to divide the text up into documents. In such religious/philosophical texts different sections might be whole stories of their own rather than follow a narrative in the sense that many other books do. A concern was, therefore, to automatically split the books into meaningful chunks in a way that was generalizable across all books. Another issue lies in the available language tools to support such an analysis. As the en_core_web_sm model from spaCy has been trained on written web text (blogs, news, comment), it might not be as sensitive towards historical text data in which the vocabulary is different from what is used on websites. 

Another thought to consider is whether many of the same words (such as man, goat, wine, son) are used repeatedly but to express different themes in a metaphorical way. If this is the case, it would require more close readings of the texts. It is probably a different case for other text sources such as religious disussions on Reddit, with which studies have been successful at mapping topics and their differences(Stinea, Deitrick & Agarwala, 2020).

Combining this type of distant and close reading in topic modelling can in some cases yield very strong results, however, in this case it seems to require more time and expertise to extract meaningful topics and elements of each religious text. Despite this, we have developed a script and pipeline that can, with quite few modifications, be applied to new texts.
    


## References
Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.

Stinea, Z. K., Deitrick, J. E., & Agarwala, N. (2020). Comparative Religion, Topic Models, and Conceptualization: Towards the Characterization of Structural Relationships between Online Religious Discourses. Proceedings http://ceur-ws. org ISSN, 1613, 0073.


