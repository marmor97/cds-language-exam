## Text classification using Deep Learning
### Assignment 6

This assignment attempts to see how successfully we can use Deep Learning models classify a specific kind of cultural data - scripts from the TV series Game of Thrones. In particular, I am testing how accurately we can model the relationship between each season and the lines spoken. That is to say - can you predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season? We will start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well the model performs. Afterwards, I try to come up with a solution which uses a DL model, such as the CNNs we went over in class.


## Data
### Game of thrones 
Before you can access the data, you will need to create a kaggle API following the instructions here https://github.com/Kaggle/kaggle-api#:~:text=To%20use%20the%20Kaggle%20API,file%20containing%20your%20API%20credentials. When you have created an API and have a kaggle.json file, upload the file to worker02 if you are working there and move to the place the .json file is saved. When you are there you can execute the commands below:

```
mkdir ~/.kaggle/ # New folder 
mv kaggle.json ~/.kaggle/ # Move kaggle.json to this folder - required to activate api
```
Now you can download the data directly from your terminal by moving to the folder containing my repository and download the data by typing:

```
# When you are at the 'start' of my repository (i.e. cds-visual-exam) you can type

cd data # Changing directory 

mkdir 4 # Create folder 

cd 4 # Change directory to this folder

kaggle datasets download -d  albenft/game-of-thrones-script-all-seasons # Download data

unzip game-of-thrones-script-all-seasons.zip # Unzip data
```
N.B.: If you do not download the data at this location, please change the input path in the commandline argument so that the script knows where to find the input data.


### Glove
To run the script, you will also need to download the glove data containing word embeddings by moving to the terminal and type:

```
cd data/4 # If you have not moved to the directory

wget http://nlp.stanford.edu/data/glove.6B.zip # Download data

unzip -q glove.6B.zip # Unzip

```
## Run script

### Commandline arguments
#### Logistic Regression
```  
 - "-f", "--filename", type=str, default= "../../../data/Game_of_Thrones_Script.csv", help="str, file name and location"    
 
 - "-o", "--outpath", type=str, default= "./../../out/4"), help="str, output location"
```
#### Convolutional Neural Network
```
-  "-p", "--preprocess", required=False, choices = "add_character", "collapse_character", type=str, default= "add_character", help="preprocess method of sentences: add_character adds name of character in every sentence, collapse_character groups by character and episode and combine all sentences in one row per character"

- "-f", "--filename", required=False, type=str, default= "../../../data/4/Game_of_Thrones_Script.csv", help="str, file name and location"
    
- "-e", "--epochs", required=False, type=int, default= 60, help="int, number of epochs"   
    
- "-o", "--outpath", required=False, type=str, default= ../../../out/4, help="str, output location"

- "-ed", "--embed_dim", required=False, type=int, choices = 50,100,200,300, default=50, help="int, glove embeddings size" 
```
To run the script, please type: 

```
cd src/4

python3 cnn_got.py
```

## Methods
I will firstly go through preprocessing steps in the baseline model and afterwards the Deep Learning model. For the baseline model, we first load the data specified in the command line argument, whereafter it is balanced to have an equal amount of data in each label to avoid biasing prediction in the direction of the season with most lines. Afterwards, data is split into train and test sets consisting of 75 % and 25 % of the data, respectively. Lastly, the sentences from each season are vectorized using CountVectorizer(). The model itself is defined as a logistic regression and after this, we fit the model and gather predictions and produce a classification report. 

With regards to the Deep Learning model, a couple more steps are required. In general, establishing this model has been an iterative process of testing parameters and inspecting performance to see how high performance it could reach. After the data is loaded, the sentences are grouped by the episode and character meaning that all sentences belonging to one character are combined in one row. Like above, the amount of sentences is balanced, season labels are binarized into dummy-variables, and text and labels are split. X values are tokenized with tf.keras.Tokenizer() using 5000 words. Afterwards, a GloVe embedding model with 50 embedding dims is chosen. The tokenized sentences are padded with the length of the longest 'sentence' at the end of the sequence. At this point, the model is defined, starting with the embedding matrix as input layer. After this, a convolutional layer is added with relu activation. Then a Max Pooling Layer is made, then a fully-connected network with 48 neurons, then a droupout layer with 20% dropout and finally, an output layer with 8 neurons and softmax activation. The model is compiled with the loss function categorical cross entropy and Adam optimizer with a learning rate of 0.00001. A summary of the model architecture can also be seen in ```out/4```.



### Results

The F1-scores for the baseline model as well as the CNN can be seen in the table below.


| Season    | F1-score (LR / CNN) |
| --------- | --------|
| 1    | 27.20 % / 50.20 %       |
| 2 | 19.09 % / 30.61 %        |
| 3 | 22.06 % / 19.58 %        |
| 4 | 19.18 % / 29.76 %        |
| 5 | 18.79 % / 34.57 %        |
| 6 | 20.56 % / 12.86 %        |
| 7 | 27.20 % / 33.21 %        |
| 8 | 26.00 % / 39.25 %        |


We see that the CNN performs slightly better than the Logistic Regression and the best model was reached when adding the name of the character talking in the sentences. Performance increased mainly due to this and the application of regularization, dropout layers and a very low learning rate. The logistic regression performs best on season 1, 7 and 8 . These differing accuracies triggers some interesting questions about whether intro and final seasons are more expressively distinct. One can, however, question the generalizability of the Convolutional Neural Network as it is including character names. When this feature is included, the models is not entirely relying on linguistic features anymore and is fit to the specific character names in Game of Thrones meaning that performance might drop if it was introduced to new TV-series.

