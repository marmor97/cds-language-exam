# System tools
import os

# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm
import neuralcoref

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")
# https://spacy.io/universe/project/neuralcoref
neuralcoref.add_to_pipe(nlp)

def main():
    input_file = os.path.join("data","fake_or_real_news.csv")
    data = pd.read_csv(input_file)

    # Selecting only the REAL labelled data 
    real_df = data[data["label"]=="REAL"]["text"]

    text_spans = []
        
    # Every text / headline in the dataframe
    for doc in tqdm(nlp.pipe(real_df, batch_size=500)):
        # create temporary list 
        tmp_entities = []
        # create doc object

        # for every named entity
        for entity in doc.ents:
            # if that entity is a person
            if entity.label_ == "PERSON":
                # If it has co-reference
                if doc[entity.start]._.in_coref:
                    # Append the main reference 
                    tmp_entities.append(str(doc[entity.start]._.coref_clusters[0][0]))
                
                else:
                    # Or the name of the entity if it does not have
                    tmp_entities.append(entity.text)
                
        # append temp list to main list
        text_spans.append(tmp_entities)   

    # iterate over every document
    edgelist = []
    for text in text_spans:
        # use itertools.combinations() to create edgelist
        edges = list(combinations(text, 2))
        # for each combination - i.e. each pair of 'nodes'
        for edge in edges:
            # append this to final edgelist
            edgelist.append(tuple(sorted(edge)))


    # Summarizing the above to a weighted edgelist
    counted_edges = []
    for key, value in Counter(edgelist).items():
        source = key[0]
        target = key[1]
        weight = value
        counted_edges.append((source, target, weight))

    # Dataframe    
    counted_edges = pd.DataFrame(counted_edges, columns=["nodeA", "nodeB", "weights"])

    # Save
    outpath = os.path.join("data", "edgelist.csv")
    counted_edges.to_csv(outpath,index=False)

if __name__=="__main__":
    main() 
