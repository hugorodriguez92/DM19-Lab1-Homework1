import nltk
import numpy as np

"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)

def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens


# Created helpers for HM1 HR

# creates a dictionary from the array of lines of the combined .tx files
def sentiment_dataframe_dictionary(array):
    result_dictionary = {'sentences':[], 'scores':[], 'sources':[]}
    temporal_array = [line.split("\t") for line in array if len(line.split("\t")) == 3 and line.split("\t")[0] is not None and line.split("\t")[1] is not None and line.split("\t")[2] is not None]
    for line in temporal_array:
        result_dictionary['sentences'] += [line[0].strip("\n\t")]
        result_dictionary['scores'] += [line[1]]
        result_dictionary['sources'] += [line[2]]
    return result_dictionary
    

## extended jaccard coefficient, own created method 
def extended_jaccard_coefficient(vector1,vector2):
    dot_product = vector1 @ vector2
    magnitude_v1 = np.linalg.norm(vector1)
    magnitude_v2 = np.linalg.norm(vector2)
    divisor = np.square(magnitude_v1) + np.square(magnitude_v2) - dot_product
    result = 0
    if divisor != 0:
        result = dot_product/divisor
    return result
        