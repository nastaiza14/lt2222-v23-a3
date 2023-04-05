import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
import glob
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
# Whatever other imports you need





def get_data(input_files, labels=False):
    """
    Function that extracts relevant information from emails.
    """
    email_data = []
    label_data = []
    for element in input_files:
        with open(element, "r") as f:
                data = f.read()
                if "----Original Message----" in data or "----Forwarded by" in data:
                    split_emails = data.split("-----")
                    email = split_emails[0]
                    # search and append label
                    label = os.path.dirname(element).split("/")[-1]
                    #label = re.search("From:([\w\ ,\.]*)", email)
                    label_data.append(label) # (label.group(1))
                    # search and append content of email
                    matched_mini_email = re.search("[\w.-]+:.*\n\B\n((.|\n)*)\n([\w.-]+)?", email)
                    if matched_mini_email == None:
                        email_data.append(" ")
                    else:
                        email_data.append(matched_mini_email.group(1))
                                         
                else:
                    # search and append label for pure email 
                    label = re.search("From:([\w\ ,\.]*)", data)
                    label_data.append(label.group(1))
                    # search and append content of pure email
                    matched_email = re.search("[\w.-]+:.*\n\B\n((.|\n)*)\n([\w.-]+)?", data)
                    if matched_email == None:
                        email_data.append(" ")
                    else:
                        email_data.append(matched_email.group(1))
                    
                    
    if labels == True:
          return label_data            
    else:
          return email_data 



def make_data_dict(texts, labels):
        """ Function to make dictionary with the data ."""
        result_dict = {label: [] for label in set(labels)}
        for label, text in zip(labels, texts):
                result_dict[label].append(text)
        return result_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()


    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.

    directory = args.inputdir
    email_files = glob.glob(os.path.join(directory,"**", "*" ), recursive=True)
    email_files = [f for f in email_files if os.path.isfile(f)]

    # Extracting the list of emails and the list of labels 
    processed_emails = get_data(email_files)
    processed_labels = get_data(email_files, labels=True)

    # Transforming data
    # Emails -> Data                       # args.dims down here
    vectorizer = CountVectorizer(max_features=args.dims)
    X = vectorizer.fit_transform(processed_emails)
    # Authors -> Labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(processed_labels)
    y = label_encoder.transform(processed_labels)
    #ordered_ys= list(label_encoder.classes_)


    # Spliting the data                                                          #args.testsize/100
    x_train, x_test, y_train, y_test = train_test_split(X, processed_labels, test_size=args.testsize/100, random_state=42)
    # Not giving it encoded labels


    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    df1 = pd.DataFrame({
        "Author": y_train,
        "Data": x_train
    })
    df2 = pd.DataFrame({
        "Author": y_test,
        "Data": x_test
    })
    df = pd.concat([df1, df2], keys=["train", "test"])


    print("Writing to {}...".format(args.outputfile))
    #Write the table out here.

    
    df.to_csv(args.outputfile, index=False)
    print("Done!")
    
