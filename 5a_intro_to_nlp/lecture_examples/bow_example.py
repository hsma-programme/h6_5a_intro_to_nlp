# Import packages
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# A list of our documents
raw_documents = ["HSMA is a brilliant programme.  It really is!",
                 "HSMA embraces the 80s",
                 "I thought the programme is too 80s"]

# Create a CountVectorizer
vectorizer = CountVectorizer()

# Generate a bag of words using the vectorizer
bag_of_words = vectorizer.fit_transform(raw_documents)

# Create a Pandas DataFrame to more easily visualise the Bag of Words
# Grab the column names (words in our vocabulary).  Rows will represent the
# counts for each document.  We can view this in the console after running the
# code by simply typing bow_df
column_names = vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(bag_of_words.toarray(), columns=column_names)

# Print each document along with its bag of words vector representation
for i in range(0, len(raw_documents)):
    print (raw_documents[i])
    print ("becomes")
    print (bow_df.iloc[i].values)
    print ()

