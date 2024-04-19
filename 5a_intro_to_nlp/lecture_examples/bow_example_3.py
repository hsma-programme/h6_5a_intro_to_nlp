# Import packages
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# A list of our documents
raw_documents = ["The HSMA Programme is brilliant.  It really is!",
                 "The HSMA Programme embraces the 80s",
                 "I thought the HSMA Programme is too 80s"]

# Create a CountVectorizer.  Specify we want English stopwords removed.
# Also specify that we want our vocabulary to look for ngrams that are minimum
# size 1 (unigrams) and maximum size 2 (bigrams).
# As we've removed stopwords, we will remove "the" so probably don't need
# trigrams here (that would have captured things like "The HSMA Programme")
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))

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

