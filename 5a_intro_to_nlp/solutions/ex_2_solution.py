# Import packages
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from numpy.linalg import norm

# Create new list to store reviews
raw_reviews = []

# Read in three reviews and add them to the list of reviews
for i in range(1, 4):
    with open(f"review_{i}.txt", "r") as f:
        review_text = f.read()
        raw_reviews.append(review_text)

# Create a CountVectorizer.  Specify we want English stopwords removed.
vectorizer = CountVectorizer(stop_words='english')
#vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
#vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))

# Generate a bag of words using the vectorizer
bag_of_words = vectorizer.fit_transform(raw_reviews)

# Create a Pandas DataFrame to more easily visualise the Bag of Words
# Grab the column names (words in our vocabulary).  Rows will represent the
# counts for each document.  We can view this in the console after running the
# code by simply typing bow_df
column_names = vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(bag_of_words.toarray(), columns=column_names)

# Create a NumPy array from the Pandas DataFrame (this will automatically
# discard column names, so you don't need to worry about that)
bow_np = bow_df.to_numpy()

# Calculate cosine similarity of reviews 1 and 2
cosine_sim_1_2 = (np.dot(bow_np[0], bow_np[1]) / 
                  (norm(bow_np[0]) * norm(bow_np[1])))

# Calculate cosine similarity of reviews 1 and 3
cosine_sim_1_3 = (np.dot(bow_np[0], bow_np[2]) / 
                  (norm(bow_np[0]) * norm(bow_np[2])))

# Calculate cosine similarity of reviews 2 and 3
cosine_sim_2_3 = (np.dot(bow_np[1], bow_np[2]) / 
                  (norm(bow_np[1]) * norm(bow_np[2])))

# Print results
print (f"Cosine similarity of reviews 1 and 2 is {cosine_sim_1_2:.2f}")
print (f"Cosine similarity of reviews 1 and 3 is {cosine_sim_1_3:.2f}")
print (f"Cosine similarity of reviews 2 and 3 is {cosine_sim_2_3:.2f}")

