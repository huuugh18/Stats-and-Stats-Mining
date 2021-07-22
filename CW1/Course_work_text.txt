# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Course Work - Statistics and Statistics Mining
# ## Patrick O'Neill
# 
# 
# 
# 
# ### Import Libraries

# %%
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform

import nltk
from nltk.stem import PorterStemmer

import re
import numpy as np
import pandas as pd
import pickle

# %% [markdown]
# ### Load and explore the data (4 marks)

# %%
data = pd.read_csv('product-category-dataset-improved.csv')
df = pd.DataFrame(data)
df.describe()

# 15 level_1 classes # 36 level_2 classes #94 level_3 classes


# %%
df.head()

# %% [markdown]
# ### Deal with Missing Data (4 marks)

# %%
# Check if data has missing values in the Description column

missing_descriptions_indices = df[df['Description'].isnull()].index.tolist()

print('There are', len(missing_descriptions_indices), 'missing indices')


# %%
# Remove missing descriptions rows from dataframe
df = df[df['Description'].notna()]
df.shape

# %% [markdown]
# Shape is 10627 rows which is 12 less than original 10639 so know we have dropped the correct amount of rows from the dataframe
# 
# ## Create subset of data to workwith as dataset too large

# %%
# take sample of 8500 of total count to reduce computational load for the classifiers

df = df.sample(n = 8500)
df.reset_index(inplace=True, drop=True)

# check shape to make sure correct transformation
print(df.shape)
df.head()

# %% [markdown]
# ### Drop Classes where the number of instances is < 10 (4 marks)

# %%
# Apply to Level_1 
print(df.Level_1.value_counts())
print('Number of Unique Level 1 Categories: ', df.Level_1.nunique())
# No classes have less than 10 instances

# %% [markdown]
# There are 15 level 1 classes all of which have more than 10 instances. 
# No classes will be dropped from level 1.

# %%
# Apply to Level_2

# create mask based on value counts
mask_2 = df.Level_2.value_counts()
# apply mask to dataset
df = df[df['Level_2'].isin(mask_2.index[mask_2>9])]
print('Number of Unique Level 2 Categories: ', df.Level_2.nunique())

#confirm no classes left have fewer than 10 instances
df.Level_2.value_counts()

# %% [markdown]
# All remaining level 2 classes have more than 10 instances.

# %%
# Apply to Level_3

#create mask 
mask_3 = df.Level_3.value_counts()
#apply mask
df = df[df['Level_3'].isin(mask_3.index[mask_3>9])]

print('Number of Unique Level 3 Categories: ', df.Level_3.nunique())

# check value counts all above 10 instances
df.Level_3.value_counts()

# %% [markdown]
# All remaining level 3 classes have more than 10 instances.
# 
# ### Now let's write a Function to Prepare Text (4 marks)
# We will apply it to our DataFrame later on
# 
# * This function receives a text string and performs the following:
# * Convert text to lower case
# * Remove punctuation marks
# * Apply stemming using the popular Snowball or Porter Stemmer (optional)
# * Apply NGram Tokenisation
# * Return the tokenised text as a list of strings

# %%
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string
import re
snowball_stemmer = SnowballStemmer(language='english')
porter_stemmer = PorterStemmer()

nltk.download('punkt')


def scrub_words(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[_]','', text)
    return text

def process_text(text):
    # 1. Convert text to lower case and remove all punctuation
    scrubbed_text =scrub_words(text)                
    
    # 2. Tokenize words
    token_words = word_tokenize(scrubbed_text)      

    #3. Apply stemming
    stem_words = [snowball_stemmer.stem(w) for w in token_words] 
    
    # 4. Apply Ngram Tokenisation
    return ' '.join(stem_words)

    


# %%
# Here is an example function call

process_text("Here we're testing the process_text function, results are as follows:")

# %% [markdown]
# ## Apply TF-IDF to extract features from plain text (10 marks)
# 

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import chain

# %% [markdown]
# ## Use TFIDF Vectorizer which combines the process of using Count Vectorizer followed and TFIDF Transformer into one Process

# %%
vectorizer = TfidfVectorizer(ngram_range=(3,3), preprocessor=process_text, max_features=15000)

X = vectorizer.fit_transform(df['Description'].values)

tfidf_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())


# %%
print(tfidf_df.shape)
tfidf_df.head()

# %% [markdown]
# ## Now the Data is Ready for Classifier Usage
# 
# ### Split Data into Train and Test sets (4 marks)
# 

# %%
# combine dfs before creating test and train datasets

tfidf_df.reset_index(inplace=True, drop=True)
df.reset_index(inplace=True, drop=True)

data = pd.concat([df, tfidf_df], axis=1)


# %%
train, test = train_test_split(data, test_size=0.2, random_state=1811)


# %%
X_train = train.iloc[:, 4:]
y_train = train.iloc[:, 0: 4]
X_test = test.iloc[:, 4:]
y_test = test.iloc[:, 0: 4]


# %%
# Reset index in each dataframe 

X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)


# %%
# Take classes as separate columns 

class1_train = y_train['Level_1'].astype(str)
class1_test = y_test['Level_1'].astype(str)

class2_train = y_train['Level_2'].astype(str)
class2_test = y_test['Level_2'].astype(str)

class3_train = y_train['Level_3'].astype(str)
class3_test = y_test['Level_3'].astype(str)

# %% [markdown]
# ## Model training for the three levels (8 marks)
# 

# %%
# Create and save model for level 1
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, class1_train)
score = classifier.score(X_test, class1_test)
print('accuracy: ', score)


# %%
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, class1_train)
score = classifier.score(X_test, class1_test)
print('Accuracy: ', score)


# %%
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier().fit(X_train, class1_train)
score = classifier.score(X_test, class1_test)
print('Accuracy: ', score)

with open('level1.pk', 'wb') as cls:
    pickle.dump(classifier, cls)

# %% [markdown]
# The MultinomialNB Model classifier had the greatest accuarcy of the three classifiers tested so using that going forward for model training of class 2 and 3.
# %% [markdown]
# ## Create and save models for level 2

# %%
# in this space you will be fitting models for the level 2 data 
#so you will want to split the test data into sub data for each classification e.g.

#get unique level 1 categories
level1cats = class1_train.unique()

lvl1_cat_indexes = []
lvl1_cat_indexes_test = []
lvl1_unique = []

for index, cat in enumerate(level1cats): 
    print('Level 1 Category: ', cat)
    #get indexes for train data and test data
    a = list(class1_train[class1_train == cat].index)
    b = list(class1_test[class1_test == cat].index)
    lvl1_cat_indexes.append(a)
    lvl1_cat_indexes_test.append(b)

    # some class 2 data only has one unique value for a particular level 1 category so can't create a model
    if class2_train.loc[a].nunique() == 1:
        unique_val = class2_train.loc[a].unique()[0]
        print('ERROR: ONLY ONE UNIQUE VALUE:', unique_val, ' SO SKIP MODEL CREATION \n')
        #put values into array for prediction later
        lvl1_unique.append([cat, unique_val])
        continue
    #create model with train data for unique level 1 category
    classifier = MultinomialNB()
    classifier.fit(X_train.loc[a], class2_train[a])

    score = classifier.score(X_test.loc[b], class2_test[b])
    print('\n Accuracy score for LVL 1 CAT: ', cat, ' SCORE: ', score)

    #save model
    model_name = 'level2_' + cat + '.pk'
    with open(model_name, 'wb') as cls: 
        pickle.dump(classifier, cls)

# %% [markdown]
# ## Create and save models for level 3

# %%
# get unique level 2 categories
level2cats = class2_train.unique()

lvl2_cat_indexes = []
lvl2_cat_indexes_test = []
lvl2_unique = []
for index, cat in enumerate(level2cats): 
    print('Level 2 Category: ', cat)
    #get indexes for train data and test data
    a = list(class2_train[class2_train == cat].index)
    b = list(class2_test[class2_test == cat].index)

    lvl2_cat_indexes.append(a)
    lvl2_cat_indexes_test.append(b)

    # some class 2 data only has one unique value for a particular level 1 category so can't create a model
    if class3_train.loc[a].nunique() == 1:
        unique_val = class3_train.loc[a].unique()[0]
        print('ERROR: ONLY ONE UNIQUE VALUE:', unique_val, ' SO SKIP MODEL CREATION \n')
        #put values into array for prediction later
        lvl2_unique.append([cat, unique_val])
        continue
    #create model with train data for unique level 2 category
    classifier = MultinomialNB()
    classifier.fit(X_train.loc[a], class3_train[a])
    score = classifier.score(X_test.loc[b], class3_test[b])
    print('\n Accuracy score for LVL 2 CAT: ', cat, ' SCORE: ', score)

    #save model
    model_name = 'level3_' + cat + '.pk'
    with open(model_name, 'wb') as cls: 
        pickle.dump(classifier, cls)

# %% [markdown]
# ## Predict the test set (8 marks)
# 
# ## Predict Level 1

# %%
# Creating an empty Dataframe with column names only
results = pd.DataFrame(columns=['Level1_Pred', 'Level2_Pred', 'Level3_Pred'])

## Here we reload the saved models and use them to predict the levels
# load model for level 1 (done for you)
with open('level1.pk', 'rb') as nb:
    model = pickle.load(nb)

## loop through the test data, predict level 1
level1_pred = model.predict(X_test)
results['Level1_Pred'] = level1_pred
results.head()

# %% [markdown]
# ## Predict Level 2

# %%
# for each category in level 1 predictions => use that plus the model for that category to predict level 2

flat_lvl1_unique = [element for sublist in lvl1_unique for element in sublist]

for index, cat in enumerate(level1cats): 
    print('Level 1 Category: ', cat)
    # get indexes
    a = list(results[results['Level1_Pred']== cat].index)
    # if cat is in the arrayof lvl1_unique => set predicted values to its pair
    if cat in flat_lvl1_unique:
        index = flat_lvl1_unique.index(cat)
        predicted = flat_lvl1_unique[index+1]
        print('Unique Category - no model')
        results['Level2_Pred'].loc[a] =  predicted
        continue
    # get model
    model_name = 'level2_' + cat + '.pk'
    with open(model_name, 'rb') as nb:
        model = pickle.load(nb)
    results['Level2_Pred'].loc[a] =  model.predict(X_test.loc[a])

# %% [markdown]
# ## Predict Level 3

# %%
# for each category in level 1 predictions => use that plus the model for that category to predict level 2

flat_lvl2_unique = [element for sublist in lvl2_unique for element in sublist]

for index, cat in enumerate(level2cats): 
    print('Level 2 Category: ', cat)
    # get indexes
    a = list(results[results['Level2_Pred']== cat].index)
    # if category is in the arraykof lvl1_unique => set predicted values to its pair
    if cat in flat_lvl2_unique:
        index = flat_lvl2_unique.index(cat)
        predicted = flat_lvl2_unique[index+1]
        print('Unique Category - no model')
        results['Level3_Pred'].loc[a] =  predicted
        continue
    # get model
    model_name = 'level3_' + cat + '.pk'
    with open(model_name, 'rb') as nb:
        model = pickle.load(nb)
    results['Level3_Pred'].loc[a] =  model.predict(X_test.loc[a])


# %%
results

# %% [markdown]
# ## Compute Accuracy on each level (4 marks)
# Now you have the predictions for each level (in the test data), and you also have the actual levels, you can compute the accurcay

# %%
# Level 1 accuracy
print('LEVEL 1 ACCURACY: ', accuracy_score(y_test['Level_1'], level1_pred))


# %%
# Level 2 accuracy
print('LEVEL 2 ACCURACY: ', accuracy_score(y_test['Level_2'], results['Level2_Pred']))


# %%
# Level 3 accuracy
print('LEVEL 3 ACCURACY: ', accuracy_score(y_test['Level_3'], results['Level3_Pred']))

# %% [markdown]
# ## Well done!

