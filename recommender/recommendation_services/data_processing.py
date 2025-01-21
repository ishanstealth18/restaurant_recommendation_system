import re
import string

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

input_source_file = pd.read_csv("../../zomato.csv", encoding='UTF-8')
pd.set_option('display.max_columns', None)

# remove unwanted columns from dataframe and returns updated dataframe
updated_input_file = input_source_file.drop(['url', 'phone'], axis=1)
# drop column using index number
updated_input_file = updated_input_file.drop(updated_input_file.columns[8], axis=1)

print(updated_input_file.info())

# check for any duplicate records
duplicate_values_flag = list(updated_input_file.duplicated())

if True in duplicate_values_flag:
    # remove duplicate values and returns same file as inplace=True
    print("Duplicates found !!")
    updated_input_file.drop_duplicates(inplace=True)
    print("After duplicates check: ", updated_input_file.info())

# check if null values are present
null_values_flag = updated_input_file.isnull().values.any()
if null_values_flag:
    print("Null values found !!")
    # remove null values
    updated_input_file.dropna(inplace=True)
    print("After removing null values: ", updated_input_file.info())

# rename few columns for more clarity
updated_input_file = updated_input_file.rename(
    columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})

# removing some more columns
updated_input_file = updated_input_file.drop(['online_order', 'book_table', 'votes', 'rest_type', 'menu_item', 'type'],
                                             axis=1)

# convert to .xlsx file
# updated_input_file.to_excel('updated_zomato.xlsx')
print("Again removing columns: ")
print(updated_input_file.info())

# transform column 'rate'
# replace '/5' with ''
updated_input_file['rate'] = updated_input_file['rate'].str.replace("/5", '')
updated_input_file['rate'] = updated_input_file['rate'].str.strip()
# removed invalid values from 'rate' column
updated_input_file = updated_input_file.drop(updated_input_file[updated_input_file['rate'] == 'NEW'].index)
updated_input_file = updated_input_file.drop(updated_input_file[updated_input_file['rate'] == '-'].index)
# convert data type to float
updated_input_file['rate'] = pd.to_numeric(updated_input_file['rate'])

# transform cost column
# remove ',' from the values
updated_input_file['cost'] = updated_input_file['cost'].str.replace(',', '')
updated_input_file['cost'] = updated_input_file['cost'].str.strip()

# print(updated_input_file.info())

# transform 'review list' column
# remove punctuations, lower case, remove junk values
updated_input_file['reviews_list'] = updated_input_file['reviews_list'].str.lower()


# function to remove punctuation from column
def remove_punctuations(input_str):
    input_str = str(input_str)
    translator = str.maketrans('', '', string.punctuation)
    return input_str.translate(translator)


updated_input_file['reviews_list'] = updated_input_file['reviews_list'].apply(lambda x: remove_punctuations(x))
updated_input_file['reviews_list'] = updated_input_file['reviews_list'].str.strip()


# function to encode and decode the column values to remove non-converted characters
def encode_decode(input_str):
    input_str = str(input_str)
    return input_str.encode('ascii', 'ignore').decode('ascii', 'ignore')


updated_input_file['reviews_list'] = updated_input_file['reviews_list'].apply(lambda x: encode_decode(x))

# Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


updated_input_file["reviews_list"] = updated_input_file["reviews_list"].apply(lambda text: remove_urls(text))


#updated_input_file.to_excel("updated_zomato.xlsx")

# create a training set
training_set_df = updated_input_file.sample(frac=0.5)
updated_input_file.reset_index(inplace=True, drop=True)

# create tf-idf matrix for 'reviews_list'
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(training_set_df['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)