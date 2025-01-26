import re
import string
from pandas.io.formats.style import Styler
import pandas as pd
import numpy as np
from crispy_forms.layout import HTML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


def process_data():
    pd.set_option("display.max_columns", None)
    pd.set_option('display.max_colwidth', None)
    input_source_file = pd.read_csv("zomato.csv", encoding='UTF-8')

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
    updated_input_file = updated_input_file.drop(
        ['online_order', 'book_table', 'votes', 'rest_type', 'menu_item', 'type'],
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

    # updated_input_file.to_excel("updated_zomato.xlsx")

    # create a training set
    training_set_df = updated_input_file.sample(frac=0.5)

    # remove duplicate restaurant names
    training_set_df = training_set_df.drop_duplicates(subset=['name'])

    # resetting index for dataframe
    training_set_df.reset_index(inplace=True, drop=True)

    # setting index as 'name' column
    training_set_df.set_index('name', inplace=True)
    indices = pd.Series(training_set_df.index)
    print("Indices: ", indices)

    # create tf-idf matrix for 'reviews_list'
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(training_set_df['reviews_list'])
    print(tfidf_matrix.shape)

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    # similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # print(cosine_similarities)
    # print(cosine_similarities)

    return training_set_df, cosine_similarities, indices


def recommend_restaurant(name):
    # list for top restaurants
    recommended_restaurant_list = []

    input_file, cos_values, indices = process_data()

    # find the index of the hotel entered
    restaurant_index = indices[indices == name].index[0]
    print("Restaurant index:", restaurant_index)

    # get cos similar values list for that entered restaurant
    restaurant_score = pd.Series(cos_values[restaurant_index]).sort_values(ascending=False)
    # restaurant_score = restaurant_score.tolist()
    print("Restaurant Scores: ", restaurant_score)

    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(restaurant_score.iloc[1:31].index)
    print(top30_indexes)


    cuisine_list = []
    cost_list = []
    # Names of the top 30 restaurants
    for each in top30_indexes:
        # print((input_file.index[each]))
        recommended_restaurant_list.append(list(input_file.index)[each])
        cuisine_list.append(list(input_file.cuisines)[each])
        cost_list.append(list(input_file.cost)[each])

    recommended_restaurant_df = pd.DataFrame(recommended_restaurant_list, columns=['Restaurant Name'])
    recommended_restaurant_df['Cuisines'] = cuisine_list
    recommended_restaurant_df['Cost'] = cost_list

    print(recommended_restaurant_df)

    return recommended_restaurant_df


#recommend_restaurant("Caf-Eleven")
