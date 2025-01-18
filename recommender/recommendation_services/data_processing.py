import pandas as pd

input_source_file = pd.read_csv("../../zomato.csv")
pd.set_option('display.max_columns', None)

print(input_source_file.head(2))
