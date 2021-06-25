import plotly.express as px
import pandas as pd

# Load the Reviews.csv file
data = pd.read_csv("C:/Users/sbtechnologies/PycharmProjects/pythonProject/Reviews.csv")
filtered_data = []

# This for loop is for removing all the special characters and prefix 'b' for all the words
for row in data['name']:
 row = row.replace("&", "")
 row = row.replace("'", "")
 row = row.replace('"', "")
 row = row.replace("b", "")
 row = row.replace('b"', "")
 row = row.replace("/", "")
 row = row.replace("\\", "")
 filtered_data.append(row)

 # Saving filtered data into new column 'name'
data['name'] = filtered_data
city = []
review_type = []
business_name = []

# Splitting data to get business_name, city_name as well as review type
for i in data.name:
 splt = i.split('-')
 # print(splt[2])
 business_name.append(splt[0])
 city.append(splt[1])
 review_type.append(splt[2])

# Saving business_name into new column 'business_name'
data['business_name'] = business_name
# Saving city into new column 'city'
data['city'] = city
# Saving review_type into new column 'review_type'
data['review_type'] = review_type

# Creating bubble plot
fig = px.scatter_3d(data, x='business_name', y='city', z='No.of reviews', hover_data=['review_type'], title='Yelp reviews',
                 size='No.of reviews', size_max=100, color='city')
fig.show()