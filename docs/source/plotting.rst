Plotting the output
====================

In the first step, reviews of yelp dataset are shown using confusion matrix. Now, in this step, all the reviews will be plotted as a bubble plot using plotly library of python.
So, let's start...

Here, plotting is done for three attributes i.e., business_name, city_name and No.of reviews for a particular business in particular city. For that we need to save business_name, city_name and No.of reviews after predicting.
First, we have to perform sentiment analysis as explained in the :ref:`previous step <step>` (upto step 6).

Then follow the below steps:
 1. :ref:`Save the required data into csv file <step1>`
 2. :ref:`Create bubble plot using plotly <step2>`

.. _step1:

Save the required data into csv file
-------------------------------------

.. code-block:: python

   business_dataset = pd.read_csv("path for yelp_academic_dataset_business.csv")
   file1 = business_dataset[0:5000]

   review_dataset = pd.read_csv("path for yelp_academic_dataset_review.csv")
   file2 = review_dataset[0:5000]

   reviews = []
   filtered_reviews = []

   for idx in file1.iterrows():
    i = idx[0]
    for id in file2['business_id']:
        if id == file1['business_id'][i]:
            cty = file1['city'][i]
            business_name = file1['name'][i]
            t = file2['text'][i]
            c_t = vectorizer.transform([t])
            predict = model.predict(c_t)[0]
            # saving business_name and city_name
            if predict == 5.0:
                reviews.append("{} - {} - Good".format(business_name, cty))
            else:
                reviews.append("{} - {} - Bad".format(business_name, cty))

   new_file = open('Reviews.csv', 'w', newline='')
   writer = csv.writer(new_file)
   header = ['name', 'No.of reviews']
   writer.writerow(header)

   # saving No.of reviews for non-repetitive reviews
   for i in reviews:
    if i not in filtered_reviews:
        filtered_reviews.append(i)
        count = reviews.count(i)
        if count >= 5:
            DATA = [i, count]
            writer.writerow(DATA)

   
Here, in the first for loop, iterrows() function is used to get business data along with index values. Business Id's of both the datasets are checked and when the Id's are matched, review_comment from review dataset and business_name, city_name from business dataset are selected in that particular index. Then prediction is done on the text and if the prediction is 5, it will save the business_name and city_name with the review as good otherwise with the review as bad.

Then in the second for loop, checking for the repetition of review is done, if the review is not repetitive it will be saved into filtered_reviews variable. Each business and city may have many number of reviews, but only No.of revirews more than 5 are selected here for plotting.


.. _step2:

Creating a bubble plot using plotly
------------------------------------

.. plotly::
   :include-source: True

   import plotly.express as px
   import pandas as pd

   # Load the Reviews.csv file
   data = pd.read_csv("Reviews.csv")
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

It's done...
