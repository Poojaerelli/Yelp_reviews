Performing contextual analysis
==============================

In this step, contextual analysis is performed on yelp reviews dataset. Actually what is it...?
Performing sentiment analysis will let you know whether a particular text is giving positive or negative review but we don't know why poeple are giving such reviews and why they are feeling in that way. So, here **contextual analysis** will help you to know the reason behind each review.

Here, contextual analysis is performed on reviews of different restaurant's in the yelp business dataset. For doing this, restaurants name from the business dataset and their respective review text from the review dataset are required.

This can be done in following steps:
 1. :ref:`merging business and review datasets <st1>`
 2. :ref:`performing contextual analysis on merged data <st2>`
 3. :ref:`plotting positive and negative contextual words using wordcloud <st3>`

.. _st1:

**1. merging yelp business and review datasets:**

Merging of both the datasets can be done as follows: 

* Firstly, load the business and review datasets using pandas library.
* Dropping out unwanted columns from the datasets.
* Select the required categories.
* Merge the datasets into a dataframe as chunks. To avoid heavy load on memory, data is merged as chunks of 500000 reviews at a time.
* Finally save the data as a csv file.

Now, let's code for this..

.. code-block:: python
   
   import pandas as pd

   # loading business and review datasets
   business_path = "path for yelp_academic_dataset_business.json file"
   review_path = "path for yelp_academic_dataset_review.json file"
   b_data = pd.read_json(business_path, lines=True)

   # dropping unwanted columns from business dataset
   columns_to_drop = ['hours', 'is_open', 'review_count', 'longitude', 'postal_code', 'latitude', 'attributes']
   b_data = b_data.drop(columns_to_drop, axis=1)

   # selecting only restaurants categories. Any of the required categories can be selected
   business_data = b_data[b_data['categories'].str.contains('Restaurants', case=False, na=False)]

   chunk_size = 500000
   r_data = pd.read_json(review_path, lines=True, chunksize=chunk_size)

   chunks = []
   for chunk in r_data:
    chunk = chunk.drop(['review_id', 'useful', 'funny', 'cool', 'user_id', 'date'], axis=1) # dropping unwanted columns from reviews dataset
    chunk = chunk.rename(columns={'stars': 'review_stars'}) # to avoid confusion with stars column in business dataset, stars column in reviews dataset is renamed as review_stars
    merged_data = pd.merge(business_data, chunk, on='business_id', how='inner') # merging the datasets
    chunks.append(merged_data)
    
   print("Completed merging")
   final_dataset = pd.concat(chunks, ignore_index=True, join='outer', axis=0)
   csv_name = "yelp_reviews_Restaurants_categories.csv"
   final_dataset.to_csv(csv_name, index=False) # writing the merged data into csv file

Cool!.. merging is done. Now, let's go for the actual task.

.. _st2:

**2. Performing contextual analysis on merged data:**

This process is somewhat similar to sentiment analysis except saving the positive and negative review texts into seperate files. So, let's dive into coding directly..

.. code-block:: python

   import csv
   import nltk
   import string
   import pandas as pd
   from nltk import word_tokenize
   from nltk.corpus import stopwords
   from sklearn.neural_network import MLPClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.feature_extraction.text import CountVectorizer

   # data preprocessing
   def preprocess(review_text):
    remove_pctn = [char for char in review_text if char not in string.punctuation]
    remove_pctn = ''.join(remove_pctn)
    lwr = [word.lower() for word in remove_pctn.split()]
    final_word = [word for word in lwr if word not in stopwords.words('english')]
    return final_word

   # loading the merged dataset
   FILE = pd.read_csv("yelp_reviews_Restaurants_categories.csv", encoding='charmap')
   file = FILE[:5000] # selecting only 5000 reviews

   filtered_data = file[(file['review_stars'] == 1) | (file['review_stars'] == 5)]
   x = filtered_data['text']
   y = filtered_data['review_stars']
   vectorizer = CountVectorizer(analyzer=preprocess).fit(x)
   x = vectorizer.transform(x)

   X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
   model = MLPClassifier()
   model.fit(X_train, y_train)

   pos_comments = []
   neg_comments = []

   for idx in file.iterrows():
    i = idx[0]
    name = file['name'][i]
    t = file['text'][i]
    c_t = vectorizer.transform([t])
    predict = model.predict(c_t)[0]
    if predict == 5:
        pos_comments.append("{}-{}".format(name, t))
    else:
        neg_comments.append("{}-{}".format(name, t))

   # For positive comments
   newfile = open('Pos_context.csv', 'w', encoding='charmap')
   header = ['business_name', 'positive_comment']
   dictwriter = csv.DictWriter(newfile, fieldnames=header)
   dictwriter.writeheader()
   b_name = []
   p_comment = []

   for row in pos_comments:
    row = row.split("-")
    if row[0] not in b_name:
        b_name.append(row[0])
    if row[1] not in p_comment:
        p_comment.append(row[1])
        token = word_tokenize(row[1])
        token = [word for word in token if word not in string.punctuation]
        bi_grams = nltk.bigrams(token)
        for gram in bi_grams:
            tags = nltk.pos_tag(gram)
            for tag in tags:
                if tag[1] == 'JJ':
                    dictwriter.writerow({'business_name': row[0], 'positive_comment': tag[0]})

   newfile.close()

   # For negative comments
   newfile1 = open('Neg_context.csv', 'w', encoding='charmap')
   header1 = ['business_name', 'negative_comment']
   dictwriter1 = csv.DictWriter(newfile1, fieldnames=header1)
   dictwriter1.writeheader()
   b_name1 = []
   n_comment = []

   for row in neg_comments:
    row = row.split("-")
    if row[0] not in b_name1:
        b_name1.append(row[0])
    if row[1] not in n_comment:
        n_comment.append(row[1])
        token = word_tokenize(row[1])
        token = [word for word in token if word not in string.punctuation]
        bi_grams = nltk.bigrams(token)
        for gram in bi_grams:
            tags = nltk.pos_tag(gram)
            for tag in tags:
                if tag[1] == 'JJ':
                    if tag[0] != 'good':
                        dictwriter1.writerow({'business_name': row[0], 'negative_comment': tag[0]})

   newfile1.close()

In the above code, after training the model each review text is saved into seperate pos_comments and neg_comments lists. Then each review text from pos_comment list is tokenized and divided into bigrams to extract the adjectives by applying pos-tagging.
Those adjectives are saved into a csv file using dictwriter function. Same thing is done for neg_comments list also. After completing this step, you will be having two csv files for positive and negative comments.

.. _st3:

**3. Plotting positive and negative contextual words using wordcloud:**

Wordcloud can be used to display the words in a sentence based on it's frequency. Frequency defines how many times the word is repeated. More the frequency, more the size of the word.

Let's code for this..

**Plotting positive comments:**

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   from wordcloud import WordCloud, STOPWORDS

   file = pd.read_csv('Pos_context.csv')
   text = " ".join(line for line in file.positive_comment)

   word = WordCloud(width=1000, height=1000, random_state=1, background_color='black', colormap='Set2',
                 collocations=False, stopwords=STOPWORDS).generate(text)
   plt.imshow(word, interpolation='bilinear')
   plt.title('Positive comments for yelp restaurants')
   plt.axis("off")
   plt.show()

Executing this code snippet will give you the output as below,

.. figure:: /images/positive.png
   :align: center

**Plotting negative comments:**

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   from wordcloud import WordCloud, STOPWORDS

   file = pd.read_csv('Neg_context.csv')
   text = " ".join(line for line in file.negative_comment)

   word = WordCloud(width=1000, height=1000, random_state=1, background_color='black', colormap='Set2',
                 collocations=False, stopwords=STOPWORDS).generate(text)
   plt.imshow(word, interpolation='bilinear')
   plt.title('Negative comments for yelp restaurants')
   plt.axis("off")
   plt.show()

This will give you the output as below,

.. figure:: /images/negative.png
   :align: center

It's done..

Thanks for reading..