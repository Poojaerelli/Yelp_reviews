.. _step:

Performing sentiment analysis
==============================

Sentiment analysis is one of the sub-fields of natural language processing (NLP). It is about finding out number of positive and negative reviews for a particular business.

This can be done in 7 steps:
 1. :ref:`importing required libraries <s1>`
 2. :ref:`loading the data <s2>`
 3. :ref:`preprocessing of text data <s3>`
 4. :ref:`vectorizing the data <s4>`
 5. :ref:`Splitting data into train and test <s5>`
 6. :ref:`Training the model <s6>`
 7. :ref:`Performing sentiment analysis <s7>`


.. _s1:

*STEP 1:* **Importing required libraries**

.. code-block:: python
   
   import string
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.feature_extraction.text import CountVectorizer

   from nltk.corpus import stopwords
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import confusion_matrix, classification_report
   from sklearn.metrics import accuracy_score
   from sklearn.neural_network import MLPClassifier


.. _s2:

*STEP 2:* **Loading the data**

.. code-block:: python

   data = pd.read_csv("path for yelp_academic_dataset_review.csv")
   print(df.head())

It will print the following output.

.. figure:: /images/yelp_head.png
   :scale: 60%
   :align: center
   

.. _s3:

*STEP 3:* **Preprocessing of text data**

.. code-block:: python
   
   data_classes = data[(data['stars'] == 1) | (data['stars'] == 5)] #Selecting only two types of reviews, 1 for negative review and 5 for positive review

   def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

This step will remove all the stopwords, numbers and punctuation marks.


.. _s4:

*STEP 4:* **Vectorizing the data**

.. code-block:: python
   
   x = data_classes['text']
   y = data_classes['stars']
   vocab = CountVectorizer(analyzer=text_process).fit(x)
   print('vocab_len', len(vocab.vocabulary_))
   x = vocab.transform(x)
   print('x', x)

This will print the output like this.

.. figure:: /images/yelp_vocab.png
   :scale: 60%
   :align: center


.. _s5:

*STEP 5:* **Splitting data into train and test**

.. code-block:: python

   X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

This step will divide the dataset into train and test sets.


.. _s6:

*STEP 6:* **Training the model**

.. code-block:: python
   
   model = MLPClassifier()
   model.fit(X_train, y_train)
   y_hat = model.predict(X_test)
   print('predict', len(y_hat))

This step will train the model and prints length of prediction. You can use any of other models from sklearn, MLPClassifier() model gives best accuracy for this dataset. 


.. _s7:

*STEP 7:* **Performing sentiment analysis**

.. code-block:: python
   
   def evaluation(y, y_hat, classes, title='Confusion_Matrix'):
    cm = confusion_matrix(y, y_hat)
    report = classification_report(y, y_hat)
    tick_marks = np.arange(len(classes))
    print("Score:", round(accuracy_score(y_test, y_hat) * 100, 2))
    print('report', report)
    sns.heatmap(cm, cmap='PuBu', annot=True, fmt='g', annot_kws={'size': 20})
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(title, fontsize=18)
    plt.show()


   evaluation(y_test, y_hat, classes=['negative(1)', 'positive(5)'])

This will show output as below.

.. figure:: /images/yelp_confusion_matrix.png
   :scale: 60%
   :align: center

   **Fig:** Confusion matrix


.. figure:: /images/yelp_clfn_report.png
   :scale: 60%
   :align: center

   **Fig:** Classification report


All together
-------------
Combining all the above steps, entire code will look like this,

.. code-block:: python

   import string
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.feature_extraction.text import CountVectorizer
   from nltk.corpus import stopwords
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import confusion_matrix, classification_report
   from sklearn.metrics import accuracy_score
   from sklearn.neural_network import MLPClassifier


   def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


   DATA = pd.read_csv("path for yelp_academic_dataset_review.csv")
   data = DATA[0:5000]

   data_classes = data[(data['stars'] == 1) | (data['stars'] == 5)]
   x = data_classes['text']
   y = data_classes['stars']
   vocab = CountVectorizer(analyzer=text_process).fit(x)
   x = vocab.transform(x)
   X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

   model = MLPClassifier()
   model.fit(X_train, y_train)
   y_hat = model.predict(X_test)
   print('predict', len(y_hat))


   def evaluation(y, y_hat, classes, title='Confusion_Matrix'):
    cm = confusion_matrix(y, y_hat)
    report = classification_report(y, y_hat)
    tick_marks = np.arange(len(classes))
    print("Score:", round(accuracy_score(y_test, y_hat) * 100, 2))
    print('report', report)
    sns.heatmap(cm, cmap='PuBu', annot=True, fmt='g', annot_kws={'size': 20})
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(title, fontsize=18)
    plt.show()


   evaluation(y_test, y_hat, classes=['negative(1)', 'positive(5)'])


In this step, sentiment analysis is performed by selecting only 5000 reviews (for reference) from the entire dataset, and the output is as below.

.. figure:: /images/yelp_final_output.png
   :scale: 60%
   :align: center

   **Fig:** Confusion matrix


.. figure:: /images/yelp_final_report.png
   :scale: 60%
   :align: center

   **Fig:** Classification report

