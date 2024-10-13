Lesson 6-Task 1 Build and Train a Simple ANN Model

Nimal has started his own mobile company in Sri Lanka. He wants to be a challenge to big companies like Apple, Samsung, etc. However, he does not know how to estimate the price of mobile phones that his company creates. In this competitive mobile phone market, you cannot simply assume things. To solve this problem, he has collected sales data for mobile phones of various companies. Nimal wants to find a relationship between the features of a mobile phone (eg:- RAM, internal memory, etc.) and its selling price. As he is not very good at machine learning, he needs your help to solve this problem.
You are given a csv file as the dataset, and it contains the necessary attributes to evaluate the price range. Here are the 2 price ranges: high(1) and low(0).
Use your ML knowledge and try to develop a small ANN model and save weights using TensorFlow which will be important for future prediction of the price for given mobile phone specifications.
Steps
Read the data in CSV file
Split the data set into training and testing data where it contains 75% of the data as training and rest as testing data
Ann model architecture: – You can have the ANN with 2 hidden layer
1stlayer with 8 neurons/nodes
2nd layer with 4 neurons/nodes
Compile the model 
Fit, train the model for 100 epoch with batch size of 32
Save the weights file.
Attempt Quiz - Week 6 - Task 1 to verify your answers


Lesson 6 - Task 2 – Sentiment Analysis using NLP

Sentiment analysis is one of the popular downstream applications of Natural Language Processing, which determines the sentiment expressed in a piece of text. The sentiment expressed in a text is usually classified as positive or negative or neutral. You are expected to train a machine learning model to predict the sentiment of Tweets posted about US Airlines. 

Step 1 – Load the Dataset
Download the dataset using the following link:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
Upload the dataset to Google Colab directly or via Google Drive.
Read the dataset using pandas library.
The column “text” contains the tweets written by the users about different Airlines and the column “airline_sentiment” contains the class label indicating the sentiment of the “text” as positive or negative or neutral. We require only the columns "airline_sentiment " and " text " for the sentiment analysis task. Hence, drop the other columns in the dataset using the following code snippet.
df = df[["airline_sentiment", "text"]]

Step 2 – Preprocess Text
The following code snippet preprocesses the text by obtaining lower case, removing URLs, removing stop words, and generating stem. Add the code snippet to your notebook.
import nltk
import string
import re

from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('punkt')
ps = PorterStemmer()


defclean_text(text):
 text = text.lower()
 text = re.sub(r'http.?://[^\s]+[\s]?', '', text)
 text = nltk.word_tokenize(text)
 y = []
for i in text:
if i notin stopwords.words('english'):
 y.append(i)
 text = y[:]
 y.clear()
for i in text:
 y.append(ps.stem(i))
return" ".join(y)

Apply “clean_text” function to “text” column and assign the resulting text to new column named “text_cleaned”.

Step 3 – Feature Extraction
Create a “TfidfVectorizer” using the “sklearn” library by specifying “max_features” as 3000
Use “TfidfVectorizer” created to generate TF-IDF vector representation of cleaned text in the column “text_cleaned”, convert the vector representation to an array, and assigned it to a new variable named “X”.
Convert the column “airline_sentiment” to an array and assigned it to a new variable named “Y”.

Step 4 – Train Model
Split the dataset into training and testing sets using the function “train_test_split” available in the “sklearn” library. Set the parameters of the functions “test_size” to 0.2 and “random_state” to 2.
Train a multinomial Naïve Bayes classifier using the training dataset and predict the sentiment of tweets in the test dataset. Find the accuracy of the model in predicting sentiment labels for the test dataset.
Train a Random Forest classifier using the training dataset, predict the sentiment of tweets in the test dataset, and find the accuracy of the model using the following code snippet,
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))



the pdf file

