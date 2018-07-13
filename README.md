# Project: Who is the author?
by Haolin Lu
edisonluhl@gmail.com
Jul 12, 2018

Given a corpus consisting of texts from Shakespeare, Charles Dickens, Herman Melville, etc, develop a system that is able to tell the most likely author who wrote the text.

## Can you explain the idea?
The data comes from the Gutenberg project ( http://www.gutenberg.org/ ). I download all the data by using the package gutenberg in python. This package contains a variety of scripts to make working with the Project Gutenberg body of public domain texts easier. Before use one of the gutenberg.query functions, we must populate the local metadata cache. 
```
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
from gutenberg.query import list_supported_metadatas

from gutenberg.acquire import get_metadata_cache
cache = get_metadata_cache()
cache.populate()
``` 
There are 100 observations in training data as for each of the ten famous authors I choose, ten English works are selected. It guarantees a balanced dataset. This makes up the corpus that would be further processed. There are 30 observations in test data as for each of the ten famous authors I choose, three English works are selected. Since some texts can't be downloaded from the website, I manually replace those texts with the other works from the same author.

In order to run machine learning algorithms we need to convert the text files into numerical feature vectors. Then I remove the stopwords, stem the texts and create the TF-IDF matrix in python to extract features from text file.

Then I run Support Vector Machine, Random Forest algorithms on training data and use 5-fold cross-validation to evaluate the performance of different models. Random forest model seems to outperform SVM on cross-validation but the difference of their performance on test data is rather small. Then I use grid search to tune the parameters and train one more model(Naive Bayes). I use majority vote as the ensemble method, which raises 0.2 in accuracy compared to single random forest model.

Last but not least, I train Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM), which is known to perform very well on text data. Unfortunately, this model performs worse than the emsembled model above. I think the main reason is we don;t have enough data to feed into the neural network. This can be addressed if we can have more data or use some data augumentation method. 

## What are good features if you were to solve it in a machine learning fashion?
In my opinion, good features are most commonly used and most representative with clear and obvious meanings. Hence I convert the text files into numerical feature vectors. I further remove the stopwords, stem the texts and create the TF-IDF matrix in python to extract features from text file.

## Is it possible to avoid manually defining features?
Yes. I think we should avoid manually defining features since they are subjective and would bring personal preference and bias to the model. At the end of the day, it's hard for us to tell good features from bad ones based on our own tastes.

## How would you evaluate the performance of this system?
I mainly use 5-fold cross-validation to evaluate the performance of my model. Both the training and test data are randomly shuffled into 5 equal size subsamples. I predict the authors for test dataset as well but that is not my primary method since test data should be held unseen before final call.

## Can you do some result analysis?
In terms of base model, I run Support Vector Machine, Random Forest and Naive Bayes algorithms. SVM gives the accuracy of 0.8(validation score) while Random Forest gives 0.92. This makes sense since Random Forest is intrinsically suited for multiclass problems, while SVM is intrinsically for two-class. Besides, Random Forest works well with a mixture of numerical and categorical features. When features are on the various scales, it is also fine. 

I do grid search and ensembling via majority vote with an overall accuracy of 0.93, which outperforms the single base model. The reason for that is by introducing more models, it can correct its wrong predictions from time to time.

RNN with LSTM works poorly here. The main reason for me is the lack of data. Given that RNN with LSTM is rather a complicated model, we need adequate data to train it and tune the parameters. 100 observations are not enough. 

## What if the target language is Chinese?
In Chinese the words are more frequently formed by a combination of characters. Hence we should train machine to learn which character/word should be combined with the right neighbor(s).

