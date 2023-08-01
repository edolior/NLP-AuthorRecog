Welcome! :)

A text classification system was implemented to identify authorship attributes.
The data is comprised from Donald Trump’s Tweets API on various media platforms.
The main goal is to differentiate between instances of Donald Trump and non-Donald Trump publishers, such as assistants who manage his social media publications.
Essential text features are extracted from the input to recognize patterns associating with a certain author. These methods include TF-IDF Vectorizing, additional feature extractions and data augmentations.
The resulting term embeddings were experimented and analyzed to find out which techniques help detect contributing patterns of author phrasing by maximum likelihood of margin separation between classes.
The last prediction model that was chosen is Long Short-Term Memory (LSTM).
Each model runs a 10-fold cross validation based on Stratified folds to better represent the imbalanced dataset.

Dataset download link: https://www.kaggle.com/datasets/headsortails/trump-twitter-archive
