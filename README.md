# Bangkit-ML_CapstoneProject
- [Bangkit-ML\_CapstoneProject](#bangkit-ml_capstoneproject)
- [1. General Info](#1-general-info)
- [2. Roadmap](#2-roadmap)
- [3. Prerequisite](#3-prerequisite)
- [4. How to testthe API](#4-how-to-testthe-api)
- [5. Contact](#5-contact)

# 1. General Info
So, for the dataset, we combine the dataset from Kaggle and our dummy_data (we manually combines the dataset using Google Sheet).
Here's the Kaggle dataset: https://www.kaggle.com/datasets/brarajit18/student-feedback-dataset
consist of 12 columns and 185 rows

# 2. Roadmap
This is our short journey working for the machine learning model
- The first thing we did is collect the data from Kaggle [dataset](https://www.kaggle.com/datasets/brarajit18/student-feedback-dataset), then, we categorized the data into 2 columns using Google Sheets. After that, we added the data manually.
- Then we clean and process the raw data to make the final dataset "ReviewsEN.csv"
- Pre-processing data is the most crucial part of NLP (Sentiment Analysis), so we were very careful in selecting the words to be removed, converting the words to their base form, removing punctuation and numbers, removing duplicate data, and converting all sentences to lowercase letters.
- Next, we tried to create BERT using TensorFlow. Unfortunately, the accuracy was too low. Then, we attempted to improve it by adding an embedding layer using a [predefined embedding file](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)  and combining it with the CNN + GRU model.
- After we got the accuracy that we want, we download the model and covert into the TFLite.
- So, the last thing we create API using Flask using the model that we downloaded before (model.h5)

There are BERT Model + (PredefinedEmbedding + CNN + GRU) models you can try as a comparison without the main model

# 3. Prerequisite
Here are the technologies you should install if you are using Jupyter-notebook. If you're using Google Colab you don't need to install it just import the libraries
- Python
- TensorFlow:
```pip install tensorflow```
- NLTK:
```pip install nltk```
- NumPy
```pip install numpy```
- Keras:
```pip install keras```
- Pandas
```pip install pandas```
- Scikit-learn
```pip install scikit-learn```

Don't forget to download the predefined embedding file
Here is the link to download the file: https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt

# 4. How to test the API
Here are the steps to run the "main.py" program:
- First, you need to clone this repository
- If you are in "/Bangkit-ML_CapstoneProject", move to the API_Model directory, by using these command
```cd API_Model```
- Activate environment:
./env/Scripts/activate

- If there is an error, when you are trying to activate the env, use this Error Handling command:
```Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted```

- Install Python requirements:
```pip3 install -r requirements.txt```

- Run Program:
```flask --app main.py --debug run```

- Deactivate environment:
```Deactivate```

# 5. Contact
- Muhammad Rakha Wiratama – rakhawiratama10@gmail.com
- Andiny N. R – 
- Elma Margaretha Br Sebayang – elmamargarethasby@gmail.com
  
