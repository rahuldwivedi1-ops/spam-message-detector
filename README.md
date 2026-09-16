# SMS Spam Message Detector

A Machine Learning based project that detects whether an SMS message is **Spam** or **Ham (Not Spam)**.

## Project Description

This project uses Natural Language Processing (NLP) and Machine Learning techniques to classify SMS messages into two categories:

**Spam** – Unwanted or promotional messages
**Ham** – Normal/legitimate messages

## Technologies Used

Python
Pandas
Scikit-learn
TF-IDF Vectorization
Multinomial Naive Bayes

## Machine Learning Workflow

1. Load the SMS dataset
2. Separate message text and labels
3. Convert text into numerical features using TF-IDF
4. Split the dataset into training and testing data
5. Train a Multinomial Naive Bayes model
6. Test the model on unseen messages
7. Calculate prediction accuracy
8. Classify new SMS messages as Spam or Ham

## Dataset

The project uses the **SMS Spam Collection Dataset**.

The dataset contains SMS messages labeled as spam or ham.

## Installation

Install the required Python libraries:

bashpip install -r requirements.txt
