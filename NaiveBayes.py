"""
@author: Jaime de Boer
"""
import numpy as np
import string
from collections import Counter
import re


class NaiveBayes:
    """
    Regular Naive Bayes class
    """

    def __init__(self, X, y):
        """
        Sets the class attributes
        """
        self.X = X
        self.y = y

        counter = Counter(self.y)
        self.prob_ham = counter['ham'] / len(self.y)
        self.prob_spam = counter['spam'] / len(self.y)

        self.total_ham, self.count_ham, self.total_spam, self.count_spam = self.count_words()

    def count_words(self):
        """
        Counts the amount of times a word occurs in both the 'spam' and 'ham' classes.
        """

        # Putting all 'ham' words into a separate array from all 'spam' words
        words_ham = [re.sub(r'[^\w\s]', '', word).lower() for sent in
                     np.array(self.X)[np.where(self.y == 'ham')[0]] for word in sent.split(" ")]
        words_spam = [re.sub(r'[^\w\s]', '', word).lower().lower() for sent in
                      np.array(self.X)[np.where(self.y == 'spam')[0]] for word in sent.split(" ")]

        # Counting the amount of times a word occurs in the 'ham' category
        count_ham = {}
        for word in words_ham:
            if word in count_ham.keys():
                count_ham[word] += 1
            else:
                count_ham[word] = 2

        # Counting the amount of times a word occurs in the 'spam' category
        count_spam = {}
        for word in words_spam:
            if word in count_spam.keys():
                count_spam[word] += 1
            else:
                count_spam[word] = 2

        # Returning the amount of non-spam words, the non-spam words and their counts, the amount of spam words,
        # and the spam words and their counts
        return len(words_ham), count_ham, len(words_spam), count_spam

    def parse_messages(self, messages):
        """
        Takes an array of messages and sorts them into either 'ham' or 'spam', based on the training data

        Parameters
        ----------
        messages :
            Array of the to be parsed messages.

        Returns
        -------
        np.array
            the predicted class of the given messages
        """
        y_pred = []

        for message in messages:
            # Split messages into separate words
            words = message.split(" ")

            # Initiate the probability of the entire message being either 'ham' or 'spam'
            p_mess_ham = self.prob_ham
            p_mess_spam = self.prob_spam

            for word in words:
                word = word.translate(str.maketrans('', '', string.punctuation)).lower()  # Remove all punctuation

                # Calculate the probability of word being 'ham'
                if word in self.count_ham.keys():
                    p_ham = self.count_ham[word]
                else:
                    p_ham = 1
                p_ham /= self.total_ham

                # Calculate the probability of word being 'spam'
                if word in self.count_spam.keys():
                    p_spam = self.count_spam[word]
                else:
                    p_spam = 1
                p_spam /= self.total_spam

                # Update total probabilities
                p_mess_ham *= p_ham
                p_mess_spam *= p_spam

            # Decide whether the message is spam or not
            if p_mess_spam > p_mess_ham:
                y_pred.append('spam')
            else:
                y_pred.append('ham')

        return np.array(y_pred)
