"""
@author: Jaime de Boer
"""
from scipy import stats
from NaiveBayes import NaiveBayes
import numpy as np
from collections import Counter
import string
import re


class AttributeWeighting(NaiveBayes):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.words_ham = [re.sub(r'[^\w\s]', '', word).lower() for sent in
                          np.array(self.X)[np.where(self.y == 'ham')[0]] for word in sent.split(" ")]
        self.words_spam = [re.sub(r'[^\w\s]', '', word).lower().lower() for sent in
                           np.array(self.X)[np.where(self.y == 'spam')[0]] for word in sent.split(" ")]
        self.all_words = list(set(self.words_ham + self.words_spam))
        self.weights, self.gains = self.calc_weights() # O(m^2n^2)

    def calc_weights(self):
        """
        Assigns weights to all words in the dataset
        """
        # Gain Ratio of all elements into dataframe
        gains = {}
        sum_gains = 0

        for word in self.all_words:
            gains[word] = self.gain(word)
            sum_gains += gains[word]

        # Calculating weights using gain
        weights = {}
        for word in gains.keys():
            weights[word] = (gains[word] * len(self.all_words)) / sum_gains
        print("weights assigned!")
        return weights, gains

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
        # predict classes of messages
        y_pred = []

        for message in messages:
            # Split messages into separate words
            words = message.split(" ")

            # Initiate the probability of the entire message being either 'ham' or 'spam'
            p_mess_ham = self.prob_ham
            p_mess_spam = self.prob_spam

            for word in words:
                if word in self.weights.keys():
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

                    # Update total probabilities, taking weight into account
                    p_mess_ham *= p_ham ** self.weights[word]
                    p_mess_spam *= p_spam ** self.weights[word]

            # Decide whether the message is spam or not
            if p_mess_spam > p_mess_ham:
                y_pred.append('spam')
            else:
                y_pred.append('ham')
        print("parsed!")

        return np.array(y_pred)

    def gain(self, word):
        """
        Is implemented in the child classes, returns the gain that will be used (either gain ratio or information gain)
        """
        return -1

    def impurity(self, word):
        """
        Calculates the impurity of splitting data into 'spam' or 'ham' based on a word
        """
        # Initialize variables
        total = 1
        count_spam = 1
        count_ham = 1

        total_out = 1
        count_spam_out = 1
        count_ham_out = 1

        # Count the amount of times a word occurs in ham and spam
        for i in range(len(self.X)):
            if word in np.array(self.X)[i]:
                total += 1
                if np.array(self.y)[i] == 'spam':
                    count_spam += 1
                else:
                    count_ham += 1

            else:
                total_out += 1
                if np.array(self.y)[i] == 'spam':
                    count_spam_out += 1
                else:
                    count_ham_out += 1

        # Set p_i values
        p_i_spam = count_spam / total
        p_i_ham = count_ham / total

        p_i_spam_out = count_spam_out / total_out
        p_i_ham_out = count_ham_out / total_out

        return self.average_of_three(p_i_spam, p_i_ham), self.average_of_three(p_i_spam_out, p_i_ham_out)

    def impurity_start(self):
        """
        Calculates the impurity of the dataset before classification
        """
        counter = Counter(self.y)
        count_spam = counter['spam'] + 1
        count_ham = counter['ham'] + 1
        total = len(self.X) + 2

        p_i_spam = count_spam / total
        p_i_ham = count_ham / total

        return self.average_of_three(p_i_spam, p_i_ham)

    def average_of_three(self, p_i_spam, p_i_ham):
        """
        Calculates the impurity as the average of the entropy, gini and classification error
        """
        entropy = stats.entropy([p_i_spam, p_i_ham])
        gini = 1 - sum((p_i_spam ** 2, p_i_ham ** 2))
        classification_error = 1 - max(p_i_spam, p_i_ham)
        return 1 / 3 * (entropy + gini + classification_error)
