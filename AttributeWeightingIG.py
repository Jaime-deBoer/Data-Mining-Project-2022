"""
@author: Jaime de Boer
"""
import numpy as np
from AttributeWeighting import AttributeWeighting


class AttributeWeightingIG(AttributeWeighting):

    def gain(self, word):
        """
        Calculates the information gain of classifying based on word
        """
        total = 1
        for i in range(len(self.X)):
            if word in np.array(self.X)[i].split(" "):
                total += 1

        ratio = total / len(self.X)
        ratio_out = (len(self.all_words) - total) / len(self.X)
        impurity_start = self.impurity_start()
        impurity_in, impurity_out = self.impurity(word)
        information_gain = impurity_start - sum((impurity_in * ratio, impurity_out * ratio_out))
        return information_gain
