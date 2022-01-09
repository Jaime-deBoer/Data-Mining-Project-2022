"""
@author: Jaime de Boer
"""
import math
import numpy as np
from AttributeWeighting import AttributeWeighting


class AttributeWeightingGR(AttributeWeighting):
    def gain(self, word):
        """
        Calculates the gain ratio of classifying based on word
        """
        total = 1
        for i in range(len(self.X)):
            if word in np.array(self.X)[i].split(" "):
                total += 1
        return self.information_gain(word, total) / self.intrinsic_info(total)

    def intrinsic_info(self, total):
        """
        Calculates the intrinsic information of classifying based on word
        """
        ratio = total / len(self.X)
        ratio_out = (len(self.all_words) - total) / len(self.X)
        return -1 * (sum((ratio * math.log(ratio), ratio_out * math.log(ratio_out))))

    def information_gain(self, word, total):
        """
        Calculates the information gain of classifying based on word
        """
        ratio = total / len(self.X)
        ratio_out = (len(self.all_words) - total) / len(self.X)
        impurity_start = self.impurity_start()
        impurity_in, impurity_out = self.impurity(word)
        gain_ratio = impurity_start - sum((impurity_in * ratio, impurity_out * ratio_out))
        return gain_ratio
