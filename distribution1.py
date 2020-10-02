'''
Author: Aaron Elledge
Date: 2/10/2020
Class: ISTA 311
Collaborators: Warren Hill
'''
import numpy as np

class Distribution (object):

    def _init_(self, dis):

#The dictionary is created:
#If the variable dis is a dict, it will become a dictionary
        if isin(dis, diction):
            self.d = diction
        elif isin(dis, tup):
            self.d = {}
            n = len(diction[0])
        for i in range(n):
            self.d.default(dis[0][i], dis[1][i])
        else:
            self.d = {}
            n = len(dis)
            for elem in dis:
                self.d.default(elem, 1/n)
        return None

    def problem(self, n):
        count = 0
        for bit in n:
            new = self.d[bit]
            count += new
        return count

    def normalize(self):
        sums = sum(self.d.values())
        for bit in self.d:
            self.d[bit] = self.d[bit] / sums

        return None

    def condition(self, n):
        liss = []
        for bit in self.d:
            if bit not in n:
                lis.append(bit)
        for key in lis:
            del self.d[key]
        self.normalize()

    def sample(self):
        dic = {}
        liss = []
        counter = 0
        counter_2 = 0
        for bit, val in self.d.items():
            lis.append([counter, counter + val])
            counter += val

        for bit in self.d:
            dic[bit] = liss[counter_2]
            counter_2 += 1

        for k, v in dic.items():
            if v[0] < np.random.uniform() < v[1]:
                return k



