'''
Name: Aaron Elledge
Date: 3/4/2020
Class: ISTA 311
Collaborators: Warren Hill
'''


import numpy as np
from distribution import Distribution



class InferenceSuite(Distribution):

    def update(self, data):
        for hypothesis in self.d:
            self.d[hypothesis] *= self.likelihood(data, hypothesis)
        self.normalize()
         

    '''
    The method map should return the hypothesis associated to the 
    highest probability.If there is a tie, return the first outcome that appears 
    in the dictionary

    '''        

    def map(self):
        max_p = 0
        max_x = 0
        for x in self.d:
            if self.d[x] > max_p:
                max_x = x
                max_p = self.d[x]
        return max_x
    '''
    The method should return the hypothesis associated to the highest
    probabilty distribution
    '''
    def mean(self):
        return sum([x * self.d[x] for x in self.d])
    '''
    The method should take a single parameter representing a probability and
    return the smallest hypothesis.

    '''
    def quantile(self, p):
        totalp = 0
        for x in sorted(self.d):
            totalp += self.d[x]
            if totalp >= p:
                return x

    def likelihood(self, data, hypothesis):
        raise NotImplementedError

class Mayor(InferenceSuite):

    def __init__(self):
        self.d = {'A':0.25, 'B':0.35, 'C':0.40}

    '''
    This will update data with True or False will upadte the dictionary 
    self.d to correct probabilities.
    
    '''    

    def likelihood(self, data, hypothesis):
        bridge_built = data                     
        if hypothesis == 'A':
            if bridge_built:
                return 0.6
            else:
                return 0.4
        if hypothesis == 'B':
            if bridge_built:
                return 0.9
            else:
                return 0.1
        if hypothesis == 'C':
            if bridge_built:
                return 0.8
            else:
                return 0.2

class Diagnostic(InferenceSuite):


    '''
    This method shows the result of a test for the disease, showing
    either '+' or '-'. If the patient has the disease the test will show '+' with probability .9 and '-'
    with probability .1. If the patient doesn't have the disease the test result will be '+' with probability .05 and '-' with probability 
    .95

    '''
    def likelihood(self, data, hypothesis):
        if hypothesis == 'healthy':
            if data == '+':
                return 0.05
            else:
                return .95
        if hypothesis == 'sick':
            if data == '+':
                return 0.9
            else:
                return 0.1

class Cookie(InferenceSuite):
    '''
    This method will take two cookie flavor distributions in each bowl
    '''
    def __init__(self, bowl1, bowl2):
        self.d = {1: 0.5, 2:0.5}
        self.bowl1probs = dict(zip(['c','v','s'], bowl1))
        self.bowl2probs = dict(zip(['c','v','s'], bowl2))
    '''
    This method returns the probability of drawing the given type
    of cookie, with the assumption we are drawing from the given bowl
    '''
    def likelihood(self, data, hypothesis):
        if hypothesis == 1:
            return self.bowl1probs[data]
        if hypothesis == 2:
            return self.bowl2probs[data]

class Locomotive(InferenceSuite):
    '''

    This method observes the number of observed train and the hypothesized
    number of total trains
    '''
    

    def likelihood(self, data, hypothesis):
        if hypothesis < data:
            return 0.0
        else:
            return 1.0 / hypothesis

def main():
    '''
    Locomotive()
    '''


if __name__ == '__main__':

