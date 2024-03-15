'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    DRxy = 0
    for i in range(len(Rx)):
        DRxy += abs(Rx[i - 1] - Ry[i - 1])
    DRxy = (1 / len(Rx)) * DRxy
    return DRxy


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    lxx = sum((theta - (1 / len(Thetax)) * sum(Thetax)) ** 2 for theta in Thetax)
    lyy = sum((theta - (1 / len(Thetax)) * sum(Thetay)) ** 2 for theta in Thetay)
    lxy = sum((Thetax[i] - (1 / len(Thetax)) * sum(Thetax)) * (Thetay[i] - len(Thetax) * sum(Thetay)) for i in range(len(Thetax)))
    Dθxy = (1 - (lxy * lxy) / (lxx * lyy)) * 100
    return Dθxy
    pass
