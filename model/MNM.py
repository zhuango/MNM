#!/usr/bin/python3

import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# for reproduction
torch.manual_seed(2018)
myRandom = np.random.RandomState(2018)

class KBMemory():
    def __init__(self, wordEmbed, entityEmbed, wordVectorLength, hopNumber, classNumber, cuda=True):
        self.wordEmbed = wordEmbed
        self.entityEmbed = entityEmbed
        self.wordVectorLength = wordVectorLength
        self.vectorLength = wordVectorLength
        self.hopNumber = hopNumber
        self.classNumber = classNumber
        self.cuda = cuda
        self.update = 5000
        self.atten_Ws = []
        self.atten_bs = []
        for i in range(self.hopNumber):
            self.atten_Ws.append(VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (1, 2 * self.vectorLength))), cuda, requires_grad=True))
            self.atten_bs.append(VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, 1)), cuda, requires_grad=True))

        self.linear_W = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (self.vectorLength, self.vectorLength))), cuda, requires_grad=True)
        self.linear_b = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (self.vectorLength, 1))), cuda, requires_grad=True)

        self.softmax_W = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, self.vectorLength*2 + wordVectorLength))), cuda, requires_grad=True)
        self.softmax_b = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, 1))), cuda, requires_grad=True)
        
        self.softmax = torch.nn.Softmax()
        
        self.parameters = [self.linear_W, self.linear_b, self.softmax_W, self.softmax_b]
        self.parameters += self.atten_Ws
        self.parameters += self.atten_bs

    def forward(self, contxtWords, e1, e2, e1p, e2p, relation, sentLength):
        linear_W = self.linear_W
        linear_b = self.linear_b
        softmax_W = self.softmax_W
        softmax_b = self.softmax_b
        vectorLength = self.vectorLength
        hopNumber = self.hopNumber

        positionsE1 = VariableDevice(torch.FloatTensor(np.array(e1p, dtype=np.float)), self.cuda).expand(vectorLength, sentLength)
        positionsE2 = VariableDevice(torch.FloatTensor(np.array(e2p, dtype=np.float)), self.cuda).expand(vectorLength, sentLength)
        contxtWords = self.wordEmbed(contxtWords).transpose(0,1)
        #=====================================================================
        positions = positionsE1
        for i in range(hopNumber):
            Vi = 1.0 - positions / sentLength - (i / vectorLength) * (1.0 - 2.0 * (positions / sentLength))
            Mi = Vi * contxtWords
            atten_W = self.atten_Ws[i]
            atten_b = self.atten_bs[i]

            attentionInputs = torch.cat([Mi, e1.expand(vectorLength, sentLength)])
            attentionA = torch.mm(atten_W, attentionInputs) + atten_b.expand(1, sentLength)

            linearLayerOut = torch.mm(linear_W, e1) + linear_b
            e1 = torch.mm(Mi, self.softmax(torch.tanh(attentionA)).transpose(0, 1)) + linearLayerOut

        #=====================================================================
        for i in range(hopNumber):
            Vi = 1.0 - positionsE2 / sentLength - (i / vectorLength) * (1.0 - 2.0 * (positionsE2 / sentLength))
            Mi = Vi * contxtWords
            atten_W = self.atten_Ws[i]
            atten_b = self.atten_bs[i]

            attentionInputs = torch.cat([Mi, e2.expand(vectorLength, sentLength)])
            attentionA = torch.mm(atten_W, attentionInputs) + atten_b.expand(1, sentLength)

            linearLayerOut = torch.mm(linear_W, e2) + linear_b
            e2 = torch.mm(Mi, self.softmax(torch.tanh(attentionA)).transpose(0, 1)) + linearLayerOut

        finalOutput = torch.cat([e1, e2, relation])
        finallinearLayerOut = torch.mm(softmax_W, finalOutput) + softmax_b
        return finallinearLayerOut


def VariableDevice(data, cuda=True, requires_grad=False):
    if cuda:
        return Variable(data.cuda(), requires_grad=requires_grad)
    else:
        return Variable(data, requires_grad=requires_grad)

def ParameterDevice(data, cuda=True, requires_grad=False):
    if cuda:
        return torch.nn.Parameter(data.cuda(), requires_grad=requires_grad)
    else:
        return torch.nn.Parameter(data, requires_grad=requires_grad)
