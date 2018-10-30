#!/usr/bin/python3
import argparse
import time

import torch.optim as optim

from LoadDataVariables import *
from merge import mergeResult
from MNM import *

def prf(predicPath, goldPath):
    p = 0.0
    r = 0.0
    f = 0.0
    sampleCount = 0
    goldCount = 0
    ppCount = 0

    goldDict = {}
    with open(goldPath, 'r') as goldStream:
        for line in goldStream:
            items= line.strip().split("\t")
            if items[0] not in goldDict:
                goldDict[items[0]] = []
            goldDict[items[0]].append(items[1] + "\t" + items[2])
            goldCount += 1

    with open(predicPath, 'r') as predicStream:
        for line in predicStream:
            items = line.strip().split("\t")
            if items[0] not in goldDict:
                #print(items[0])
                #sampleCount += 1
                continue
            gold = goldDict[items[0]]
            pairStr0 = items[1] + "\t" + items[2]
            pairStr1 = items[2] + "\t" + items[1]
            
            if pairStr0 in gold:
                ppCount += 1
            elif pairStr1 in gold:
                ppCount += 1

            sampleCount += 1
    try:
        # print(sampleCount)
        p = float(ppCount) / float(sampleCount)
        r = float(ppCount) / float(goldCount)
        f = 2 * p * r / (p + r)
    except:
        return 0.0, 0.0, 0.0
    return p*100, r*100, f*100

def train(mnm, trainset, numEpoches, batchSize, paraPathPref='./parameters/model'):
    maxp = 0
    maxr = 0
    maxf = 0
    trainsetSize = len(trainset)
    oldf = 0.0
    optimizer = optim.Adam(mnm.parameters, weight_decay=0.1)
    test_idx = 0
    
    myRandom.shuffle(trainset)
    validSet = trainset[:200]
    trainset = trainset[200:]
    for epoch_idx in range(numEpoches):
        myRandom.shuffle(trainset)
        sum_loss= VariableDevice(torch.Tensor([0]), cuda)
        print("=====================================================================")
        print("epoch " + str(epoch_idx) + ", trainSize: " + str(trainsetSize))
        print("hop size: ", str(mnm.hopNumber))

        count = 0
        correct = 0
        time0 = time.time()
        optimizer.zero_grad()
        for sample in trainset:
            sentid, contxtWords, e1ID, e1, e2ID, e2, relation, sentLength, label, e1p, e2p = sample
            if sentLength == 0:
                continue
            finallinearLayerOut = mnm.forward(contxtWords, 
                                            e1, 
                                            e2,
                                            e1p,
                                            e2p,
                                            relation,
                                            sentLength)

            log_prob = F.log_softmax(finallinearLayerOut.view(1, classNumber))
            loss = loss_function(log_prob, VariableDevice(torch.LongTensor([label]), cuda))
            calssification = softmax(finallinearLayerOut.view(1, classNumber))

            sum_loss = torch.add(sum_loss, loss)

            predict = np.argmax(calssification.cpu().data.numpy())
            if predict == label:
                correct += 1.0
            count += 1

        ####################Update#######################
            if count % batchSize == 0:
                sum_loss.backward()
                optimizer.step()
                sum_loss = VariableDevice(torch.Tensor([0]), cuda)
                optimizer.zero_grad()

        sum_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ####################Update#######################

        time1 = time.time()
        print("Iteration", epoch_idx, "Loss", sum_loss.cpu().data.numpy()[0], "train Acc: ", float(correct / count) , "time: ", str(time1 - time0))
        
        currentValidResult = resultOutput + "result_valid_" + str(epoch_idx) + ".txt"
        mergedValidResult = currentValidResult + ".merged"
        resultValidStream = open(currentValidResult, 'w')

        test(mnm, validSet, resultValidStream)
        resultValidStream.close()
        mergeResult(currentValidResult, mergedValidResult)
        p, r, f = prf(mergedValidResult, "../data/trainGold.txt")
        print("valid P: {} R: {} F: {}".format(p, r, f))

        if f > oldf:
            oldf = f
            currentResult = resultOutput + "result_" + str(test_idx) + ".txt"
            mergedResult = currentResult + ".merged"
            resultStream = open(currentResult, 'w')
            probPath   = resultOutput + "prob_" + str(test_idx) + ".txt"
            test(mnm, testset, resultStream, probPath)
            resultStream.close()
            mergeResult(currentResult, mergedResult)
            p, r, f = prf(mergedResult, "../data/testGold.txt")
            print("test P: {} R: {} F: {}".format(p, r, f))
            test_idx += 1
        else:
            break
def test(mnm, testSet, resultStream=None, probPath=None):
    count = 0
    correct = 0
    time0 = time.time()
    probs = []
    for sample in testSet:
        sentid, contxtWords, e1ID, e1, e2ID, e2, relation, sentLength, label, e1p, e2p = sample

        finallinearLayerOut = mnm.forward(contxtWords, 
                                        e1,
                                        e2,
                                        e1p,
                                        e2p,
                                        relation,
                                        sentLength)
        calssification = softmax(finallinearLayerOut.view(1, classNumber))
            
        prob = calssification.cpu().data.numpy().reshape(classNumber)
        predict = np.argmax(prob)
        probs.append(prob)
        if resultStream and predict == 1:
            resultStream.write("\t".join([sentid, e1ID, e2ID]) + "\n")

        if predict == label:
            correct += 1.0
        count += 1
    if probPath:
        np.savetxt(probPath, probs, '%.5f',delimiter=' ')
    time1 = time.time()
    acc = correct/count
    # print("test Acc: ", acc)
    print("time    : ", str(time1 - time0))

def GetSampleProperty(sample):
    sentid = sample[0]
    words = sample[1].split(" ")
    e1Index = words.index("$1")
    e2Index = words.index("$2")
    e1p = []
    e2p = []
    for i in range(len(words)):
        if e1Index == i or e2Index == i:
            continue
        e1p.append(abs(e1Index - i))
        e2p.append(abs(e2Index - i))
    words = sample[1].replace("$1", " ").replace("$2", " ").replace("  ", "").split(" ")
    n = len(words)

    contxtWords = VariableDevice(torch.LongTensor([word2id[words[i]] for i in range(0, n)]), cuda)

    e1ID = sample[2]
    e1   = "-1" 
    if e1ID in entity2id:
        e1 = entity2id[e1ID]
        e1V = VariableDevice(torch.LongTensor([int(e1)]), cuda)
        e1V = entityEmbed(e1V).transpose(0,1)
    else:
        entityWords = sample[4].split(" ")
        e1V = torch.sum(wordEmbed(VariableDevice(torch.LongTensor([word2id[entityWord] for entityWord in entityWords]), cuda)), 0) / len(entityWords)
        e1V = e1V.view(wordVectorLength, 1)

    e2ID = sample[3]
    e2   = "-1"
    if e2ID in entity2id:
        e2 = entity2id[e2ID]
        e2V = VariableDevice(torch.LongTensor([int(e2)]), cuda)
        e2V = entityEmbed(e2V).transpose(0,1)
    else:
        entityWords = sample[5].split(" ")
        e2V = torch.sum(wordEmbed(VariableDevice(torch.LongTensor([word2id[entityWord] for entityWord in entityWords]), cuda)), 0) / len(entityWords)
        e2V = e2V.view(wordVectorLength, 1)

    pairStr0 = str(e1) + "_" + str(e2)
    pairStr1 = str(e2) + "_" + str(e1)
    if pairStr0 in triples:
        relation = relation2vector[triples[pairStr0]]
    elif pairStr1 in triples:
        relation = relation2vector[triples[pairStr1]]
    else:
        relation = relation2vector['unknow']
    
    label = int(sample[-1])

    return sentid, contxtWords, e1ID, e1V, e2ID, e2V, relation, n, label, e1p, e2p


cuda = torch.cuda.is_available()
softmax = torch.nn.Softmax()
loss_function = torch.nn.NLLLoss()

parser = argparse.ArgumentParser()
parser.add_argument("--trainPath", default="../data/train.txt")
parser.add_argument("--testPath", default="../data/test.txt")
parser.add_argument("--batchSize", default=100, type=int)
parser.add_argument("--wd", default=100, type=int)
parser.add_argument("--ed", default=100, type=int)
parser.add_argument("--hop", default=4, type=int)
parser.add_argument("--clas", default=2, type=int)
parser.add_argument("--epoch", default=20, type=int)
parser.add_argument("--wePath", default="../data/wordEmb/bio-word2id100")
parser.add_argument("--w2IDPath", default="../data/wordEmb/bio-embed100")
parser.add_argument("--eePath", default="../data/KB/entity2vec.vec")
parser.add_argument("--rePath", default="../data/KB/relation2vec.vec")
parser.add_argument("--t2idPath", default="../data/KB/triple2id.txt")
parser.add_argument("--e2idPath", default="../data/KB/entity2id.txt")
parser.add_argument("--paraPath", default="./parameters/")
parser.add_argument("--results", default="./results/")
args = parser.parse_args()

batchSize           = args.batchSize
wordVectorLength    = args.wd
entityVecSize       = args.ed
hopNumber           = args.hop
classNumber         = args.clas
numEpoches          = args.epoch


print("Loading word id...")
word2id = LoadWord2id(args.wePath)
print("Loading word vectors...")
ten = np.loadtxt(args.w2IDPath)
wordEmbed = nn.Embedding(ten.shape[0], wordVectorLength)
wordEmbed.weight = ParameterDevice(torch.FloatTensor(ten), cuda)

print("Loading entity vectors...")
entity2vector = np.loadtxt(args.eePath)
entityEmbed = nn.Embedding(entity2vector.shape[0], entity2vector.shape[1])
entityEmbed.weight = ParameterDevice(torch.FloatTensor(entity2vector), cuda)

print("Loading relation vectors...")
relation2vector = LoadRelationVectors(args.rePath, entityVecSize, cuda)
print("Loading triples...")
triples = LoadTriples(args.t2idPath)
print("Loading entity id mapping...")
entity2id = LoadEntity2Id(args.e2idPath)

paraPathPref = args.paraPath
resultOutput = args.results
if not os.path.exists(resultOutput):
    os.makedirs(resultOutput)
if not os.path.exists(paraPathPref):
    os.makedirs(paraPathPref)

trainPath = args.trainPath
testsPath = args.testPath
print("Load training samples...")
trainSet = LoadSamples(trainPath)
print("Load test samples...")
testSet = LoadSamples(testsPath)

trainset = []
for sample in trainSet:
    sampleTuple = GetSampleProperty(sample)
    trainset.append(sampleTuple)

testset = []
for sample in testSet:
    sampleTuple = GetSampleProperty(sample)
    testset.append(sampleTuple)

mnm = KBMemory(wordEmbed, entityEmbed, wordVectorLength, hopNumber, classNumber, cuda)
train(mnm, trainset, numEpoches, batchSize, paraPathPref)
