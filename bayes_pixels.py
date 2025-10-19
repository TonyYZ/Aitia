# -*- coding: utf-8 -*-
import copy
import math


import networkx as nx
import matplotlib.pyplot as plt
import compute
import numpy as np
import random
import Levenshtein


class Node(object):
    def __init__(self, label):
        self.label = label
        self.children = list()

    def addkid(self, node):
        self.children.append(node)
        return self


leafSymbols = compute.leafSymbols
nodeSymbols = compute.nodeSymbols
replaceDict = compute.replaceDict
print('Leaf Symbols', leafSymbols)
samplePercentage = 1
prunePercentage = 1 # pursued tree completeness
global axes

def computeJaccardSimilarity(graph1, graph2):
    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())

    intersection = nodes1.intersection(nodes2)
    union = nodes1.union(nodes2)

    similarity = len(intersection) / len(union)
    return similarity


def sortGuideTree(guideTree):
    # freq.append(len(guideTree[1:]))
    if guideTree[0] == 'parallel':
        guideTree[1:] = sorted(guideTree[1:])
    if guideTree[0] in ['parallel', 'series']:
        for branch in guideTree[1:]:
            # freq = sortGuideTree(branch, freq=freq)
            sortGuideTree(branch)
    # return freq


def convert2NodeTree(guideTree, labelCounter=None, graph=None, prevLabel=None):
    if labelCounter is None:
        labelCounter = {}

    if isinstance(guideTree, int):  # ignore body code
        return
    if guideTree[0] in nodeSymbols:
        nodeTree = Node(guideTree[0])
        node = guideTree[0]
        # print(node)
        if graph is not None:
            uniqueLabel = getUniqueLabel(node, labelCounter)
            graph.add_node(uniqueLabel, label=node)
        else:
            uniqueLabel = None
        if prevLabel is not None and graph is not None:
            graph.add_edge(prevLabel, uniqueLabel)
        for branch in guideTree[1:]:
            result = convert2NodeTree(branch, labelCounter, graph, uniqueLabel)
            if result is not None:
                nodeTree.addkid(result)
            # print(nodeTree.children)
    else:
        nodeTree = Node(guideTree)
        elemName = guideTree
        if graph is not None:
            uniqueLabel = getUniqueLabel(elemName, labelCounter)
            graph.add_node(uniqueLabel, label=elemName)
        if prevLabel is not None and graph is not None:
            graph.add_edge(prevLabel, uniqueLabel)
    return nodeTree


def convert2GuideTree(nodeTree, labelCounter=None, graph=None, prevLabel=None):
    if labelCounter is None:
        labelCounter = {}
    if nodeTree.label in nodeSymbols:
        if graph is not None:
            uniqueLabel = getUniqueLabel(nodeTree.label, labelCounter)
            graph.add_node(uniqueLabel, label=nodeTree.label)
        else:
            uniqueLabel = None
        if prevLabel is not None and graph is not None:
            graph.add_edge(prevLabel, uniqueLabel)
        guideTree = [nodeTree.label]
        guideTree += [convert2GuideTree(child, labelCounter, graph, uniqueLabel) for child in nodeTree.children]
    else:
        guideTree = nodeTree.label
        if graph is not None:
            uniqueLabel = getUniqueLabel(nodeTree.label, labelCounter)
            graph.add_node(uniqueLabel, label=nodeTree.label)
        if prevLabel is not None and graph is not None:
            graph.add_edge(prevLabel, uniqueLabel)
    return guideTree


def getUniqueLabel(label, labelCounter):
    if isinstance(label, list):
        label = compute.lst2tup(label)
    if label in labelCounter:
        labelCounter[label] += 1
        return f"{label}{labelCounter[label]}"
    else:
        labelCounter[label] = 1
        return label


def countTree(tree):
    global nodeSymbols
    num = 1
    if tree[0] in nodeSymbols:
        for child in tree[1:]:
            num += countTree(child)
    return num


def countNode(node):
    global nodeSymbols
    num = 1
    if node.label in nodeSymbols:
        for child in node.children:
            num += countNode(child)
    return num


def generateRandomTree(tree, maxDepth):
    global leafSymbols
    global nodeSymbols
    # print(maxDepth)
    if (maxDepth <= 0 or random.random() < 0.5) and tree[0] == nodeSymbols[1]:
        tree.append([random.choice(leafSymbols), random.choice([0, 1]),
                     random.randint([-2, 1])])  # random lowercase letter
        return
    else:
        numBranches = random.randint(1, 4)  # random number of branches
        if len(tree[1:]) < numBranches:
            for i in range(numBranches - len(tree[1:])):
                if tree[0] == nodeSymbols[0]:
                    tree.append([nodeSymbols[1]])
                else:
                    tree.append([random.choice(nodeSymbols)])
        elif len(tree[1:]) > numBranches:
            indices2Remove = np.random.choice(len(tree[1:]), size=len(tree[1:]) - numBranches,
                                              replace=False)
            tree[1:] = [child for i, child in enumerate(tree[1:]) if i not in indices2Remove]

        for child in tree[1:]:
            if child[0] in nodeSymbols:
                generateRandomTree(child, maxDepth - 1)


def getRandomLayer():
    # shot = random.choice([2] * 10 + [1] * 25 + [0] * 30 + [-1] * 25 + [-2] * 10)
    shot = random.choice([0] * 30 + [-1] * 70)
    return shot


def getRandomBranch(maxDepth):  # random distribution based on frequency statistics
    # shot = random.choice([1] * 30 + [2] * 55 + [3] * 15)
    shot = random.choice([1] * 50 + [2] * 5 + [3] * 30 + [4] * 5 + [5] * 10)
    # shot = random.randint(1, int(5 * math.log(maxDepth + 1)))
    # shot = round(random.random() * 7.5 * math.exp(-1.5 * (0.7 * maxDepth - 2) ** 2) + 0.5)
    # available = {4: [2], 3: [2], 2: [1, 8], 1: [1], 0: [0]}
    # shot = random.choice(available[maxDepth])
    return shot


def generateRandomNode(node, maxDepth, index=-1):  # max depth does not include the top layer
    global leafSymbols
    global nodeSymbols

    # terminate tree if exceeds depth or create a element if randomly agrees at a new series node
    if (maxDepth <= 0 or random.random() < 0.8
        and node.label != nodeSymbols[0]) \
            and node.children == []:
        # print("add one")
        node.addkid(Node(
            [random.choice(leafSymbols), random.choice([0, 1]), getRandomLayer()]))  # random lowercase letter
    else:  # expand/shrink tree
        if maxDepth == 2 and False:
            if index == 0:
                numBranches = 1
            if index == 1:
                numBranches = 8
        else:
            numBranches = getRandomBranch(maxDepth)
        # print('len comparison', len(node.children), numBranches)
        if len(node.children) < numBranches:
            for i in range(numBranches - len(node.children)):
                if node.label == nodeSymbols[0]:
                    node.addkid(Node(nodeSymbols[1]))
                else:
                    if maxDepth - 1 <= 0:  # prevent parallel at the penultimate
                        node.addkid(Node([random.choice(leafSymbols), random.choice([0, 1]), getRandomLayer()]))
                    else:
                        node.addkid(Node(nodeSymbols[0]))
        elif len(node.children) > numBranches:
            indices2Remove = np.random.choice(len(node.children), size=len(node.children) - numBranches,
                                              replace=False)
            node.children = [child for i, child in enumerate(node.children) if i not in indices2Remove]

        for i, child in enumerate(node.children):
            if child.label not in nodeSymbols:  # the child is an element
                if random.random() < 0.5:  # alter element pattern
                    currentPattern = list(compute.charMapAll[child.label[0]])
                    # print('current pattern', currentPattern, child.label)
                    shot = random.randint(0, 12)
                    if shot <= 7:
                        currentPattern = [random.choice([0, 1]) for i in range(3)]
                    elif shot == 8:
                        currentPattern = [0.5, 0.5, 0.5]  # accidentally falls into middle zone
                    elif shot >= 9:
                        currentPattern = [random.choice([0, 1]) for i in range(2)]

                    isDynamic = child.label[0] in compute.reversedCharMapDyn
                    if len(currentPattern) == 2:
                        isDynamic = False
                    elif random.random() < 0.6:  # flip the dynamic nature
                        isDynamic = 1 - isDynamic

                    newChar = compute.reversedCharMapDyn[tuple(currentPattern)] if isDynamic else compute.charMap[
                        tuple(currentPattern)]
                    # print('new char', newChar)
                    node.children[i] = Node([newChar, child.label[1], child.label[2]])
                if random.random() < 0.3:  # alter element dimension
                    node.children[i] = Node([child.label[0], 1 - child.label[1], child.label[2]])
                if random.random() < 0.3:  # alter element layer
                    node.children[i] = Node([child.label[0], child.label[1], getRandomLayer()])
            else:
                generateRandomNode(child, maxDepth - 1, i)

def scaleUpNearest(mat, target_shape):
    """
    Scales up a 2D NumPy array by repeating its values (nearest-neighbor style)
    to match a desired shape.

    Parameters
    ----------
    mat : np.ndarray
        The small 2D array to scale up.
    target_shape : tuple of int
        The desired (rows, cols) of the output array.

    Returns
    -------
    scaled : np.ndarray
        The scaled-up array, same shape as `target_shape`.
    """
    rows, cols = mat.shape
    target_rows, target_cols = target_shape

    # Compute integer scale factors
    scale_row = target_rows // rows
    scale_col = target_cols // cols

    # Use Kronecker product to repeat values
    scaled = np.kron(mat, np.ones((scale_row, scale_col)))

    # Crop in case target size isn’t an exact multiple
    return scaled[:target_rows, :target_cols]

def evaluateTree(currentGuideTree, sourceTrials, verbose=0, showZero=True, fillDyn=False):
    # sortGuideTree(currentGuideTree)
    currentTree = convert2NodeTree(currentGuideTree)
    candidateTree = currentTree

    if verbose > 0:
        print("Current tree", currentGuideTree)


    videos, layerLst, zeroLayerTrees = compute.compLayers(currentGuideTree,keyLen=len(sourceTrials[0]), verbose=verbose > 2, fillDyn=fillDyn)

    print("test tree", zeroLayerTrees, layerLst, len(videos[0]))
    # handwritten: no reverse, generated: needs reverse
    if showZero:
        zeroGraphs = []
        for _ in range(len(zeroLayerTrees)):
            zeroGraphs.append(nx.DiGraph())
        testNodeCol = [convert2NodeTree(testTree, graph=zeroGraphs[i]) for i, testTree in enumerate(zeroLayerTrees)]
    else:
        zeroGraphs = None
        testNodeCol = [convert2NodeTree(testTree) for i, testTree in enumerate(zeroLayerTrees)]

    nodeNumAvg = []
    for zero in testNodeCol:
        nodeNumAvg.append(countNode(zero))
    nodeNumAvg = sum(nodeNumAvg) / len(nodeNumAvg)
    # candidatePrior = 2 - 2 / (1 + pow(math.e, -countNode(candidateTree) / 10))
    priorK = 15
    candidatePrior = priorK / nodeNumAvg if nodeNumAvg >= priorK else 1

    # compare each pixel between the two videos
    similarityLst = []
    if 0 in layerLst:
        comparedVideo = videos[layerLst.index(0)][::compute.tweening]
    else:
        comparedVideo = videos[0][::compute.tweening]

    for sourceSeq in sourceTrials:
        for i, sourceFrame in enumerate(sourceSeq):
            print('simliar lst', scaleUpNearest(sourceFrame, comparedVideo[0].shape) - comparedVideo[i])
            similarityLst.append(np.mean(255 - np.abs(scaleUpNearest(sourceFrame, comparedVideo[0].shape) - comparedVideo[i])))
    candidateLikelihood = sum(similarityLst) / len(similarityLst)
    print('Node total', countNode(candidateTree), 'prior', candidatePrior, '* likelihood', candidateLikelihood, '=',
          candidatePrior * candidateLikelihood)
    return candidatePrior, candidateLikelihood, candidateTree, videos, layerLst, zeroGraphs


def treeSubstitutionMCMC(sourceTrials, numIterations, verbose=0, showTree=True, showImage=True,
                         replaceText=False, showZero=True, useSymbols=False,
                         fillDyn=True, mirrorMode=False):
    bestTree = None
    bestLikelihood = 0.0
    currentLikelihood = 0.0
    currentPrior = 0.0
    currentTree = Node('parallel')
    successIter = 0

    for iteration in range(numIterations):
        if verbose > 0:
            print(f"Iteration: {iteration + 1}/{numIterations}, Accuracy: {bestLikelihood}")
        generateRandomNode(currentTree, 4)
        graph = nx.DiGraph()
        labelCounter = {}
        currentGuideTree = convert2GuideTree(currentTree, labelCounter=labelCounter, graph=graph)
        candidatePrior, candidateLikelihood, candidateTree, videos, layerLst, zeroGraphs = evaluateTree(currentGuideTree, sourceTrials, verbose=verbose, showZero=showZero, fillDyn=fillDyn)
        print('abc', candidateLikelihood, candidatePrior)
        if currentLikelihood * currentPrior == 0:
            acceptanceProb = min(1.0, candidateLikelihood * candidatePrior)
        else:
            acceptanceProb = min(1.0, (candidateLikelihood * candidatePrior) / (currentLikelihood * currentPrior))
        if verbose > 0:
            print("Acceptance Prob:", acceptanceProb)

        if random.random() < acceptanceProb:
            currentTree = candidateTree
            currentPrior = candidatePrior
            currentLikelihood = candidateLikelihood

        if candidateLikelihood > bestLikelihood:
            bestTree = copy.deepcopy(candidateTree)
            bestLikelihood = candidateLikelihood
            successIter = iteration
            if verbose > 0:
                bestGuideTree = currentGuideTree
                print("Likelihood:", bestLikelihood, "Current Best Tree:", bestGuideTree)
                # if iteration >= burnInIterations and verbose:
                if not showTree:
                    graph = None
                if not showTree:
                    zeroGraphs = None
                if not showImage:
                    videos = None
                showPlot(graph=graph, replaceText=replaceText, videos=videos, layerLst=layerLst, zeroGraphs=zeroGraphs, useSymbols=useSymbols, mirrorMode=mirrorMode)

    return bestTree, successIter, bestLikelihood


def showPlot(graph=None, replaceText=True, videos=None, layerLst=None, zeroGraphs=None, isSource=False, useSymbols=False, mirrorMode=False):
    showMode = [videos is not None, zeroGraphs is not None, graph is not None]
    if not any(showMode):
        return None
    if not isSource:
        if videos is not None:
            compute.playLayers(videos, layerLst, showMode=showMode, isSource=isSource, mirrorMode=mirrorMode)
            axes = compute.axes
        else:
            fig = compute.initAxes(showMode=showMode, numAxes=1)
            axes = compute.axes

    if isSource:
        if showMode[0] and showMode[2]:
            startAx = 2
        elif showMode[0] and not showMode[2]:
            startAx = -1
        elif not showMode[0] or not showMode[2]:
            startAx = 0
        compute.initButtons(graph, startAx, replaceText=replaceText, videos=videos, useSymbols=useSymbols,
                           isSource=isSource)
        compute.updateGallery(0, graph, startAx, replaceText=replaceText, videos=videos, useSymbols=useSymbols)
    else:
        if showMode[1] and showMode[2]:
            startAx = -4
        elif showMode[1] and not showMode[2]:
            startAx = -3
        elif not showMode[1] or not showMode[2]:
            startAx = -1
        for i, curMode in enumerate(showMode[1:]):  # applies to both zero layer graphs and complete graph
            if not curMode:
                continue
            if i == 0:
                compute.initButtons(zeroGraphs, startAx, replaceText=replaceText, useSymbols=useSymbols)
                compute.updateGallery(0, zeroGraphs, startAx, replaceText=replaceText, useSymbols=useSymbols)
                # title = 'Zero Layer 1/' + str(len(zeroGraphs))
            elif i == 1:
                compute.drawNetwork(graph, startAx, replaceText, useSymbols)
                title = 'Guide Tree'
                axes[startAx].set_title(title)
            startAx += 3
    plt.show(block=True)


def getLabelSimilarity(label1, label2):
    isDynamic = [label[0] in compute.reversedCharMapDyn for label in [label1, label2]]
    #if isDynamic[0] != isDynamic[1] or label1[1] != label2[1] or label1[2] != label2[2]:
    #    return 0

    staticDifSum = 0
    staticPatterns = [None, None]
    dynamicDifSum = 0
    dynamicPatterns = [None, None]
    dynamicAbsDifSum = 0
    dynamicAbsPatterns = [None, None]
    for i, label in enumerate([label1, label2]):
        staticPatterns[i] = compute.charMapAll[label[0]]
        if len(staticPatterns[i]) == 2:
            staticPatterns[i] = [staticPatterns[i][0], sum(staticPatterns[i]) / 2, staticPatterns[i][1]]
        if isDynamic[i]:
            dynamicPatterns[i] = compute.vecMap[compute.reversedHexMap[compute.reversedCharMapDyn[label[0]]]][0]
            dynamicAbsPatterns[i] = compute.vecMap[compute.reversedHexMap[compute.reversedCharMapDyn[label[0]]]][1]
        else:
            dynamicPatterns[i] = [0, 0]
            dynamicAbsPatterns[i] = [0, 0]

    for i in range(3):
        staticDifSum += abs(staticPatterns[0][i] - staticPatterns[1][i])
    staticDist = staticDifSum / 3
    for i in range(2):
        dynamicDifSum += abs(dynamicPatterns[0][i] - dynamicPatterns[1][i])
        dynamicAbsDifSum += abs(dynamicAbsPatterns[0][i] - dynamicAbsPatterns[1][i])
    dynamicDist = (dynamicDifSum + dynamicAbsDifSum) / 4
    if not isDynamic[0] and not isDynamic[1]:
        patternDistSum = staticDist
    elif isDynamic[0] and isDynamic[1]:
        patternDistSum = dynamicDist
    else:
        patternDistSum = staticDist * 0.05 + dynamicDist * 0.95
        # print(label1, label2, staticDist, dynamicDist, patternDistSum)
        # print(dynamicPatterns, dynamicAbsPatterns)

    '''
    dist = patternDistSum * 0.7\
                 + (abs(int(isDynamic[0]) - int(isDynamic[1])) * 0.5 +
                    abs(label1[1] - label2[1]) * 0.5) * 0.3
    '''

    dist = patternDistSum if label1[1] == label2[1] else 1
    dist += int(label2[0] in ['ż', 'm', 'ċ']) * 0.15 + int(label1[0] in ['ż', 'm', 'ċ']) * 0.15
    #if not (label1[0] in ['ż', 'ṡ', 'm', 'n', 'ċ', 'ġ'] or label2[0] in ['ż', 'm', 'ċ']):
    #    dist += 0.1 * abs(label1[1] - label2[1])

    #if label2[0] in ['ż', 'm', 'ċ'] and label1[0] not in ['ż', 'm', 'ċ']:  # punishment for unknown tree
    #    dist += 0.3
    if dist > 1:
        dist = 1
    # dist = 0.4 * staticDifSum / 3 + 0.3 * abs(int(label1[0] in compute.reversedCharMap) - int(label2[0] in compute.reversedCharMap)) + 0.3 * abs(label1[1] - label2[1])
    # print('dist between', label1, 'and', str(label2) + ':', dist)
    # print('static', staticPatterns, staticDifSum, 'dynamic', dynamicPatterns, dynamicDifSum, 'abs', dynamicAbsPatterns, dynamicAbsDifSum)
    return 1 - dist


def flattenList(lst):
    stack = lst[::-1]  # Reverse the list to simulate a stack
    result = []
    if isinstance(lst, tuple):
        return [lst]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item[::-1])  # Reverse and add nested list to stack
        else:
            result.append(item)

    return result


def extractNodes(node, depth=0, needTotal=False, needDepth=False):
    if node.label not in nodeSymbols:
        if needTotal:
            total = 1
        else:
            total = None
        if not needDepth:
            depth = None
            return node, total, depth
    else:
        result = []
        for child in node.children:
            extracted = extractNodes(child, depth=depth + 1, needTotal=needTotal, needDepth=needDepth)
            if extracted is not None:
                result.append(extracted)
        if needTotal:
            total = 1
            for branch in result:
                if not isinstance(branch, list):
                    total += branch[1]
                else:
                    total += branch[0][1]
        else:
            total = None
        if not needDepth:
            depth = None
        result = [(node, total, depth), result]
        return result


def pruneTree(tree, needPrune=True):
    global prunePercentage
    total = 1
    newTree = Node(tree.label)
    for child in tree.children:
        if not needPrune or random.random() < prunePercentage:
            resultTup = pruneTree(child)
            newTree.addkid(resultTup[0])
            total += resultTup[1]
    return newTree, total


def checkEquivalence(tree1, tree2, verbose=False):
    if tree1.label is None and tree2.label is None:
        return 1
    if None in [tree1.label, tree2.label]:
        return 0.5
    if tree1.label != tree2.label and not (isinstance(tree1.label, list) and isinstance(tree2.label, list)):
        # print("unequal label", tree1.label, tree2.label)
        return 0
    elif tree1.label in nodeSymbols:
        if tree1.children == tree2.children == []:
            return 1
        treeTup = [tree1, tree2]
        pad = len(tree1.children) - len(tree2.children)

        treeTup[int(math.copysign(1, pad) / 2 + 0.5)].children += [Node(None) for i in range(abs(pad))]
        # print("tree tup", [ch.label for ch in treeTup[0].children], [ch.label for ch in treeTup[1].children])
        availableChoices = list(range(len(tree1.children)))
        if tree1.label in nodeSymbols[1]:
            seq1 = ''.join([str(val) for val in availableChoices])
            seq2 = ''
        score = 0
        if tree1.label in nodeSymbols[0]:
            dominantSingles = [-1, -1]  # the orientation that the dominant single node determines
        for child1 in tree1.children:
            comparison = []
            if tree1.label in nodeSymbols[0] and\
                dominantSingles[0] == -1 and child1.label == 'series' and len(child1.children) == 1:
                # print("ddiscovered 1", child1.children[0])
                dominantSingles[0] = child1.children[0].label[1]
            for child2 in tree2.children:
                if tree1.label in nodeSymbols[0] and\
                        dominantSingles[1] == -1 and child2.label == 'series' and len(child2.children) == 1:
                    # print("ddiscovered 2", child2.children[0])
                    dominantSingles[1] = child2.children[0].label[1]
                if verbose > 0:
                    print("comparing", child1.label, child2.label)
                comparison.append(checkEquivalence(child1, child2, verbose=verbose))
            if verbose:
                print('comparison', comparison)
            while True:
                choice = np.argmax(comparison)
                if choice in availableChoices:
                    break
                else:
                    comparison[choice] = -1
            availableChoices.remove(choice)
            if tree1.label in nodeSymbols[1]:
                seq2 += str(choice)
            score += np.max(comparison)
        if tree1.label in nodeSymbols[0]:  # parallel
            # if the orientation is different, everything is different
            punishmentFactor = 0 if dominantSingles[0] != dominantSingles[1] else 1
            score = score / len(tree1.children) * punishmentFactor
        else:  # series
            score = score / len(tree1.children) * Levenshtein.ratio(seq1, seq2)
        if verbose:
            print('score', score, convert2GuideTree(tree1))
            if tree1.label in nodeSymbols[1]:
                print('sequences', seq1, seq2)
            print(convert2GuideTree(tree2))
        return score
    else:
        if verbose:
            print(tree1.label, tree2.label)
        labelSimilarity = getLabelSimilarity(tree1.label, tree2.label)
        if verbose:
            print("similarity", labelSimilarity)
        return labelSimilarity

def main():
    useTotal = False
    useDepth = not useTotal

    sourceTrials = [[np.array([[255, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [255, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0], [255, 0, 0]])],
                    [np.array([[0, 255, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0], [0, 255, 0]])],
                    [np.array([[0, 0, 255], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 255], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 255]])]]
    learnedTree, successIter, bestLikelihood = \
        treeSubstitutionMCMC(sourceTrials, numIterations=2, verbose=1, showTree=True,
                             showImage=True,
                             replaceText=True, showZero=True,
                             useSymbols=False, mirrorMode=True, fillDyn=False)
    # verbose: 1 = basic, 2 = label comparison, 3 = layer calculations
    print("Learned Tree: ", convert2GuideTree(learnedTree), "Best Likelihood", bestLikelihood,
          "Successful Iterations", successIter)


if __name__ == "__main__":
    main()
