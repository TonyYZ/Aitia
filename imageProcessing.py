import math
import sys
from random import random

import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from matplotlib import animation

# sys.path.append('D:/Project/Ete/ReEte')
sys.path.append('/home/ukiriz/Ete')
import copy
from line_profiler import LineProfiler
from matplotlib.widgets import Button
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib.image import imread
import demoBackups
import ItiParser
import imageio

# fontNames = ['Yu Gothic', 'Segoe UI Symbol']
fontNames = ['Noto Sans CJK JP', 'Noto Sans CJK JP']

profiler = LineProfiler()

depthMax = 1
punishment = 5  # punishment factor to simple hexagrams (0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)
punishmentDyn = 2
hexMap = {'坤卦': (0, 0, 0), '艮卦2': (1, 0, 0), '艮卦': (1, 0, 0), '坎卦': (0, 1, 0), '巽卦': (1, 1, 0),
          '中卦': (0.5, 0.5, 0.5),
          '震卦2': (0, 0, 1), '震卦': (0, 0, 1), '離卦': (1, 0, 1), '兌卦': (0, 1, 1), '乾卦': (1, 1, 1)}
reversedHexMap = {value: key for key, value in hexMap.items()}
reversedSymMap = {(0, 0, 0): '☷', (1, 0, 0): '☶', (0, 1, 0): '☵', (1, 1, 0): '☴', (0, 0, 1): '☳',
                  (1, 0, 1): '☲', (0, 1, 1): '☱', (1, 1, 1): '☰', (0.5, 0.5, 0.5): '𝌀'}
'''
vecMap = {'坤卦': [(0, 0, 0), (0, 0, 0)], '艮卦': [(0, 1, 0), (0, 1, 0)], '坎卦': [(0, 0, 0), (0, 1, 0)],
          '巽卦': [(0, 1, 0), (0, 1, 0)],
          '中卦': [(0, 0, 0), (0.5, 0.5, 0.5)], '震卦': [(0, -1, 0), (0, 1, 0)], '離卦': [(1, 0, -1), (1, 0, 1)],
          '離卦2': [(1, 0, 0), (1, 0, 0)], '離卦3': [(0, 0, -1), (0, 0, 1)],
          '兌卦': [(0, -1, 0), (0, 1, 0)], '乾卦': [(0, 0, 0), (1, 1, 1)]}
'''
vecMap = {'坤卦': [(0, 0), (0, 0)], '艮卦': [(1, 1), (1, 1)], '坎卦': [(-1, 1), (1, 1)],
          '巽卦': [(0, 1), (0, 1)],
          '中卦': [(0, 0), (0.5, 0.5)], '震卦': [(-1, -1), (1, 1)], '離卦': [(1, -1), (1, 1)],
          '艮卦2': [(1, 0), (1, 0)], '震卦2': [(0, -1), (0, 1)],
          '兌卦': [(-1, 0), (1, 0)], '乾卦': [(0, 0), (1, 1)]}
charMap = {(0, 0, 0): 'h', (1, 0, 0): 'g', (0, 1, 0): 'r', (1, 1, 0): 'v', (0.5, 0.5, 0.5): 'n',
           (0, 0, 1): 'd', (1, 0, 1): 'b', (0, 1, 1): 's', (1, 1, 1): 'ṡ'}
reversedCharMap = {value: key for key, value in charMap.items()}
charMapDyn = {(0, 0, 0): 'ċ', (1, 0, 0): 'c', (0, 1, 0): 'l', (1, 1, 0): 'f', (0.5, 0.5, 0.5): 'm',
              (0, 0, 1): 't', (1, 0, 1): 'p', (0, 1, 1): 'z', (1, 1, 1): 'ż'}
reversedCharMapDyn = {value: key for key, value in charMapDyn.items()}
reversedCharMapDbl = {**reversedCharMap, **reversedCharMapDyn}
dimMap = {0: 'y', 1: 'x'}
dynMap = {0: '靜', 1: '動'}
leafSymbols = list(reversedCharMapDbl.keys())
nodeSymbols = ['parallel', 'series']
replaceDict = {'parallel': '并', 'series': '串'}
staticMode = False
dynamicMode = True
colorMode = False
colorA = (255, 0, 0)
colorB = (0, 200, 200)
unknownTree = ['parallel', ['series', ['ṡ', 0, 0]], ['series', ['m', 0, 0]]]
unknownTree = ['parallel', ['series', ['h', 0, 0]]]
global axes
global fig
global ani
global axPrev
global axNext
global buttonPrevTree
global buttonNextTree
global buttonPrevSeq
global buttonNextSeq
global curGalleryInd
global curSeqInd
global numGallery


def thresholdImage(image):
    thresh = 100
    fn = lambda x: 255 if x > thresh else 0
    image = image.convert("L").point(fn, mode='1')
    return image


def detectFlow(prevFrame, curFrame):
    flow = cv2.calcOpticalFlowFarneback(prevFrame, curFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def visualizeFlow(image, flow, step=16, arrowScale=5, scaleFactor=1):
    image = image.convert('RGB')
    image = np.array(image)
    h, w = flow.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T * arrowScale
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    for (x1, y1), (x2, y2) in lines:
        x1, y1, x2, y2 = x1 * scaleFactor, y1 * scaleFactor, x2 * scaleFactor, y2 * scaleFactor
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(image, (x2, y2), 1, (0, 255, 0), -1)
    return Image.fromarray(image)


def detectEdge(image, useCanny=False):
    if useCanny:
        threshold = 50  # Adjust this threshold to control edge density
        # Apply edge detection based on pixel intensity differences
        edges = cv2.Canny(np.array(image), threshold, threshold * 3)
        ''' 
        # Display and save the resulting image
        cv2.imshow('Dense Edges Image', edges)
        # Wait for a key press or window close event
        key = cv2.waitKey(1) & 0xFF

        # Continue running until 'q' key is pressed or the window is closed
        while key != ord('q') and cv2.getWindowProperty('Dense Edges Image', cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()
        '''
        # cv2.imwrite('./images/denseEdgesImage.png', edges)
        return Image.fromarray(edges)
    else:
        # Converting the image to grayscale, as edge detection
        # requires input image to be of mode = Grayscale (L)
        image = image.convert("L")
        # image.save(r"./images/grayscale.png")
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        image = image.filter(ImageFilter.FIND_EDGES)

        # Saving the Image Under the name Edge_Sample.png
        # image.save(r"./images/edgeSample.png")
        return image


def detectContour(image):
    image = np.array(image)
    # Step 3: Preprocess the image (optional)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Edge detection (optional)
    edges = cv2.Canny(blurred_image, 50, 150)  # Adjust the thresholds as needed

    # Step 5: Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy)
    # Analyze the hierarchy to find outer and inner contours
    outer_contours = []
    inner_contours = []

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            # Contour with no parent, it's an outer contour
            outer_contours.append(contours[i])
        else:
            # Contour with a parent, it's an inner contour
            inner_contours.append(contours[i])

    # Draw outer contours in green and inner contours in red
    contour_image = np.copy(image)
    cv2.drawContours(contour_image, outer_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_image, inner_contours, -1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Detected Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 6: Draw contours on the original image (optional)
    # contour_image = np.copy(image)
    # Create a blank canvas to draw the contours on
    canvas = np.zeros_like(image)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0),
                     2)  # Draw all contours in green with a thickness of 2 pixels

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Detected Contours', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extractChannels(image):
    rgbImage = image.convert('RGB')
    redImage = rgbImage.getchannel('R')
    greenImage = rgbImage.getchannel('G')
    blueImage = rgbImage.getchannel('B')
    print(redImage, greenImage, blueImage)
    return redImage, greenImage, blueImage


def getScore(pixelSums=None, pixelLens=None, flowSums=None, flowAbsSums=None, verbose=False):  # how well the
    # pixel values fit some trigram
    global punishment
    global punishmentDyn
    maxSums = [length * 255 for length in pixelLens]
    if flowSums is None:
        sensations = [0, 0, 0]
    else:
        sensations = [[0, 0], [0, 0]]  # 1st dim: vector sum, 2nd dim: sum of their abs values

    if flowSums is None:
        for i in range(3):
            if maxSums[i] == 0:
                ratio = 0
            else:
                ratio = pixelSums[i] / maxSums[i]
            # print("ratio", ratio, pixelSums[i], maxSums[i])
            # k = 0.3  # detected contours
            k = 1  # gray image
            sensations[i] = math.sqrt(1 - math.pow(ratio - k, 2) / math.pow(k, 2))
            if sensations[i] > 1:
                sensations[i] = 1
            # sensations[i] = ratio
            # sensations[i] = 1 / (1 + 15 * math.exp(0.2 - ratio))
            if verbose:
                print("ratio", ratio)
    else:
        for i in range(2):
            for j, sums in enumerate([flowSums, flowAbsSums]):
                ratio = sums[i] / pixelLens[i]
                if verbose:
                    hint = ' abs ' if j == 1 else ' '
                    print("flow" + hint + "ratio", ratio)
                k = 1
                if ratio < 0:
                    if ratio < -k:
                        sensations[i][j] = -1
                    else:
                        sensations[i][j] = -math.sqrt(1 - math.pow(ratio + k, 2) / math.pow(k, 2))
                else:
                    if ratio > k:
                        sensations[i][j] = 1
                    else:
                        sensations[i][j] = math.sqrt(1 - math.pow(ratio - k, 2) / math.pow(k, 2))
    scores = []
    if flowSums is None:
        for hexName in hexMap.keys():
            example = hexMap[hexName]
            absDif = []
            avgSensations = sum(sensations) / len(sensations)
            avgExample = sum(example) / len(example)
            relSensations = []
            relExample = []
            relDif = []
            for j in range(3):
                absDif.append(abs(sensations[j] - example[j]))
                relSensations.append(sensations[j] - avgSensations)
                relExample.append(example[j] - avgExample)
                relDif.append(math.pow(abs(relSensations[j] - relExample[j]), 1))
            absDif = sum(absDif)
            relDif = sum(relDif)
            scores.append(absDif / 2 + relDif / 2)
            if example[0] == example[1] and example[1] == example[2]:
                if verbose:
                    print("Punished", scores[-1], scores[-1] * punishment)
                scores[-1] *= punishment
            if verbose:
                print(hexName, "Absolute", absDif, "Relative", relDif, "Total", scores[-1])
    else:
        for hexName in hexMap.keys():
            flowDif = []
            flowAbsDif = []
            for j in range(2):  # upper/lower region
                flowDif.append(abs(sensations[j][0] - vecMap[hexName][0][j]))
                flowAbsDif.append(abs(sensations[j][1] - vecMap[hexName][1][j]))
            flowDif = sum(flowDif)
            flowAbsDif = sum(flowAbsDif)
            scores.append(flowDif / 2 + flowAbsDif / 2)
            example = hexMap[hexName]
            if example[0] == example[1] and example[1] == example[2]:
                if verbose:
                    print("Punished", scores[-1], scores[-1] * punishmentDyn)
                scores[-1] *= punishment
            if verbose:
                print(hexName, "Vector", flowDif, "Absolute", flowAbsDif, "Total", scores[-1])
    score = np.min(scores)
    bestHex = list(hexMap.values())[np.argmin(scores)]

    return score, sensations, bestHex


def getVariance(lst):
    lstSum = sum([math.pow(val - 127.5, 2) for val in lst])
    return lstSum / len(lst)


def splitImage(pixels, origin, depth=0, flow=None, bothOrient=False, flexible=True, buddhist=False):
    global depthMax
    width = len(pixels[0])
    height = len(pixels)
    print("Depth", depth)
    globalBestOrient = 0
    globalBestDynOrient = 0
    globalBestScore = 10
    globalBestDynScore = 10
    bestHex = [None, None]
    bestOffsets = [None, None]
    bestDynHex = [None, None]
    bestDynOffsets = [None, None]
    for orient in [0, 1]:
        offsets = [0, 0]
        if flow is not None:
            dynOffset = 0
        bestScore = 10
        bestDynScore = 10
        print("Orientation:", orient, "Width:", width, "Height:", height)
        if orient == 0:  # tripartite using horizontal lines
            measureA = height
            measureB = width
        elif orient == 1:  # tripartite using vertical lines
            measureB = height
            measureA = width
            pixels = np.array(pixels).T.tolist()
            if flow is not None:
                flow = np.array(flow).transpose((1, 0, 2)).tolist()

        division = measureA / 3
        pixelSums = [0, 0, 0]
        pixelLens = []
        # print([sum(pixels[pos]) for pos in range(measureA)])
        for i in range(3):
            pixelDict = []
            for j in range(round(division * i), round(division * (i + 1))):
                pixelDict += pixels[j]
            pixelLens.append(len(pixelDict))
            pixelSums[i] = sum(pixelDict)
        results = getScore(pixelSums=pixelSums, pixelLens=pixelLens, verbose=True)
        score = results[0]
        sensations = results[1]
        hexagram = results[2]
        bestPixelSums = pixelSums.copy()
        bestPixelLens = pixelLens.copy()
        bestSensations = sensations.copy()
        for i in range(3):
            print("Sensation:", sensations[i], "Pixel sum:", pixelSums[i], "Max sum:", pixelLens[i] * 255,
                  "From", round(division * i), "to", round(division * (i + 1)))
        if flow is not None:
            flowSums = [None, None]
            flowAbsSums = [None, None]
            division = measureA / 2
            for i in range(2):
                flowLst = []
                flowAbsLst = []
                for j in range(round(division * i), round(division * (i + 1))):
                    flowLst += [tup[1 - orient] for tup in flow[j]]
                    flowAbsLst += [abs(tup[1 - orient]) for tup in flow[j]]

                flowSums[i] = sum(flowLst)
                flowAbsSums[i] = sum(flowAbsLst)
            results = getScore(pixelLens=pixelLens, flowSums=flowSums, flowAbsSums=flowAbsSums, verbose=True)
            dynScore = results[0]
            flowSensations = results[1]
            dynHexagram = results[2]
            bestFlowSensations = flowSensations.copy()
            bestFlowSums = flowSums.copy()
            bestFlowAbsSums = flowAbsSums.copy()
            for i in range(2):
                print("Flow:", flowSensations[i])

        division = measureA / 3
        if round(division) > 0 and flexible:
            for i in [0, 1]:
                for j, moveDir in enumerate([1, -1]):
                    newPixelSums = pixelSums.copy()
                    newPixelLens = pixelLens.copy()
                    newOffsets = offsets.copy()
                    newSensations = sensations.copy()
                    division = measureA / 3
                    while True:
                        position = round(division * (i + 1)) + newOffsets[i] + moveDir * int(moveDir == -1)
                        '''
                        if position < 4 or position >= measureA - 4:  # avoids leaving the image
                            print("avoids leaving the image", position, measureA)
                            break
                        '''
                        pixelNum = sum(pixels[position])
                        deltaPixelSums = pixelNum * moveDir
                        deltaPixelLens = measureB * moveDir
                        if newPixelSums[i] + deltaPixelSums <= 0 or newPixelSums[i + 1] - deltaPixelSums <= 0 \
                                or newPixelLens[i] + deltaPixelLens <= 0 or newPixelLens[i + 1] - deltaPixelLens <= 0:
                            # avoids squeezing the middle space or creating empty space
                            print("avoids squeezing & empty", newPixelSums[i], newPixelSums[i + 1], deltaPixelSums)
                            break
                        newOffsets[i] += moveDir
                        for k in [0, 1]:
                            addSign = 1 if k == 0 else -1
                            newPixelSums[i + k] += deltaPixelSums * addSign
                            newPixelLens[i + k] += deltaPixelLens * addSign
                        results = getScore(pixelSums=newPixelSums, pixelLens=newPixelLens)
                        newScore = results[0]
                        newSensations = results[1]
                        hexagram = results[2]
                        '''
                        print("Captured pixels:", pixelNum, "Sensations", newSensations, 'Variances', newVariances,
                              "Pos:", position,
                              "Line:", i, "Dir:", moveDir, "Offsets", newOffsets, "New score", newScore)
                        '''
                        # print("subtle dif", score - newScore)
                        if newScore >= score:  # error-tolerant rate
                            break
                        score = newScore
                    print("Current score:", score, "Best score", bestScore)
                    if score < bestScore:
                        print("Best score", score, "Hexagram", hexagram, "Sensations", newSensations, "Offsets",
                              newOffsets)
                        bestScore = score
                        bestPixelSums = newPixelSums
                        bestPixelLens = newPixelLens
                        bestSensations = newSensations
                        bestOffsets[orient] = newOffsets
                        bestHex[orient] = hexagram
                    if score < globalBestScore:
                        globalBestScore = score  # globally best offsets consider every orientation
                        globalBestOrient = orient

                    if flow is not None and i == 0:
                        newFlowSums = flowSums.copy()
                        newFlowAbsSums = flowAbsSums.copy()
                        newFlowSensations = flowSensations.copy()
                        newDynOffset = dynOffset
                        division = measureA / 2
                        while True:
                            position = round(division) + newDynOffset + moveDir * int(moveDir == -1)
                            if position >= len(flow):
                                break
                            print("position", position, "len", len(flow))
                            flowNum = sum([tup[1 - orient] for tup in flow[position]])
                            flowAbsNum = sum([abs(tup[1 - orient]) for tup in flow[position]])
                            deltaFlowSums = flowNum * moveDir
                            deltaFlowAbsSums = flowAbsNum * moveDir
                            if newFlowAbsSums[i] + deltaFlowAbsSums <= 0 \
                                    or newFlowAbsSums[i + 1] - deltaFlowAbsSums <= 0:
                                # avoids squeezing the middle space or creating empty space
                                print(newFlowAbsSums[i], newFlowAbsSums[i + 1], deltaFlowAbsSums)
                                break
                            newDynOffset += moveDir
                            for k in [0, 1]:
                                addSign = 1 if k == 0 else -1
                                newFlowSums[i + k] += deltaFlowSums * addSign
                                newFlowAbsSums[i + k] += deltaFlowAbsSums * addSign
                            results = getScore(pixelLens=newPixelLens, flowSums=newFlowSums,
                                               flowAbsSums=newFlowAbsSums)
                            newDynScore = results[0]
                            newFlowSensations = results[1]
                            dynHexagram = results[2]
                            '''
                            print("Captured pixels:", pixelNum, "Sensations", newSensations, 'Variances', newVariances,
                                  "Pos:", position,
                                  "Line:", i, "Dir:", moveDir, "Offsets", newOffsets, "New score", newScore)
                            '''
                            # print("subtle dif", score - newScore)
                            if newDynScore >= dynScore:  # error-tolerant rate
                                break
                            dynScore = newDynScore

                        if dynScore < bestDynScore:
                            print("Best score", dynScore, "Hexagram", dynHexagram, "Sensations", newFlowSensations,
                                  "Offsets",
                                  newDynOffset)
                            bestFlowSums = newFlowSums
                            bestFlowAbsSums = newFlowAbsSums
                            bestFlowSensations = newFlowSensations
                            bestDynOffsets[orient] = newDynOffset
                            bestDynHex[orient] = dynHexagram
                        if dynScore < globalBestDynScore:
                            globalBestDynScore = dynScore  # globally best offsets consider every orientation
                            globalBestDynOrient = orient
                pixelSums = bestPixelSums
                pixelLens = bestPixelLens
                sensations = bestSensations
                offsets = bestOffsets[orient]
                if flow is not None and i == 0:
                    flowSums = bestFlowSums
                    flowAbsSums = bestFlowAbsSums
                    flowSensations = bestFlowSensations
                    dynOffset = bestDynOffsets[orient]
        else:
            bestHex[orient] = hexagram
            bestOffsets[orient] = [0, 0]
            if flow is not None:
                bestDynHex[orient] = dynHexagram
                bestDynOffsets[orient] = 0
            if score < globalBestScore:
                globalBestScore = score
                globalBestOrient = orient
                if flow is not None:
                    globalBestDynScore = dynScore
                    globalBestDynOrient = [0, 0]

    print("Best Hex", bestHex, "Best score", globalBestScore, "Best orientation", globalBestOrient,
          "Best Offsets",
          bestOffsets)

    offsetMap = [[0, 0], [1, 0], [0, 1]]

    iterable = [0, 1] if bothOrient else [globalBestOrient]
    tree = [[], []]
    rays = [[], []]
    tPixels = pixels
    pixels = np.array(tPixels).T.tolist()
    if flow is not None:
        tFlow = flow
        flow = np.array(flow).transpose((1, 0, 2)).tolist()

    for curOrient in iterable:
        extOffsets = [0] + bestOffsets[curOrient] + [0]
        if curOrient == 0:
            division = height / 3
            rays[curOrient] += [[(origin[0], origin[1] + height / 3 * i + bestOffsets[curOrient][i - 1]),
                                 (origin[0] + len(tPixels), origin[1] + height / 3 * i + bestOffsets[curOrient][i - 1])]
                                for i in [1, 2]]
            if flow is not None:
                rays[curOrient] += [[(origin[0], origin[1] + height / 2 + bestDynOffsets[curOrient]),
                                     (origin[0] + len(tPixels), origin[1] + height / 2 + bestDynOffsets[curOrient])]]
            hexTup = [bestHex[curOrient], bestDynHex[curOrient]]
            if not buddhist:  # takes into account of the default evenness & stillness
                if hexTup[0] == (0.5, 0.5, 0.5):
                    hexTup[0] = []
                if hexTup[1] == (0, 0, 0):
                    hexTup[1] = []
            tree[curOrient] = [(origin[0] + width / 2, origin[1] + height / 2), tuple(hexTup)]
            if depth < depthMax:
                results = []
                for i in range(3):
                    print("Section", i, "Start", round(division * i) + extOffsets[i],
                          "End", round(division * (i + 1)) + extOffsets[i + 1],
                          "Origin", origin[0], origin[1] + height / 3 * i +
                          offsetMap[i][0] * bestOffsets[curOrient][0] + offsetMap[i][1] * bestOffsets[curOrient][1])
                    if (round(division * (i + 1)) + extOffsets[i + 1]) - (round(division * i) + extOffsets[i]) < 3:
                        print("too narrow")
                        results.append([])
                        continue
                    if flow is not None:
                        subFlow = flow[round(division * i) + extOffsets[i]:
                                       round(division * (i + 1)) + extOffsets[i + 1]]
                    else:
                        subFlow = None
                    results.append(splitImage(pixels[round(division * i) + extOffsets[i]:
                                                     round(division * (i + 1)) + extOffsets[i + 1]],
                                              (origin[0],
                                               origin[1] + height / 3 * i +
                                               offsetMap[i][0] * bestOffsets[curOrient][0] + offsetMap[i][1] *
                                               bestOffsets[curOrient][1]),
                                              depth=depth + 1,
                                              flow=subFlow,
                                              bothOrient=bothOrient,
                                              flexible=flexible, buddhist=buddhist))
                print('results', results, bestHex)
                if (results[0] == [] or results[0][1][0] == [] or not bestHex[0] == results[0][1][0][1]) or \
                        (results[1] == [] or results[1][1][0] == [] or not bestHex[0] == results[1][1][0][1]) or \
                        (results[2] == [] or results[2][1][0] == [] or not bestHex[0] == results[2][1][0][1]):
                    for result in results:
                        if result:
                            rays[curOrient] += result[0][0] + result[0][1]
                            tree[curOrient].append(result[1])
                        else:
                            tree[curOrient].append([])
                else:
                    if not bestHex[0] == results[0][1][1][1] \
                            or not bestHex[0] == results[1][1][1][1] \
                            or not bestHex[0] == results[2][1][1][1] \
                            or bestHex[0] not in [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)]:
                        for result in results:
                            if result:
                                rays += result[0][1]  # divide the other dimension
                                tree[curOrient].append([[], result[1][1]])
                            else:
                                tree[curOrient].append([])
                        print("only add orient", orient, tree)
                    else:
                        print("skip monotonous", bestHex[0])

        else:
            division = width / 3
            rays[curOrient] += [[(origin[0] + width / 3 * i + bestOffsets[curOrient][i - 1], origin[1]),
                                 (origin[0] + width / 3 * i + bestOffsets[curOrient][i - 1],
                                  origin[1] + len(tPixels[0]))] for i
                                in [1, 2]]
            if flow is not None:
                rays[curOrient] += [[(origin[0] + width / 2 + bestDynOffsets[curOrient], origin[1]),
                                     (origin[0] + width / 2 + bestDynOffsets[curOrient], origin[1] + len(tPixels[0]))]]
            hexTup = [bestHex[curOrient], bestDynHex[curOrient]]
            if not buddhist:
                if hexTup[0] == (0.5, 0.5, 0.5):
                    hexTup[0] = []
                if hexTup[1] == (0, 0, 0):
                    hexTup[1] = []
            tree[curOrient] = [(origin[0] + width / 2, origin[1] + height / 2), tuple(hexTup)]
            if depth < depthMax:
                results = []
                for i in range(3):
                    print("Section", i, "Start", round(division * i) + extOffsets[i],
                          "End", round(division * (i + 1)) + extOffsets[i + 1],
                          "Ext offsets", extOffsets, "Division", division,
                          "Origin", origin[0] + width / 3 * i +
                          offsetMap[i][0] * bestOffsets[curOrient][0] + offsetMap[i][1] * bestOffsets[curOrient][1]
                          , origin[1])
                    if (round(division * (i + 1)) + extOffsets[i + 1]) - (round(division * i) + extOffsets[i]) < 3:
                        print("too narrow")
                        results.append([])
                        continue
                    if flow is not None:
                        subFlow = np.array(tFlow[round(division * i) + extOffsets[i]:
                                                 round(division * (i + 1)) + extOffsets[i + 1]]).transpose(
                            (1, 0, 2)).tolist()
                    else:
                        subFlow = None
                    results.append(splitImage(np.array(tPixels[round(division * i) + extOffsets[i]:
                                                               round(division * (i + 1)) + extOffsets[
                                                                   i + 1]]).T.tolist(),
                                              (origin[0] + width / 3 * i +
                                               offsetMap[i][0] * bestOffsets[curOrient][0] + offsetMap[i][1] *
                                               bestOffsets[curOrient][1],
                                               origin[1]),
                                              depth=depth + 1,
                                              flow=subFlow,
                                              bothOrient=bothOrient,
                                              flexible=flexible, buddhist=buddhist))
                print(results, bestHex)
                if (results[0] == [] or results[0][1][1] == [] or not bestHex[1] == results[0][1][1][1]) or \
                        (results[1] == [] or results[1][1][1] == [] or not bestHex[1] == results[1][1][1][1]) or \
                        (results[2] == [] or results[2][1][1] == [] or not bestHex[1] == results[2][1][1][1]):
                    for result in results:
                        if result:
                            rays[curOrient] += result[0][0] + result[0][1]
                            tree[curOrient].append(result[1])
                        else:
                            tree[curOrient].append([])
                else:
                    if not bestHex[1] == results[0][1][0][1] \
                            or not bestHex[1] == results[1][1][0][1] \
                            or not bestHex[1] == results[2][1][0][1] \
                            or bestHex[1] not in [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)]:
                        for result in results:
                            if result:
                                rays[curOrient] += result[0][0]  # divide the other dimension
                                tree[curOrient].append([result[1][0], []])
                            else:
                                tree[curOrient].append([])
                        print("only add orient", orient, tree)
                    else:
                        print("skip monotonous", bestHex[1])
    return rays, tree


def produceDense():
    # Define image properties
    image_size = (1000, 1000)

    # Generate random noise image
    # random_image = np.random.randint(0, 256, image_size, dtype=np.uint8)
    random_image = np.random.choice([0, 255], image_size)
    # Save the random image
    cv2.imwrite('./images/randomImage.png', random_image)
    # detectEdge(random_image, useCanny=True)


def drawRays(image, rays, scaleFactor=1):
    global colorA
    global colorB
    rgbImage = image.convert('RGB')
    rayImage = ImageDraw.Draw(rgbImage)
    for orient in [0, 1]:
        branch = rays[orient]
        for ray in branch:
            newRay = [(ray[0][0] * scaleFactor, ray[0][1] * scaleFactor),
                      (ray[1][0] * scaleFactor, ray[1][1] * scaleFactor)]
            if ray[0][0] == ray[1][0]:
                color = colorB
            if ray[0][1] == ray[1][1]:
                color = colorA
            rayImage.line(newRay, fill=color, width=0)
    return rgbImage


def drawLabels(image, tree, allTrees=False, scaleFactor=1):
    rgbImage = image.convert('RGB')
    labelImage = ImageDraw.Draw(rgbImage)
    labelHelper(labelImage, tree, allTrees, scaleFactor=scaleFactor)
    return rgbImage


def labelHelper(image, tree, allTrees, scaleFactor=1):
    global colorA
    global colorB
    size = 50
    if not tree:
        return None
    unicodeFont = ImageFont.truetype("uming.ttc", size)
    for orient in [0, 1]:
        branch = tree[orient]
        if not branch:
            continue
        dynamicExists = branch[1][1] is not None

        for i, isDynamic in enumerate([False, True]):
            if not branch[1][i]:
                continue
            if dynamicExists:
                shiftX = -1 if isDynamic is False else 1
                wordLen = 2.5
            else:
                if isDynamic:
                    break
                shiftX = 0
                wordLen = 1.5
            shiftY = -0.5 if orient == 0 else 0.5
            coord = ((branch[0][0] + size / scaleFactor * wordLen / 2 * shiftX) * scaleFactor,
                     (branch[0][1] + size / scaleFactor * shiftY) * scaleFactor)
            if len(branch) == 2 or allTrees:
                if not orient:
                    color = colorA
                else:
                    color = colorB
                hexName = reversedHexMap[tuple(branch[1][i])][0]
                # print('thar barr', branch, isDynamic, hexName + dynMap[i] * dynamicExists + dimMap[orient], branch[0], size / scaleFactor * wordLen / 2 * shiftX)
                # print(hexName)
                image.text(coord, hexName + dynMap[i] * dynamicExists + dimMap[orient], font=unicodeFont, fill=color)
            if len(branch) > 2:
                for subtree in branch[2:]:
                    labelHelper(image, subtree, allTrees, scaleFactor=scaleFactor)


def drawTree(tree):
    global depthMax
    width = height = int(math.pow(3, depthMax + 1))
    pixels = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append([0, 0])
        pixels.append(row)
    pixels = treeHelper(tree, (0, 0), width, height, pixels)
    print(pixels)

    for i in range(height):
        for j in range(width):
            pixels[i][j] = pixels[i][j][0] / pixels[i][j][1] * 255
    array = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(array, 'L')
    return image


def treeHelper(tree, origin, width, height, pixels):
    # print(tree, origin, width, height, pixels)
    for orient in [0, 1]:
        branch = tree[orient]
        if not branch or not branch[1][0]:
            continue
        if orient == 0:
            for i in range(3):
                for k in range(height // 3):
                    for j in range(width):
                        pixels[origin[1] + i * height // 3 + k][origin[0] + j][0] += branch[1][0][i]
                        pixels[origin[1] + i * height // 3 + k][origin[0] + j][1] += 1
                if len(branch) > 2 and branch[2 + i] != []:
                    pixels = treeHelper(branch[2 + i], (origin[0], origin[1] + i * height // 3), width, height // 3,
                                        pixels)
        elif orient == 1:
            for i in range(3):
                for k in range(width // 3):
                    for j in range(height):
                        pixels[origin[1] + j][origin[0] + i * width // 3 + k][0] += branch[1][0][i]
                        pixels[origin[1] + j][origin[0] + i * width // 3 + k][1] += 1
                if len(branch) > 2 and branch[2 + i] != []:
                    pixels = treeHelper(branch[2 + i], (origin[0] + i * width // 3, origin[1]), width // 3, height,
                                        pixels)

    return pixels


def convert2GuideTree(tree):
    guideTree = []
    for orient in [0, 1]:
        branch = tree[orient]
        if not branch:
            continue
        guideBranch = []
        for i, isDynamic in enumerate([False, True]):
            if not branch[1][i] or branch[1][i] is None:
                continue
            referent = charMapDyn if isDynamic else charMap
            char = referent[branch[1][i]]
            guideBranch.append(['series', [char, 1 - orient, 0]])
        compLst = []
        if len(branch) > 2:
            for comp in branch[2:]:
                if not comp:
                    continue
                subTree = convert2GuideTree(comp)
                if subTree:
                    if subTree[0] == 'series':
                        subTree = subTree[1:]
                        compLst += subTree
                    else:
                        compLst.append(subTree)
            if compLst:
                guideBranch.append(['series'] + compLst)
        if not guideBranch:
            continue
        elif len(guideBranch) > 1:
            guideTree.append(['series', ['parallel'] + guideBranch])
        else:
            guideTree.append(guideBranch[0])
    if not guideTree:
        return []
    elif len(guideTree) > 1:
        guideTree = ['parallel'] + guideTree
    else:
        guideTree = guideTree[0]
    return guideTree


def reverseGuideTree(guideTree):
    comps = guideTree[1:]
    if guideTree[0] == 'series':
        newGuideTree = ['series'] + comps[::-1]
        for i in range(len(comps)):
            if newGuideTree[1 + i] == 'filler':
                continue
            newGuideTree[1 + i] = reverseGuideTree(newGuideTree[1 + i])
    elif guideTree[0] == 'parallel':
        newGuideTree = ['parallel']
        for i in range(len(comps)):
            if guideTree[1 + i] == 'filler':
                continue
            newGuideTree.append(reverseGuideTree(guideTree[1 + i]))
    else:
        newGuideTree = guideTree.copy()

    return newGuideTree


def isPair(lst):
    return isinstance(lst, list) and len(lst) == 2 and isinstance(lst[0], int) and isinstance(lst[1], int)


def filterLayer(guideTree, layer, codeMode=True, code=None, mirrorMode=True, verbose=False):
    if code is None:
        code = 0
    newGuideTree = [guideTree[0]]
    addCode = codeMode and guideTree[0] == 'parallel' and len(guideTree) > 2 and guideTree[-1] != 'filler'
    if addCode:
        impureBranches = []
        pureBranches = {}
        totalLayerLst = []
        for branch in guideTree[1:]:
            layerLst = getAllLayers(branch)
            layerSet = set(layerLst)
            if len(layerSet) > 1:
                impureBranches.append(branch)
            else:
                curLayer = list(layerSet)[0]
                if curLayer in pureBranches.keys():
                    pureBranches[curLayer].append(branch)
                else:
                    pureBranches[curLayer] = [branch]
                totalLayerLst += layerLst
            # print(branch, layerLst, totalLayerLst)
        totalLayerLst = sorted(set(totalLayerLst))
        for i in range(len(totalLayerLst)):
            if totalLayerLst[i] == layer or mirrorMode and totalLayerLst[i] == -layer:
                codePair = [code, code + 1]
                if i == 0 or totalLayerLst[i] - 1 != totalLayerLst[i - 1]:
                    codePair[0] = 0
                if i == len(totalLayerLst) - 1 or totalLayerLst[i] + 1 != totalLayerLst[i + 1]:
                    codePair[1] = 0
                for pureBranch in pureBranches[totalLayerLst[i]]:
                    newGuideTree.append(soakCode(pureBranch, codePair, totalLayerLst[i]))
                    if verbose:
                        print('soaked', pureBranch, codePair, totalLayerLst, i)
            if i < len(totalLayerLst) - 1:
                code += 1
        for branch in impureBranches:
            branch, code = filterLayer(branch, layer, codeMode=codeMode, code=code, mirrorMode=mirrorMode,
                                       verbose=verbose)
            newGuideTree.append(branch)
    elif guideTree[0] in ['parallel', 'series']:
        for comp in guideTree[1:]:
            if comp == 'filler':
                newGuideTree.append(comp)
                continue
            if comp[0] in ['parallel', 'series']:
                comp, code = filterLayer(comp, layer, codeMode=codeMode, code=code, mirrorMode=mirrorMode,
                                         verbose=verbose)
                if verbose:
                    print("comp", comp, "code", code)
                newGuideTree.append(comp)
            elif comp[2] == layer or mirrorMode and comp[2] == -layer:
                newGuideTree.append(comp)
    return newGuideTree, code


def soakCode(tree, codePair, layer):
    if tree[0] in ['series', 'parallel']:
        for comp in tree[1:]:
            if comp == 'filler':
                continue
            soakCode(comp, codePair, layer)
    else:
        if isPair(tree[-1]) and tree[-2] == layer:
            tree[-1] = codePair
        elif tree[-1] == layer:
            tree.append(codePair)
    return tree


def checkLayersExist(guideTree, layer):
    foundLayers = [False, False, False]
    if guideTree[0] == 'series':
        for comp in guideTree[1:]:
            if comp == 'filler':
                continue
            results = checkLayersExist(comp, layer)
            if comp[0] == 'parallel' and sum([int(val) for val in results]) > 1:  # fuseau with mixed layers is invalid
                results = [False, False, False]
            foundLayers = list(map(any, zip(foundLayers, results)))
    elif guideTree[0] == 'parallel':
        for comp in guideTree[1:]:
            if comp == 'filler':
                continue
            results = checkLayersExist(comp, layer)
            foundLayers = list(map(any, zip(foundLayers, results)))
    else:
        for i in [-1, 0, 1]:
            if guideTree[2] == layer + i:
                foundLayers[i + 1] = True
    return foundLayers


def fillDynamic(guideTree):
    hasStatic = []
    if guideTree[0] in ['series', 'parallel']:
        newGuideTree = [guideTree[0]]
        for comp in guideTree[1:]:
            newComp, compHasStatic = fillDynamic(comp)
            if guideTree[0] == 'parallel':
                for layer in compHasStatic:
                    if layer not in hasStatic:
                        hasStatic.append(layer)
            newGuideTree.append(newComp)
        if guideTree[0] == 'parallel':
            for i, branch in enumerate(newGuideTree):
                if i == 0:
                    continue
                for j, comp in enumerate(branch):
                    if j == 0:
                        continue
                    if comp[0] not in ['series', 'parallel'] and newGuideTree[i][j][0] in reversedCharMapDyn and \
                            newGuideTree[i][j][2] not in hasStatic:
                        newGuideTree[i][j] = ['parallel', ['series', newGuideTree[i][j]],
                                              ['series', ['ṡ', newGuideTree[i][j][1], newGuideTree[i][j][2]]], 'filler']
    else:
        newGuideTree = guideTree.copy()
        hasStatic = [guideTree[2]] * int(guideTree[0] in reversedCharMap)
    return newGuideTree, hasStatic


def hollowDynamic(guideTree):
    if guideTree[0] == 'parallel' and guideTree[-1] == 'filler':
        if len(guideTree) == 2:
            return []
        else:
            if len(guideTree[1]) > 1:
                return guideTree[1][1]
            else:
                return []
    elif guideTree[0] in ['parallel', 'series']:
        newGuideTree = [guideTree[0]]
        for comp in guideTree[1:]:
            result = hollowDynamic(comp)
            if result:
                newGuideTree.append(result)
        return newGuideTree
    else:
        return guideTree.copy()


def removeVoid(guideTree):
    if guideTree[0] == 'parallel':
        newGuideTree = guideTree[:-1] if guideTree[-1] == 'filler' else guideTree
        guideTree = ['parallel']
        for branch in newGuideTree[1:]:
            if not all([branch[i][0] == 'h' for i in range(1, len(branch))]):
                guideTree.append(branch)
        if len(guideTree) == 1:
            guideTree.append(['series', ['h', 0, 0]])

    if guideTree[0] in ['series', 'parallel']:
        newGuideTree = [guideTree[0]]
        for comp in guideTree[1:]:
            newGuideTree.append(removeVoid(comp))
    else:
        newGuideTree = guideTree.copy()

    return newGuideTree

def getAllLayers(guideTree):
    layerLst = []
    if guideTree[0] in ['series', 'parallel']:
        for comp in guideTree[1:]:
            if comp == 'filler':
                continue
            layerLst += getAllLayers(comp)
    else:
        layerLst = [guideTree[2]]
    return layerLst


def countDepth(tree, maxDepth=0):
    if tree[0] == 'series':
        maxDepth += 1
    if tree[0] == 'parallel' or tree[0] == 'series':
        pastMaxDepth = maxDepth
        for branch in tree[1:]:
            depth = countDepth(branch, maxDepth=pastMaxDepth)
            if depth > maxDepth:
                maxDepth = depth
    return maxDepth


def countBranch(tree):
    branchNum = 0
    if tree[0] == 'series':
        for comp in tree[1:]:
            branchNum += countBranch(comp)
    elif tree[0] == 'parallel':
        for branch in tree[1:]:
            if branch == 'filler':
                continue
            newBranchNum = countBranch(branch)
            if newBranchNum > branchNum:
                branchNum = newBranchNum
    else:
        branchNum = 3

    return branchNum


def drawGuideTree(tree, precision=1, lastProbDicts=[], layerMode=False, normalize=True, verbose=False,
                  hollowDyn=False, acceleration=1, frameStart=0, frameDiv=1, moveHistDict=None, displayTree=False, reverse=True):
    # precision means the divisor of the frame gap between each coordinate update
    global depthMax
    global unknownTree
    # width = height = round(math.pow(3, math.ceil(countDepth(tree) / 2) + 1))
    width = height = countBranch(tree)
    frameLen = round(width * acceleration / frameDiv * (frameStart + 1)) - round(width * acceleration / frameDiv * frameStart)
    if frameLen == 0 or frameLen == 1:
        frameLen = 2
    frameLst = []
    if verbose:
        print("Frame length", frameLen, 'Frame div', frameDiv, 'Precision', precision, 'Step', frameLen / precision,
              'Width', width, 'Height', height, 'Frame Start', frameStart, tree)
    probDicts = [] if layerMode else None
    guideTreeLst = []
    if displayTree:
        exhibitTreeLst = []
    framePrecI = frameLen * frameStart
    frameI = round(framePrecI)
    lastFrameI = -1
    frameILst = []
    while frameI < frameLen * (1 + frameStart):
        if lastFrameI != -1 and frameI == lastFrameI:
            lastFrameI = frameI
            framePrecI += frameLen / precision
            frameI = round(framePrecI)
            continue
        currentTree = tree
        if verbose:
            print('Frame Precise I', framePrecI, 'frame I', frameI)

        if lastProbDicts and layerMode:
            if verbose:
                print('Last Prob Dict', lastProbDicts[int((framePrecI - frameLen * frameStart)
                                                          / frameLen * (len(lastProbDicts) - 1))])
                # print('Original Tree', currentTree)
            choiceDict = {}
            currentProbDict = lastProbDicts[int((framePrecI - frameLen * frameStart)
                                                / frameLen * (len(lastProbDicts) - 1))]
            for key in currentProbDict:
                choiceDict[key] = np.random.rand() < currentProbDict[key]
            if verbose:
                print("Choice Dict", choiceDict)
            currentTree = drawByLots(currentTree, choiceDict)
            if verbose:
                print("Drawn by Lots", currentTree)

        if len(currentTree) <= 1:
            currentTree = unknownTree

        currentTree = cleanGuideTree(currentTree, keepCode=True)
        if len(currentTree) <= 1:
            currentTree = unknownTree
        #else:
        #    currentTree = ['parallel', ['series', currentTree]]
        if verbose:
            print('Cleaned Tree', currentTree)
        guideTreeLst.append(currentTree)

        if displayTree and frameI == round(frameLen * frameStart):  # display as zeroth layer tree
            if reverse:
                exhibitTree = reverseGuideTree(currentTree)  # only needed if the test tree is generated, not handwritten
            else:
                exhibitTree = currentTree
            if hollowDyn:
                exhibitTree = hollowDynamic(exhibitTree)
                if verbose:
                    print("Hollowed Tree", exhibitTree)
            exhibitTree = cleanGuideTree(exhibitTree, keepCode=False)
            if len(exhibitTree) <= 1:
                exhibitTree = unknownTree
            #else:
            #    exhibitTree = ['parallel', exhibitTree]
            exhibitTreeLst.append(exhibitTree)
        frameILst.append(frameI)
        lastFrameI = frameI
        framePrecI += frameLen / precision
        frameI = round(framePrecI)
    if verbose:
        print("Guide Tree List", guideTreeLst)

    pixelDict = {}  # pixel arrays covering every frame
    if layerMode:
        codeMap = []  # code maps covering every frame
        for i in range(height):
            codeRow = []
            for j in range(width):
                codeRow.append([])
            codeMap.append(codeRow)
    else:
        codeMap = None
    for frameI in frameILst:
        pixels = []
        for i in range(height):
            row = []
            for j in range(width):
                row.append([0, 0])
            pixels.append(row)
        pixelDict[frameI] = pixels
    if verbose:
        print('frameILst', frameILst)
    calcSeqDict = extractSeq(guideTreeLst, frameILst)
    if verbose:
        print("calcSeqDict", calcSeqDict)
    if calcSeqDict == {}:
        calcSeqDict[lst2tup(unknownTree)] = frameILst
    newMoveHistDict = {}
    for guideTree, calcSeq in calcSeqDict.items():
        lstGuideTree = tup2lst(guideTree)
        if moveHistDict is not None and guideTree in moveHistDict.keys():
            moveHist = moveHistDict[guideTree]
        else:
            # print("lack move hist", guideTree, moveHistDict)
            moveHist = None
        # calcSeq = a series of frame Is that should be calculated for a certain guide tree
        pixelDict, codeMap, newMoveHist = guideTreeHelper(lstGuideTree, (0, 0), width, height,
                                                          pixelDict, codeMap=codeMap, calcSeq=calcSeq,
                                                          precision=precision,
                                                          moveHist=moveHist,
                                                          bigWidth=len(pixelDict[min(pixelDict.keys())][0]),
                                                          bigHeight=len(pixelDict[min(pixelDict.keys())]))
        newMoveHistDict[guideTree] = newMoveHist
    if verbose:
        print("Done Calculating Sequences")

    for frameI, pixels in pixelDict.items():
        # print("code map", codeMap)
        # print("pixels", pixels)
        if layerMode:
            probDict = {}
        for i in range(height):
            for j in range(width):
                # pixels[i][j][1] -= 1
                if pixels[i][j][1] == 0:
                    pixels[i][j][1] = 1
                pixels[i][j] = pixels[i][j][0] / pixels[i][j][1]
                if pixels[i][j] < 0:
                    pixels[i][j] = 0
                if pixels[i][j] > 1:
                    pixels[i][j] = 1
                if layerMode and codeMap[i][j] != []:
                    for mapLegend in codeMap[i][j]:
                        if mapLegend not in probDict.keys():
                            probDict[mapLegend] = [pixels[i][j]]
                        else:
                            probDict[mapLegend].append(pixels[i][j])
                pixels[i][j] *= 255

        if layerMode:
            if normalize:
                maxProb = 0
            for key in probDict.keys():
                probDict[key] = sum(probDict[key]) / len(probDict[key])
                if normalize and probDict[key] > maxProb:
                    maxProb = probDict[key]
            if normalize:
                for key in probDict.keys():
                    if maxProb > 0:
                        probDict[key] /= maxProb
            probDicts.append(probDict)
            if verbose:
                print('New Prob Dict', probDict)

        array = np.array(pixels, dtype=np.uint8)
        image = Image.fromarray(array, 'L')
        frameLst.append(image)
    if displayTree:
        guideTreeLst = exhibitTreeLst
    return frameLst, probDicts, guideTreeLst, newMoveHistDict


def lst2tup(lst):
    if isinstance(lst, list):
        return tuple(lst2tup(item) for item in lst)
    else:
        return lst


def tup2lst(tup):
    if isinstance(tup, tuple):
        return [tup2lst(item) for item in tup]
    else:
        return tup


def extractSeq(guideTreeLst, frameILst):  # extract the possible sequences under various probabilistic factors
    calcSeqDict = {}
    for i, guideTree in enumerate(guideTreeLst):
        guideTree = lst2tup(guideTree)
        if guideTree not in calcSeqDict:
            calcSeqDict[guideTree] = [frameILst[i]]
        else:
            calcSeqDict[guideTree].append(frameILst[i])
    return calcSeqDict


def drawByLots(guideTree, choiceDict):
    if guideTree[0] in ['series', 'parallel']:
        newGuideTree = [guideTree[0]]
        for comp in guideTree[1:]:
            if comp == 'filler':
                newGuideTree.append('filler')
                continue
            result = drawByLots(comp, choiceDict)
            if result:
                newGuideTree.append(result)
        return newGuideTree
    else:
        if isPair(guideTree[-1]) and guideTree[-1][0] != 0:
            if guideTree[-1][0] not in choiceDict:
                # choice = np.random.choice([True, False])
                choice = True
            else:
                choice = choiceDict[guideTree[-1][0]]

            if not choice:
                return []
        return guideTree.copy()


def cleanGuideTree(guideTree, keepCode=True):  # only complete: only clean the one-element lists
    if isPair(guideTree[-1]) and not keepCode:
        guideTree = guideTree[:-1]
    if guideTree[0] in ['series', 'parallel']:
        if len(guideTree) == 1:
            return []
        else:
            newGuideTree = [guideTree[0]]
            completeExists = False
            for comp in guideTree[1:]:
                result = cleanGuideTree(comp, keepCode=keepCode)
                if result == 'filler':
                    newGuideTree.append('filler')
                    # if the fuseau has only a filler, then not one complete branch exists
                    continue
                if len(result) > 1:
                    completeExists = True
                if result:
                    if result[0] == 'series' and guideTree[0] == 'series':
                        newGuideTree += result[1:]
                    else:
                        newGuideTree.append(result)
            if completeExists:
                if newGuideTree[0] == 'parallel' and len(newGuideTree) == 2:
                    if newGuideTree[1] == 'filler':
                        return []
                    else:
                        return newGuideTree[1]
                else:
                    return newGuideTree
            else:
                return []
    else:
        if guideTree == 'filler':
            return guideTree
        else:
            return guideTree.copy()


def findTwo(lst):
    for sublist in lst:
        if len(sublist) == 2:
            return sublist
    return None  # Return None if no such list is found


def guideTreeHelper(tree, origin, width, height, pixelDict, orient=0, pixelMovements=None, needCalc=False,
                    depth=0, codeMap=None, calcSeq=None, precision=1, moveHist=None, bigWidth=-1, bigHeight=-1):
    if tree[0] == 'parallel':
        if tree[-1] == 'filler':
            tree = tree[:-1]
        foundBranch = findTwo(tree[1:])
        if foundBranch is not None:
            orient = foundBranch[1][1]
        if depth == 0:
            pixelMovements = [[[0, 0] for _ in range(bigWidth)] for _ in range(bigHeight)]
            for i in range(2):
                for branch in tree[1:]:
                    if len(branch) == 1:
                        continue
                    if i == 0:
                        pixelMovements = guideTreeHelper(branch, (origin[0], origin[1]), width, height,
                                                         pixelDict, orient=orient, pixelMovements=pixelMovements,
                                                         needCalc=True, depth=depth + 1)
                    else:
                        pixelDict, codeMap, newMoveHist = guideTreeHelper(branch, (origin[0], origin[1]), width, height,
                                                                          pixelDict, orient=orient,
                                                                          pixelMovements=pixelMovements,
                                                                          needCalc=False, depth=depth + 1,
                                                                          codeMap=codeMap,
                                                                          calcSeq=calcSeq, precision=precision,
                                                                          moveHist=moveHist,
                                                                          bigWidth=bigWidth, bigHeight=bigHeight)
                # print("pixel movements", pixelMovements, bigWidth, bigHeight)
            # print("new move hist", newMoveHist)
            return pixelDict, codeMap, newMoveHist
        else:
            newMoveHist = {}
            for branch in tree[1:]:
                if len(branch) == 1:
                    continue
                result = guideTreeHelper(branch, (origin[0], origin[1]), width, height,
                                         pixelDict, orient=orient,
                                         pixelMovements=pixelMovements, needCalc=needCalc, depth=depth + 1,
                                         codeMap=codeMap, calcSeq=calcSeq, precision=precision, moveHist=moveHist,
                                         bigWidth=bigWidth, bigHeight=bigHeight)
                if not needCalc:
                    newMoveHist |= result[-1]
            # print("parallel result", tree, result)
            if not needCalc:
                result = result[0], result[1], newMoveHist
            return result
    elif tree[0] == 'series':
        newMoveHist = {}
        if orient == 1:  # vertical alignment
            division = height / len(tree[1:])
            for i, comp in enumerate(tree[1:]):
                if len(comp) == 1:
                    continue
                result = guideTreeHelper(comp, (origin[0], origin[1] + round(division * i)), width,
                                         round(division * (i + 1)) - round(division * i),
                                         pixelDict, orient=orient, pixelMovements=pixelMovements,
                                         needCalc=needCalc, depth=depth, codeMap=codeMap,
                                         calcSeq=calcSeq, precision=precision, moveHist=moveHist,
                                         bigWidth=bigWidth, bigHeight=bigHeight)
                if len(result) == 3:
                    newMoveHist |= result[-1]
        else:  # horizontal alignment
            division = width / len(tree[1:])
            for i, comp in enumerate(tree[1:]):
                if len(comp) == 1:
                    continue
                result = guideTreeHelper(comp, (origin[0] + round(division * i), origin[1]),
                                         round(division * (i + 1)) - round(division * i), height,
                                         pixelDict, orient=orient, pixelMovements=pixelMovements,
                                         needCalc=needCalc, depth=depth, codeMap=codeMap,
                                         calcSeq=calcSeq, precision=precision, moveHist=moveHist,
                                         bigWidth=bigWidth, bigHeight=bigHeight)
                if not needCalc:
                    newMoveHist |= result[-1]
        if not needCalc:
            result = result[0], result[1], newMoveHist
        # print("series result", tree, needCalc, result)
        return result
    elif needCalc and tree[0] in reversedCharMapDyn and tree[0] not in ['ż', 'm', 'ċ']:
        # is a dynamic element and has not calculated the movements yet
        orient = tree[1]  # accent
        if orient == 1:  # vertical
            division = height / 2
        else:  # horizontal
            division = width / 2

        dynamicPattern = vecMap[reversedHexMap[reversedCharMapDyn[tree[0]]]][0]
        movements = []
        for i in range(2):
            movements.append(dynamicPattern[i])
            if tree[0] == 'p' and abs(movements[-1]) > division / 2:  # movement boundary
                movements[-1] = math.ceil(division / 2) * math.copysign(1, movements[-1])

        for i in range(2):
            for k in range(round(division * i), round(division * (i + 1))):
                if orient == 1:
                    for j in range(width):
                        pixelMovements[origin[1] + k][origin[0] + j][0] += int(movements[i])
                else:
                    for j in range(height):
                        pixelMovements[origin[1] + j][origin[0] + k][1] += int(movements[i])
        # print('new pixel movement', tree, pixelMovements)
    elif not needCalc:
        if isPair(tree[-1]):
            mapLegend = tree[-1][1]
            tree = tree[:-1]
        else:
            mapLegend = 0
        # print('origin', origin, height, width, tree)
        newPtLst = {}
        for j in range(height):
            for k in range(width):
                if tree[0] in ['ż', 'm', 'ċ'] or tree[0] in reversedCharMap:
                    startPt = (origin[0] + k, origin[1] + j)

                    if moveHist is not None and (origin[0] + k, origin[1] + j) in moveHist:
                        closestValid = -1
                        for frameI in moveHist[(origin[0] + k, origin[1] + j)][0].keys():
                            if closestValid < frameI < calcSeq[0]:
                                closestValid = frameI
                        if closestValid == -1:
                            lastPt = startPt
                            closestValid = calcSeq[0]
                        else:
                            # print("closest valid", closestValid, moveHist[(origin[0] + k, origin[1] + j)])
                            lastPt = (moveHist[(origin[0] + k, origin[1] + j)][0][closestValid],
                                      moveHist[(origin[0] + k, origin[1] + j)][1][closestValid])
                    else:
                        lastPt = startPt
                        closestValid = calcSeq[0]

                    XLst, YLst = approxPos(lastPt, pixelMovements, calcSeq, lastStep=closestValid)
                    # print("cur calc seq", calcSeq)
                    if startPt not in newPtLst:
                        newPtLst[startPt] = (XLst, YLst)
                    else:
                        print("already exists", startPt)
                    for frameI in calcSeq:
                        if tree[0] in ['ż', 'm', 'ċ']:
                            fluctuation = np.random.rand() * reversedCharMapDyn[tree[0]][0] * np.random.choice([1, -1])
                            if 0 <= YLst[frameI] < bigHeight and 0 <= XLst[frameI] < bigWidth:
                                pixelDict[frameI][YLst[frameI]][XLst[frameI]][0] += fluctuation
                        if tree[0] not in ['h'] + list(reversedCharMapDyn.keys()):
                            if 0 <= YLst[frameI] < bigHeight and 0 <= XLst[frameI] < bigWidth:
                                pixelDict[frameI][round(YLst[frameI])][round(XLst[frameI])][1] += 1
                        # print('background', tree, frameI, YLst[frameI], XLst[frameI], pixelDict[frameI][YLst[frameI]][XLst[frameI]])
                if codeMap is not None:
                    if mapLegend != 0:
                        codeMap[origin[1] + j][origin[0] + k].append(mapLegend)
        newMoveHist = newPtLst
        if tree[0] not in ['ż', 'm', 'ċ'] and tree[0] in reversedCharMap:
            orient = tree[1]  # accent
            if orient == 1:  # vertical
                division = height / 3
            else:  # horizontal
                division = width / 3
            staticPattern = reversedCharMapDbl[tree[0]]
            for i in range(3):
                if staticPattern[i] == 0:
                    continue
                for k in range(round(division * i), round(division * (i + 1))):
                    if orient == 1:
                        for j in range(width):
                            startPt = (origin[0] + j, origin[1] + k)
                            XLst, YLst = newPtLst[startPt]
                            # print(j, XLst, YLst)
                            for frameI in calcSeq:
                                if 0 <= YLst[frameI] < bigHeight and \
                                        0 <= XLst[frameI] < bigWidth:
                                    pixelDict[frameI][round(YLst[frameI])][round(XLst[frameI])][0] += staticPattern[i]

                                    # print("figure", tree, frameI, YLst[frameI], XLst[frameI], pixelDict[frameI][YLst[frameI]][XLst[frameI]])
                    else:
                        for j in range(height):
                            startPt = (origin[0] + k, origin[1] + j)
                            XLst, YLst = newPtLst[startPt]
                            for frameI in calcSeq:
                                if 0 <= YLst[frameI] < bigHeight and \
                                        0 <= XLst[frameI] < bigWidth:
                                    # print(moveTup)
                                    pixelDict[frameI][round(YLst[frameI])][round(XLst[frameI])][0] += staticPattern[i]
                                    # print("figure", tree, frameI, YLst[frameI], XLst[frameI], pixelDict[frameI][YLst[frameI]][XLst[frameI]])

    if needCalc:
        return pixelMovements
    else:
        return pixelDict, codeMap, newMoveHist


def approxPos(lastPt, pixelMovements, calcSeq, lastStep):  # Euler's method
    newX, newY = lastPt
    # print("new x", newX, "new y", newY)
    if pixelMovements is not None:
        XLst = {}  # {moment t -> x at t}
        YLst = {}  # {moment t -> y at t}

        for i, step in enumerate(calcSeq):
            XLst[lastStep] = newX
            YLst[lastStep] = newY
            exceeds = detectExcess(newX, newY, len(pixelMovements[0]) - 1, len(pixelMovements) - 1)
            if exceeds:
                lastStep = step
                continue
            moveTup = pixelMovements[round(newY)][round(newX)]
            # print("move tup", moveTup, newX, newY, step, lastStep)
            deltaNewX = moveTup[1] * (step - lastStep)
            deltaNewY = moveTup[0] * (step - lastStep)
            #if newX + deltaNewX == 0 and deltaNewX != 0 or newY + deltaNewY == 0 and deltaNewY != 0:
            #    print("warning", newX, deltaNewX, newY, deltaNewY, moveTup, step, lastStep)
            newX += deltaNewX
            newY += deltaNewY
            lastStep = step
        # newX, newY, exceeds = detectExcess(newX, newY, len(pixelMovements[0]) - 1, len(pixelMovements) - 1)
        XLst[lastStep] = newX
        YLst[lastStep] = newY
    else:
        XLst = {}  # {moment t -> x at t}
        YLst = {}  # {moment t -> y at t}
        for i, step in enumerate(calcSeq):
            XLst[step] = newX
            YLst[step] = newY
    return XLst, YLst


def drawLine(x1, y1, x2, y2):
    ptLst = []
    # Calculate the differences between the endpoints
    dx = x2 - x1
    dy = y2 - y1
    # Determine the sign of the differences
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)
    # Initialize error term
    error = dx - dy
    # Initialize current position
    x = x1
    y = y1
    # Iterate over the line
    while True:
        # Plot the pixel at (x, y)
        ptLst.append((x, y))

        if x == x2 and y == y2:
            break

        # Calculate the next position
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x += sx
        if e2 < dx:
            error += dx
            y += sy
    return ptLst


def detectExcess(newX, newY, boundX, boundY):
    YExceeds = newY > boundY or newY < 0
    XExceeds = newX > boundX or newX < 0
    return YExceeds or XExceeds


def readVideo(name, jump=1):
    video = cv2.VideoCapture('./images/videos/' + name + '.mp4')
    success, image = video.read()
    count = 0
    while success:
        cv2.imwrite("./images/frames/frame %d.png" % count, image)  # save frame as JPEG file
        for i in range(jump):
            success, image = video.read()
        print('Read a new frame: ', success)
        count += jump
    print("Finished reading the video.", count, "frames in total.")

def readGif(name, jump=1):
    gif = imageio.get_reader('./images/videos/' + name + '.gif')
    for count in range(0, len(gif), jump):
        imageio.imsave("./images/frames/frame %d.png" % count, gif.get_data(count))
    print("Finished reading the gif.", len(gif), "frames in total.")

def frameDif(pixels1, pixels2):
    difPixels = []
    for i in range(len(pixels1)):  # iterates through y
        difPixels.append([])
        for j in range(len(pixels1[0])):  # iterates through x
            val = (pixels2[i][j] - pixels1[i][j] + 255) / 2
            # print(pixels2[i][j], pixels1[i][j])
            difPixels[i].append(val)
    return difPixels


def initAxes(showMode=[True, False, True], numAxes=1):  # image, zero layer tree, complete tree
    global axes
    global fig
    fig = plt.figure()
    axes = []
    numCol = int(showMode[0]) + int(showMode[1]) * 2 + int(showMode[2])
    sumMode = sum([int(mode) for mode in showMode])
    widthRatios = [1 / sumMode / 3] * int(showMode[0]) + [0.5 / sumMode / 2] * 2 * int(showMode[1]) + [1 / sumMode] * int(
        showMode[2])
    realNumAxes = numAxes + 1 if showMode[1] else numAxes  # add the buttons below
    heightRatios = [0.9 / (realNumAxes - 1)] * (realNumAxes - 1) + [0.1] if showMode[1] else [1 / numAxes] * numAxes
    gs = fig.add_gridspec(realNumAxes, numCol, height_ratios=heightRatios, width_ratios=widthRatios)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    curCol = 0
    if showMode[0]:
        for i in range(numAxes):
            textImgPair = gs[i, curCol].subgridspec(2, 1, height_ratios=[0.2, 0.8])
            axes.append(fig.add_subplot(textImgPair[0]))  # title
            axes.append(fig.add_subplot(textImgPair[1]))  # image
        curCol += 1
    if showMode[1]:
        axes.append(fig.add_subplot(gs[:-1, curCol:curCol + 2]))
        axes.append(fig.add_subplot(gs[-1, curCol], ))  # button 1
        axes.append(fig.add_subplot(gs[-1, curCol + 1]))  # button 2
        curCol += 2
    if showMode[2]:
        axes.append(fig.add_subplot(gs[:, curCol]))
    return fig


def initSourceAxes(showMode):
    global axes
    global fig
    fig = plt.figure()
    axes = []
    print("source show mode", showMode)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 0.1])
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    upper = gs[0, 0].subgridspec(1, 2, width_ratios=[0.3, 0.7]) if showMode[0] and showMode[2] else None
    if showMode[0]:
        if upper is None:
            upperLeft = gs[0, 0].subgridspec(3, 1, height_ratios=[0.1, 0.8, 0.1])
        else:
            upperLeft = upper[0].subgridspec(3, 1, height_ratios=[0.1, 0.8, 0.1])
        axes.append(fig.add_subplot(upperLeft[0]))
        axes.append(fig.add_subplot(upperLeft[1]))
    if showMode[2]:
        if upper is None:
            upperRight = gs[0, 0].subgridspec(2, 1, height_ratios=[0.9, 0.1])
        else:
            upperRight = upper[1].subgridspec(2, 1, height_ratios=[0.9, 0.1])
        axes.append(fig.add_subplot(upperRight[0]))
        upperRightButtons = upperRight[1].subgridspec(1, 2, width_ratios=[0.5, 0.5])
        axes.append(fig.add_subplot(upperRightButtons[0]))
        axes.append(fig.add_subplot(upperRightButtons[1]))

    lower = gs[1, 0].subgridspec(1, 3, width_ratios=[0.3, 0.4, 0.3])
    axes.append(fig.add_subplot(lower[0]))
    axes.append(fig.add_subplot(lower[1]))
    axes.append(fig.add_subplot(lower[2]))

    return fig


# Function to update the animation at each frame
def updateFrame(frame, frames, numAxes):
    axes[numAxes + 1].clear()  # Clear previous frame's content
    axes[numAxes + 1].imshow(frames[frame], cmap='gray', vmin=0, vmax=255)  # Display new frame's content
    axes[numAxes + 1].axis('on')
    axes[numAxes + 1].tick_params(left=False, right=False, labelleft=False, labelbottom=False)
    return [axes[numAxes], axes[numAxes + 1]]


def playFrames(frames, numAxes, maxLayer):
    global fig
    # Create the animation
    # print('len frames', len(frames))
    #for frame in frames:
    #    axes[numAxes].imshow(frame)
    axes[numAxes].axis("off")
    axes[numAxes].text(0.5, 0.5, "Simulated Layer " + str(int(maxLayer - numAxes / 2)), ha='center', va='center', fontsize=10, family='sans-serif')
    ani = animation.FuncAnimation(fig, updateFrame, frames=len(frames), fargs=(frames, numAxes),
                                  init_func=None,
                                  blit=True, interval=20)

    return ani


def calcLayered(guideTrees, normalize=True, layerMode=True, keyLen=20, reverse=True, needZero=False, verbose=False,
                fillDyn=True, acceleration=1, mirrorMode=True, precision=5):
    global unknownTree
    if guideTrees[0] == 'parallel':
        guideTrees = [guideTrees]
    channelVideos = []
    zeroLayerTrees = []
    for guideTree in guideTrees:  # layered videos from each RGB channel
        if reverse:
            guideTree = reverseGuideTree(guideTree)
        layerLst = getAllLayers(guideTree)
        if needZero:  # optimize layer calculation by omitting incontinuous layers
            layerSetDif = set(range(min(layerLst), max(layerLst) + 1)).difference(set(layerLst))
            if layerSetDif == set():
                minLayer = min(layerLst)
                maxLayer = max(layerLst) + 1
            else:
                negLayers = {val for val in layerSetDif if val < 0}
                posLayers = {val for val in layerSetDif if val > 0}
                minLayer = max(negLayers) + 1 if negLayers else min(layerLst)
                maxLayer = min(posLayers) if posLayers else max(layerLst) + 1
                if maxLayer == 0:
                    maxLayer = 1
            layerLst = list(range(minLayer, maxLayer))
        else:
            minLayer = min(layerLst)
            maxLayer = max(layerLst) + 1
        # print("min layer", minLayer, 'max', maxLayer)
        layerTrees = {}
        for layer in range(minLayer, maxLayer):
            if -layer not in layerTrees.keys():
                if fillDyn:
                    filledTree = fillDynamic(guideTree)[0]
                    if verbose:
                        print('Filled Tree', filledTree)
                else:
                    filledTree = copy.deepcopy(guideTree)
                layerTrees[layer] = \
                    filterLayer(filledTree, layer, codeMode=True, mirrorMode=mirrorMode, verbose=verbose)[0]
                if verbose:
                    print('Layer', layer, 'Tree', layerTrees[layer])
                if not cleanGuideTree(layerTrees[layer]):
                    layerTrees[layer] = unknownTree
            else:
                layerTrees[layer] = layerTrees[-layer]

        probDictCol = {}  # a collection of prob dicts across every layer
        lastProbDicts = []
        videos = {}
        # frameDiv = round(max([countBranch(layerTrees[layer]) for layer in range(minLayer, maxLayer)]))
        frameDiv = keyLen
        moveHistDict = {}
        if needZero:
            zeroLayerTrees = []
        for frameStart in range(frameDiv):  # update mirror layer components in every frame division
            for layer in range(minLayer, maxLayer):
                if layer - 1 in probDictCol.keys():
                    lastProbDicts = probDictCol[layer - 1]
                    # print('layer-1', lastProbDicts)
                    if layer != 0 and -layer - 1 in probDictCol.keys():
                        # print('merge layers', lastProbDicts, probDictCol[-layer - 1], layer)
                        pairCol = [lastProbDicts, probDictCol[-layer - 1]]
                        longCol = pairCol[np.argmax([len(col) for col in pairCol])]
                        lastProbDicts = longCol
                        pairCol.remove(longCol)
                        shortCol = pairCol[0]
                        for i, probDict in enumerate(longCol):
                            lastProbDicts[i] |= shortCol[int(i / len(longCol) * len(shortCol))]
                        # lastProbDicts = list(map(lambda x, y: x | y, *zip(lastProbDicts, probDictCol[-layer - 1])))
                elif layer != 0 and -layer - 1 in probDictCol.keys():
                    # print('-layer-1', lastProbDicts)
                    lastProbDicts = probDictCol[-layer - 1]
                # print('lastProbDicts', lastProbDicts, layer)
                video, probDicts, guideTreeLst, newMoveHistDict = drawGuideTree(layerTrees[layer], precision=precision,
                                                                                lastProbDicts=lastProbDicts,
                                                                                layerMode=layerMode,
                                                                                normalize=normalize,
                                                                                displayTree=layer == 0 and needZero,
                                                                                hollowDyn=fillDyn,
                                                                                verbose=verbose, acceleration=acceleration,
                                                                                frameStart=frameStart,
                                                                                frameDiv=frameDiv,
                                                                                moveHistDict=moveHistDict,
                                                                                reverse=reverse)
                moveHistDict |= newMoveHistDict
                if layer == 0 and needZero:
                    zeroLayerTrees += [['parallel', cleanGuideTree(tree)] for tree in guideTreeLst]
                    # zeroLayerTrees += guideTreeLst
                try:
                    videos[layer] += video
                except KeyError:
                    videos[layer] = video
                probDictCol[layer] = probDicts

        for layer in range(minLayer, maxLayer):
            lenRatio = frameDiv * precision / len(videos[layer])
            for i in range(len(videos[layer]) - 1, -1, -1):
                videos[layer] = videos[layer][:i] + \
                                [videos[layer][i]] * (round(lenRatio * (i + 1)) - round(lenRatio * i)) + \
                                videos[layer][i + 1:]
        videos = list(videos.values())
        channelVideos.append(videos)
    if len(guideTrees) == 3:
        mergedVideos = []
        for i in range(len(channelVideos[0])):  # layer
            mergedVideo = []
            for j in range(len(channelVideos[0][i])):  # frame
                mergedVideo.append(
                    Image.merge("RGB", (channelVideos[0][i][j], channelVideos[1][i][j], channelVideos[2][i][j])))
            mergedVideos.append(mergedVideo)
        videos = mergedVideos
    if needZero:
        if not zeroLayerTrees:
            zeroLayerTrees = [unknownTree] * keyLen
        return videos, layerLst, zeroLayerTrees
    else:
        return videos, layerLst


def initButtons(zeroGraphs, startAx, replaceText=True, videos=None, useSymbols=False, isSource=False):
    global axPrev
    global axNext
    global buttonPrevTree
    global buttonNextTree
    global buttonPrevSeq
    global buttonNextSeq
    global curGalleryInd
    global curSeqInd
    global numGallery
    global axes
    curGalleryInd = 0
    curSeqInd = 0
    if zeroGraphs is not None:
        if isinstance(zeroGraphs[0], list):
            numSeq = len(zeroGraphs)
            numGallery = len(zeroGraphs[0])
        else:
            numGallery = len(zeroGraphs)
    else:
        numSeq = len(videos)
    arrowPrevImg = imread('./images/arrowPrev.png')  # Replace with your arrow image file
    arrowNextImg = imread('./images/arrowNext.png')

    if not isSource or zeroGraphs is not None:
        buttonPrevTree = Button(axes[startAx + 1], '', image=arrowPrevImg)
        buttonNextTree = Button(axes[startAx + 2], '', image=arrowNextImg)
        buttonPrevTree.on_clicked(lambda event: updateGallery((curGalleryInd - 1) % numGallery, zeroGraphs, startAx,
                                                          replaceText=replaceText, useSymbols=useSymbols))
        buttonNextTree.on_clicked(lambda event: updateGallery((curGalleryInd + 1) % numGallery, zeroGraphs, startAx,
                                                          replaceText=replaceText, useSymbols=useSymbols))
    if isSource:
        buttonPrevSeq = Button(axes[startAx + 3], '', image=arrowPrevImg)
        axes[startAx + 4].axis("off")
        axes[startAx + 4].text(0.5, 0.5, 'Sequence ' + str(curSeqInd + 1), ha='center', va='center', fontsize=12)
        buttonNextSeq = Button(axes[startAx + 5], '', image=arrowNextImg)

        buttonPrevSeq.on_clicked(lambda event: updateGallery((curSeqInd - 1) % numSeq, zeroGraphs, startAx,
                                                          replaceText=replaceText, videos=videos, useSymbols=useSymbols, changeSeq=True))
        buttonNextSeq.on_clicked(lambda event: updateGallery((curSeqInd + 1) % numSeq, zeroGraphs, startAx,
                                                          replaceText=replaceText, videos=videos, useSymbols=useSymbols, changeSeq=True))


def updateGallery(index, zeroGraphs, startAx, replaceText=True, videos=None, useSymbols=False, changeSeq=False):
    global axes
    global curGalleryInd
    global curSeqInd
    global ani
    global textProps
    if changeSeq:
        curSeqInd = index
        axes[startAx + 4].cla()
        axes[startAx + 4].axis("off")
        axes[startAx + 4].text(0.5, 0.5, 'Sequence ' + str(curSeqInd + 1), ha='center', va='center', fontsize=12)
    else:
        curGalleryInd = index
    if zeroGraphs is not None:
        axes[startAx].cla()
        if isinstance(zeroGraphs[0], list):
            drawNetwork(zeroGraphs[curSeqInd][curGalleryInd], startAx, replaceText, useSymbols)
            axes[startAx].set_title('Zeroth Layer ' + str(curGalleryInd + 1) + '/' + str(len(zeroGraphs[0])))
        else:
            drawNetwork(zeroGraphs[curGalleryInd], startAx, replaceText, useSymbols)
            axes[startAx].set_title('Zeroth Layer ' + str(curGalleryInd + 1) + '/' + str(len(zeroGraphs)))
    if videos is not None:
        axes[0].clear()
        axes[0].cla()  # Clear the current axes
        ani[0] = None
        ani[0] = playFrames(videos[curSeqInd], 0, 0)
    plt.ion()


def drawNetwork(graph, startAx, replaceText, useSymbols):
    global axes
    labels, colorMap, pos, fontFamily = decorateNetwork(graph, replaceText=replaceText, useSymbols=useSymbols)

    # nx.draw_networkx_nodes(graph, pos, ax=axes[startAx], node_size=1000, node_color='white', edgecolors=colorMap)
    nx.draw_networkx_edges(graph, pos, ax=axes[startAx])
    for tup, labelData in labels.items():
        isStatic = tup[1][0] in reversedCharMap if isinstance(tup[1], tuple) else False
        unique = tup[0]
        fontColor = 'white' if isStatic else 'black'
        if isinstance(labelData, dict):
            label = labelData['label']
            rotation = labelData['rotation']
            ha = labelData['ha']
            va = labelData['va']
            axes[startAx].text(pos[unique][0], pos[unique][1], label, rotation=rotation, ha=ha, va=va, size=12,
                               color=fontColor, weight='bold', family=fontFamily[unique])
        else:
            label = labelData
            nx.draw_networkx_labels(graph, pos, ax=axes[startAx], labels={unique: label}, font_size=12,
                                    font_color=fontColor, font_weight='bold', font_family=fontFamily[unique])
        if isStatic:
            nx.draw_networkx_nodes([unique], pos, ax=axes[startAx], node_size=1000, node_color=colorMap[unique],
                                   edgecolors='white')
        else:
            nx.draw_networkx_nodes([unique], pos, ax=axes[startAx], node_size=1000, node_color='white',
                                   edgecolors=colorMap[unique])


def decorateNetwork(graph, replaceText=True, useSymbols=False):
    global fontNames
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")  # Using Spectral Layout
    colorMap = {}
    fontFamily = {}
    labels = {}
    label2Color = {0: 'mediumseagreen', 1: 'coral', 2: 'crimson', -1: 'blueviolet', -2: 'darkslategrey'}
    for tup in graph.nodes(data=True):
        unique = tup[0]  # unique node name
        if isinstance(tup[1]['label'], list):
            tup = (unique, lst2tup(tup[1]['label']))
        else:
            tup = (unique, tup[1]['label'])
        if replaceText:
            layer = None
            if isinstance(tup[1], tuple):
                layer = tup[1][2]
            if layer is not None:
                colorMap[unique] = mcolors.CSS4_COLORS[label2Color[layer]]
            else:
                colorMap[unique] = mcolors.CSS4_COLORS['dimgray']

            if not isinstance(tup[1], list) and tup[1] in replaceDict:
                if useSymbols:
                    fontFamily[unique] = fontNames[1]
                    labels[tup] = '//' if tup[1] == nodeSymbols[0] else '—'
                else:
                    fontFamily[unique] = fontNames[0]
                    labels[tup] = replaceDict[tup[1]]
            elif tup[1][0] in leafSymbols:
                if useSymbols:
                    labels[tup] = {
                        'label': reversedSymMap[reversedCharMapDbl[tup[1][0]]][
                            0],
                        'rotation': 90 * (1 - tup[1][1]), 'ha': 'center', 'va': 'center'}
                    '''
                    labels[tup] += (tup[1][0] in reversedCharMap) * "/" \
                                      + (tup[1][0] in reversedCharMapDyn) * "\\"
                    labels[tup] += dimMap[1 - tup[1][1]]
                    '''
                    fontFamily[unique] = fontNames[1]
                else:
                    labels[tup] = \
                        reversedHexMap[reversedCharMapDbl[tup[1][0]]][
                            0]
                    '''
                    labels[tup] += (tup[1][0] in reversedCharMap) * '靜' \
                                      + (tup[1][0] in reversedCharMapDyn) * '動'
                    '''
                    labels[tup] += dimMap[1 - tup[1][1]]
                    fontFamily[unique] = fontNames[0]
        else:
            colorMap[unique] = 'white'
            fontFamily[unique] = fontNames[0]
            labels[tup] = tup[1]
    return labels, colorMap, pos, fontFamily


def playLayered(videos, layerLst=[0], showMode=[True, False, True], isSource=False, mirrorMode=False):
    global axes
    global ani
    global fig
    layerLst = sorted(set(layerLst))
    if mirrorMode:
        layerLst = [val for val in layerLst if val <= 0] if any([val for val in layerLst if val <= 0]) else layerLst
    if isSource:
        fig = initSourceAxes(showMode=showMode)
    else:
        fig = initAxes(showMode=showMode, numAxes=max(layerLst) + 1 - min(layerLst))

    if showMode[0]:
        layerNum = max(layerLst) + 1 - min(layerLst)
        ani = []
        for i in range(layerNum):
            ani.append(None)
        for i in range(layerNum):
            # Create an empty image plot for the first animation
            # im1 = gridAx.imshow(videos[layer][0], animated=True)
            # print(mirrorMode, layerNum, layerLst, layerNum - 1 - i, ani, len(videos))
            for j, frame in enumerate(videos[i]):
                frame.save('./images/results/simulations/layer ' + str(layerLst[i]) + ' - ' + str(j) + '.png')
            gifPath = './images/results/simulations/layer ' + str(layerLst[i]) + '.gif'
            videos[i][0].save(gifPath, save_all=True, append_images=videos[i][1:], loop=0, duration=150)
            ani[layerNum - 1 - i] = playFrames(videos[i], (layerNum - 1 - i) * 2, max(layerLst))


def flattenLists(nestedLst):
    flattened = []

    for item in nestedLst:
        if isinstance(item, list):
            flattened.extend(flattenLists(item))
        else:
            flattened.append(item)

    return flattened


def main():
    global axes
    global ani
    txt = '''
    Ưrah rơm, izucun. Ucunizaṡ, ĩhemõṡũl. Itaé gớt, izupun. Upunizág, ĩhemõṡũl. Ưtar ịtát, íábipaé ghom. Ĩrágót, itatư bah ésehomoż. Ưbẽhonós. Ưżuṡazailian. Uṡaz: Upun atư bah, édhom éctaupuzużuċul. Éṡb ẻtớtaulupużuzuluċul, atifẽ hơm ịfusus. Ipáde żơt, rarmoż, mơt mơc ịfẽ hơm. Ágizud, ĩmabon? Ibe mfiáhailiág? Ibzaé gơc ịpr. Hơm ịzásiláṡ, izátilaé dơṡ. Bớs échom óv, izaribád, ré cớs ịzaé sơċ, ifẽ hmód ịbẽ ṡớd. Évhom ịzásiláṡ, izátilẻ bơṡ. Izulupużuzuluċul ifaé vehomód, itrfaé vhom. Ágé gud ịlár, itrẽ ruput bah. Brũt ịbư żágót ưzab ịbaz: Ĩram żớt, tde rớtũy ịbẽ rớtũf. Ĩbz olatditaé dơh óciṡ, ĩmz õtulupużuzuluċul itatư bah? Te gơh ịzư faṡot ogitac, mizaripaṡ. Tucututulu izabitẽ rơṡ ịrtuh. Tucużupuċulu izapṡirtufus. Ĩrulõr ĩmz õrz! Rraz ĩlgõ ṡraz, ritig lgõ ṡitig. Ĩmz õrzir? Ẽraremoṡũżuż ịlarzuṡibuh, ẽbrũhug ịlarzudibus, ĩrtigõr. Ưżah ịzubizé gnũmul, użuzutugizud, użuzutugizus; ĩcr ohgzẽ ṡerótũż, ucużupuċuċugizud, ucużupuċuċugizus. Ibrư puzal otrifẽ ṡitig, idabe mơṡ, ĩml ũfu!
    	'''
    # txt = "mơc omilammg, e'eṡmo'm, żlom e'ermo'mm, e'e'eṡimõ'mo'm, m'etżo' omibamĩ li'ẽdco'm, mơt mơr m, mmom, mtaitõraiz om, me srom, m'etmo' miz m'ĩbamo'o'ĩdamo' om, miz mơt ṡrapar, ze nơm ịtam, m'etmo', milẽ 'etmo'it og, e'etmo'itig ote rơm, e'etmo'itig ote rơm, m'etmomo' ịtĩ mơh og, mili'ẽ'etmo'itigo', mili'ẽ'etmmo'itigo', mm'etito' tơm, misẽ tơm, me 'ĩlai'iti'iro''ĩpai'iro' ẽtm, tidom ẽtmo, miċili'ẽ'etmo'itigo', e'emmoro'i'elṡo', m rĩ'i żait õde cơm, mizẽ mơp ịtam"

    txt = "éznịt hai'ih ĩneáboboż"
    # txt = "e'eházahrịto'i'ehano'i'etano'a'ébebożịho'"
    txt = "miṡõi'ic ẽmilõi'ih otah."
    # txt = "ĩmiatiamm"
    # txt = "ưcun'ĩbadago'"

    '''
    longGuideTree = ItiParser.read(txt, guideMode=True)[0]
    for guideTree in longGuideTree:
        print(guideTree)
        guideTree = ['parallel', guideTree]
        videos, layerLst = calcLayered(guideTree, normalize=True, reverse=False, verbose=True, keyLen=7, acceleration=2,
                                       fillDyn=True, mirrorMode=True, precision=5)
        playLayered(videos, layerLst, showMode=[True, False, False], mirrorMode=True)
        plt.show()
    '''

    import bayesianLearning as bayes
    # sourceTrees = demoBackups.periodicity
    sourceTrees = ['parallel', ['series', ['h', 1, -1]], ['series', ['h', 1, -1]], ['series', ['c', 1, -1], ['parallel', ['series', ['h', 0, -1]], ['series', ['c', 0, -1], ['parallel', ['series', ['c', 1, -1]], ['series', ['t', 0, -1]], ['series', ['c', 1, -1]], ['series', ['t', 0, -1]], ['series', ['h', 0, -1], ['h', 0, -1], ['ṡ', 0, -1], ['h', 0, -1], ['h', 0, -1]], ['series', ['h', 1, -1], ['h', 1, -1], ['ṡ', 1, -1], ['h', 1, -1], ['h', 1, -1]],  ['series', ['g', 0, 0]], ['series', ['r', 1, 0]], ['series', ['c', 1, 0]], ['series', ['c', 0, 0]]],  # state 1
                ['parallel', ['series', ['t', 1, -1]], ['series', ['t', 0, -1]], ['series', ['t', 1, -1]], ['series', ['t', 0, -1]], ['series', ['g', 1, 0]], ['series', ['r', 0, 0]], ['series', ['c', 1, 0]], ['series', ['t', 0, 0]]], ['t', 0, -1]]],  # state 4
                ['parallel', ['series', ['h', 0, -1]], ['series', ['c', 0, -1],
                  ['parallel', ['series', ['c', 1, -1]], ['series', ['c', 0, -1]], ['series', ['c', 1, -1]], ['series', ['c', 0, -1]], ['series', ['d', 1, 0]], ['series', ['r', 0, 0]], ['series', ['c', 0, 0]], ['series', ['t', 1, 0]]],  # state 2
                  ['parallel', ['series', ['t', 1, -1]], ['series', ['c', 0, -1]], ['series', ['t', 1, -1]], ['series', ['c', 0, -1]], ['series', ['d', 0, 0]], ['series', ['r', 1, 0]], ['series', ['t', 1, 0]], ['series', ['t', 0, 0]]], ['t', 0, -1]]], ['t', 1, -1]  # state 3
                                                 ]]
    sourceTrees = demoBackups.orOp
    #sourceTrees = demoBackups.revolve
    #sourceTrees = ['parallel', ['series', ['parallel', ['series', ['ż', 1, 0]], ['series', ['ċ', 0, 1]]], ['parallel', ['series', ['n', 0, 1]], ['series', ['n', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['v', 1, 0], ['g', 1, 1]], ['series', ['n', 1, 0], ['r', 0, 1]]]], ['series', ['l', 0, 1]]]

    #sourceTrees = ['parallel', ['series', ['parallel', ['series', ['l', 0, -1]], ['series', ['m', 0, 0]]]]]
    # sourceTrees = ['parallel', ['series', ['ṡ', 1, 0], ['ṡ', 1, 0], ['ṡ', 1, 0]], ['series', ['l', 0, 0]]]
    videos, layerLst, zeroLayerTrees = calcLayered(sourceTrees, normalize=True, reverse=False, needZero=True, keyLen=14, verbose=1 > 2, fillDyn=False, precision=2, acceleration=1)
    print("videos", videos)
    print("layer lst", layerLst)
    print("zero", zeroLayerTrees)
    graph = nx.DiGraph()
    labelCounter = {}
    nodeTree = bayes.convert2NodeTree(sourceTrees, labelCounter=labelCounter, graph=graph)
    zeroGraphs = []
    for i in range(len(zeroLayerTrees)):
        zeroGraphs.append(nx.DiGraph())
    testNodeCol = [bayes.convert2NodeTree(testTree, graph=zeroGraphs[i]) for i, testTree in enumerate(zeroLayerTrees)]
    bayes.showPlot(graph=graph, videos=videos, layerLst=layerLst, zeroGraphs=zeroGraphs, replaceText=True, mirrorMode=True, useSymbols=False)
    #plt.show()
    
    readGif('revolve', jump=2)

    pixels = None
    lastPixels = [None, None, None] if colorMode else [None]
    frameJump = 1
    imageSeq = []
    resultGuideTreeCollection = []
    for frameNum in range(0, 42, frameJump):
        # image = Image.open(r"./images/frames/umbrella.jpg")
        # image = Image.open(r"./images/frames/umbrella" + str(frameNum) + ".jpg")
        image = Image.open(r"./images/frames/layer -1 - " + str(frameNum) + ".png")
        # detectContour(image)
        # sourceImages = [detectEdge(image, useCanny=True)]
        # sourceImages = [Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY))]
        #sourceImages = extractChannels(image) if colorMode else [
        #    Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY))]
        # sourceImages = [image]
        sourceImages = [image.convert('L')]
        resultImages = []
        resultVideos = []
        resultGuideTrees = []
        for i, image in enumerate(sourceImages):
            if len(sourceImages) == 3:
                image.save('./images/results/' + ['red', 'green', 'blue'][i] + 'Image.png')
            # image.rotate(-90)
            pixels = list(image.getdata())
            width, height = image.size
            pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
            if staticMode:
                rays, tree = splitImage(pixels, (0, 0), bothOrient=True, flexible=True)
                print('rays', rays)
                print('tree', tree)
                treeImage = drawTree(tree)
                treeImage.save('./images/results/tree' + str(frameNum) + '-' + str(i) + '.png')
                resultImages.append(treeImage)
                scaleFactor = 30  # Change this to your desired scale factor
                image = image.resize((image.size[0] * scaleFactor, image.size[1] * scaleFactor))
                image = drawRays(image, rays, scaleFactor=scaleFactor)
                image = drawLabels(image, tree, allTrees=True, scaleFactor=scaleFactor)
                image.save(r'./images/results/result' + str(frameNum) + '-' + str(i) + '.png')
                guideTree = convert2GuideTree(tree)
                if guideTree[0] == 'series':
                    guideTree = ['parallel', guideTree]
                resultGuideTrees.append(guideTree)
            if dynamicMode:
                if lastPixels[i] is not None:
                    flow = detectFlow(np.array(lastPixels[i]), np.array(pixels))
                    rays, tree = splitImage(pixels, (0, 0), flow=flow, bothOrient=True, flexible=True, buddhist=False)
                    print('rays', rays)
                    print('tree', tree)
                    if not staticMode:
                        treeImage = drawTree(tree)
                        treeImage.save('./images/results/treeDyn' + str(frameNum) + '-' + str(i) + '.png')
                        resultImages.append(treeImage)
                    scaleFactor = 30  # Change this to your desired scale factor
                    image = image.resize((image.size[0] * scaleFactor, image.size[1] * scaleFactor))
                    image = visualizeFlow(image, flow, scaleFactor=scaleFactor)
                    image = drawRays(image, rays, scaleFactor=scaleFactor)
                    image = drawLabels(image, tree, allTrees=True, scaleFactor=scaleFactor)
                    image.save(r'./images/results/resultDyn' + str(frameNum) + '-' + str(i) + '.png')
                    guideTree = convert2GuideTree(tree)
                    if guideTree[0] == 'series':
                        guideTree = ['parallel', guideTree]
                    resultGuideTrees.append(guideTree)
                    video, probDicts, guideTreeLst, moveHistDict = drawGuideTree(guideTree, frameDiv=20)
                    video[0].save(r'./images/results/treeDynGuide' + str(frameNum) + '-' + str(i) + '.png')

                    resultVideos.append(video)
                lastPixels[i] = pixels
        if len(resultImages) == 3:
            mergedImage = Image.merge("RGB", (resultImages[0], resultImages[1], resultImages[2]))
            mergedImage.save('./images/results/mergedTree' + str(frameNum) + '.png')
            imageSeq.append(mergedImage)
        elif len(resultImages) == 1:
            imageSeq.append(resultImages[0])
        if len(resultVideos) == 3:
            mergedVideo = []
            for i in range(len(resultVideos[0])):
                mergedVideo.append(Image.merge("RGB", (resultVideos[0][i], resultVideos[1][i], resultVideos[2][i])))
            mergedVideo[0].save('./images/results/mergedVideo' + str(frameNum) + '.gif'
                                , save_all=True, append_images=mergedVideo[1:], loop=0, duration=100)
        print('Result guide trees', resultGuideTrees)
        resultGuideTreeCollection.append(resultGuideTrees)
    print('Result guide tree collection', resultGuideTreeCollection)  # attention: dynamic lacks the first guide tree
    imageSeq[0].save('./images/results/treeDynSeq.gif'
                         , save_all=True, append_images=imageSeq[1:], loop=0, duration=250)
    print(len(imageSeq))


if __name__ == "__main__":
    main()
