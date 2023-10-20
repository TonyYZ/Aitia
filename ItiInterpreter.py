from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2
import math


punishment = 3  # punishment factor to simple hexagrams (0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)
punishmentDyn = 1
hexMap = {'坤卦': (0, 0, 0), '艮卦': (1, 0, 0), '坎卦': (0, 1, 0), '巽卦': (1, 1, 0), '中卦': (0.5, 0.5, 0.5),
          '震卦': (0, 0, 1), '離卦': (1, 0, 1), '兌卦': (0, 1, 1), '乾卦': (1, 1, 1)}
reversedHexMap = {value: key for key, value in hexMap.items()}
hexMapDyn = {'坤卦': (0.5, 0.5, 0.5), '艮卦': (0, 1, 0.5), '坎卦': (0.75, 0, 0.75), '巽卦': (0.5, 0, 1),
             '中卦': (0.5, 0.5, 0.5), '震卦': (0.5, 1, 0), '離卦': (0.25, 1, 0.25), '兌卦': (1, 0, 0.5), '乾卦': (0.5, 0.5, 0.5)}

dimMap = {0: 'y', 1: 'x'}
staticMode = True
dynamicMode = False
colorA = (255, 0, 0)
colorB = (0, 200, 200)
totalScore = 0


def thresholdImage(image):
    thresh = 100
    fn = lambda x: 255 if x > thresh else 0
    image = image.convert("L").point(fn, mode='1')
    return image


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


def extractChannels(image):
    rgbImage = image.convert('RGB')
    redImage = rgbImage.getchannel('R')
    greenImage = rgbImage.getchannel('G')
    blueImage = rgbImage.getchannel('B')
    print(redImage, greenImage, blueImage)
    return redImage, greenImage, blueImage


def countDepth(tree):
    depth = 1
    comparedDepths = []
    for comp in tree[1:]:
        if comp[0] == 'parallel':  # is an element
            for branch in comp[1:]:
                comparedDepths.append(countDepth(branch))
    if comparedDepths:
        depth += max(comparedDepths)
    return depth

def getScore(pixelSums, maxSums, variances, hexName, examplei,
             verbose=False):  # how well the pixel values fit some trigram
    global punishment
    global punishmentDyn
    sensations = [0, 0, 0]
    for i in range(3):
        if maxSums[i] == 0:
            ratio = 0
        else:
            ratio = pixelSums[i] / maxSums[i]
        # print("ratio", ratio, pixelSums[i], maxSums[i])
        if verbose:
            print("ratio", ratio)
        if variances:
            k = 0.4
            if ratio < 0.5:
                if ratio < k:
                    sensations[i] = 0
                else:
                    sensations[i] = -math.sqrt(0.25 - 0.25 * math.pow(ratio - k, 2) / math.pow(0.5 - k, 2)) + 0.5
            else:
                if ratio > 1 - k:
                    sensations[i] = 1
                else:
                    sensations[i] = math.sqrt(0.25 - 0.25 * math.pow(ratio - 1 + k, 2) / math.pow(0.5 - k, 2)) + 0.5
        else:
            k = 0.21
            if ratio <= k:
                sensations[i] = math.sqrt(1 - math.pow(ratio - k, 2) / math.pow(k, 2))
            else:
                sensations[i] = 1
            # sensations[i] = ratio
    if variances:
        adjustedVariances = []
        maxAbsVal = 0
        # print(variances)
        for j in range(3):
            k = 6600
            if variances[j] < k:
                adjustedVariances.append(math.sqrt(1 - math.pow(variances[j] - k, 2) / math.pow(k, 2)))
            else:
                adjustedVariances.append(1)
            if abs(sensations[j] - 0.5) > maxAbsVal:
                maxAbsVal = abs(sensations[j] - 0.5)
        # print("maxAbsVal", maxAbsVal)
    iterable = hexMapDyn if variances else hexMap
    example = iterable[hexName]
    if variances:
        propDif = []
        varDif = []
    else:
        absDif = []
        avgSensations = sum(sensations) / len(sensations)
        avgExample = sum(example) / len(example)
        relSensations = []
        relExample = []
        relDif = []

    for j in range(3):
        if variances:
            if maxAbsVal == 0:
                normalizedSensation = 0
            else:
                normalizedSensation = (sensations[j] - 0.5) / maxAbsVal
            # print("normalizedSensation", j, normalizedSensation)
            propDif.append(abs(normalizedSensation - (example[j] - 0.5) * 2))
            varDif.append(abs(adjustedVariances[j] - list(hexMap.values())[examplei][j]))
        else:
            absDif.append(abs(sensations[j] - example[j]))
            relSensations.append(sensations[j] - avgSensations)
            relExample.append(example[j] - avgExample)
            relDif.append(math.pow(abs(relSensations[j] - relExample[j]), 1))

    if variances:
        propDif = sum(propDif)
        varDif = sum(varDif)
        score = propDif * 0.8 + varDif * 0.2
        if verbose:
            print(list(iterable.keys())[examplei], "Proportion", propDif, "Variance", varDif, "Total", score)
    else:
        absDif = sum(absDif)
        relDif = sum(relDif)
        score = absDif * 0.5 + relDif * 0.5
        if verbose:
            print(list(iterable.keys())[examplei], "Absolute", absDif, "Relative", relDif, "Total", score)

    if variances:
        '''
        midCount = example.count(0.5)
        if midCount > 0:
            score *= punishmentDyn * midCount
        '''
    else:
        if example[0] == example[1] and example[1] == example[2]:
            score *= punishment
        if verbose:
            print("Punished", score, list(iterable.keys())[examplei])
    return score, sensations


def getVariance(lst):
    lstSum = sum([math.pow(val - 127.5, 2) for val in lst])
    return lstSum / len(lst)


def exDir(c):
    if c in 'ṡsbdnvrgh':  # passive
        return 0
    elif c in 'żzptmflcċ':  # active
        return 1
    else:
        return None


def exHex(c):
    if c == 'ṡ':
        return '乾卦', False
    elif c == 's':
        return '兌卦', False
    elif c == 'b':
        return '離卦', False
    elif c == 'd':
        return '震卦', False
    elif c == 'n':
        return '中卦', False
    elif c == 'v':
        return '巽卦', False
    elif c == 'r':
        return '坎卦', False
    elif c == 'g':
        return '艮卦', False
    elif c == 'h':
        return '坤卦', False
    elif c == 'ż':
        return '乾卦', True
    elif c == 'z':
        return '兌卦', True
    elif c == 'p':
        return '離卦', True
    elif c == 't':
        return '震卦', True
    elif c == 'm':
        return '中卦', True
    elif c == 'f':
        return '巽卦', True
    elif c == 'l':
        return '坎卦', True
    elif c == 'c':
        return '艮卦', True
    elif c == 'ċ':
        return '坤卦', True
    else:
        return None


def analyzeElem(pixels, origin, guideTree, flexible=True):
    width = len(pixels[0])
    height = len(pixels)
    orient = 1 if guideTree[1] == 0 else 0
    hexName, dynamic = exHex(guideTree[0])
    global totalScore
    bestOffsets = [None, None]

    offsets = [0, 0]
    bestScore = 10
    print("Orientation:", orient, "Width:", width, "Height:", height)
    if orient == 0:  # tripartite using horizontal lines
        measureA = height
        measureB = width
    elif orient == 1:  # tripartite using vertical lines
        measureB = height
        measureA = width
        pixels = np.array(pixels).T.tolist()

    division = measureA / 3
    pixelSums = [0, 0, 0]
    maxSums = [0, 0, 0]
    variances = []
    # print([sum(pixels[pos]) for pos in range(measureA)])
    pixelLens = []
    for i in range(0, 3):
        intDivision = round(division * (i + 1)) - round(division * i)
        pixelLst = []
        for j in range(round(division * i), round(division * (i + 1))):
            pixelLst += pixels[j]
        if dynamic:
            variances.append(getVariance(pixelLst))
        pixelLens.append(len(pixelLst))
        pixelSums[i] = sum(pixelLst)
        maxSums[i] = 255 * intDivision * measureB

    results = getScore(pixelSums, maxSums, hexName=hexName, examplei=list(hexMap.keys()).index(hexName),
                       variances=variances,
                       verbose=True)
    score = results[0]
    sensations = results[1]

    bestPixelSums = pixelSums.copy()
    bestVariances = variances.copy()
    bestPixelLens = pixelLens.copy()
    bestMaxSums = maxSums.copy()
    bestSensations = sensations.copy()
    for i in range(0, 3):
        print("Sensation:", sensations[i], "Pixel sum:", pixelSums[i], "Max sum:", maxSums[i],
              "From", round(division * i), "to", round(division * (i + 1)))
        if dynamic:
            print("Variance:", variances[i])
    if round(division) > 0 and flexible:
        for i in [0, 1]:
            for moveDir, j in [(1, 0), (-1, 1)]:
                newPixelSums = pixelSums.copy()
                newVariances = variances.copy()
                newPixelLens = pixelLens.copy()
                newMaxSums = maxSums.copy()
                newOffsets = offsets.copy()
                newSensations = sensations.copy()
                while True:
                    position = round(division * (i + 1)) + newOffsets[i] + moveDir
                    if position < 4 or position >= measureA - 4:  # avoids leaving the image
                        print("avoids leaving the image", position, measureA)
                        break
                    pixelNum = sum(pixels[position])
                    deltaPixelSums = pixelNum * moveDir
                    deltaPixelLens = len(pixels[position]) * moveDir
                    if newPixelSums[i] + deltaPixelSums <= 0 or newPixelSums[i + 1] - deltaPixelSums <= 0 \
                            or newPixelLens[i] + deltaPixelLens <= 0 or newPixelLens[i + 1] - deltaPixelLens <= 0:
                        # avoids squeezing the middle space or creating empty space
                        print("avoids squeezing & empty", newPixelSums[i], newPixelSums[i + 1], deltaPixelSums)
                        break
                    newOffsets[i] += moveDir
                    oldPixelLens = newPixelLens.copy()
                    for k in [0, 1]:
                        addSign = 1 if k == 0 else -1
                        newPixelSums[i + k] += deltaPixelSums * addSign
                        newPixelLens[i + k] += deltaPixelLens * addSign
                        deltaMaxSums = 255 * measureB * moveDir * addSign
                        newMaxSums[i + k] += deltaMaxSums
                        if dynamic:
                            newVariances[i + k] *= oldPixelLens[i + k]
                            newVariances[i + k] = round(newVariances[i + k], 2)
                            newVariances[i + k] += sum(
                                [math.pow(val - 127.5, 2) for val in pixels[position]]) * moveDir * addSign
                            newVariances[i + k] /= newPixelLens[i + k]
                            '''
                            blendVariance = getVariance(pixels[position])

                            newVariances[i + k] = (newVariances[i + k] * oldPixelLens[i + k]
                                                   + blendVariance * len(pixels[position]) * moveDir * addSign) \
                                                   / newPixelLens[i + k]
                            print("new new", newVariances[i + k])
                            '''
                            # return sum([math.pow(val - 127.5, 2) for val in lst]) / len(lst)
                    results = getScore(newPixelSums, newMaxSums, hexName=hexName,
                                       examplei=list(hexMap.keys()).index(hexName), variances=newVariances)
                    newScore = results[0]
                    newSensations = results[1]
                    '''
                    print("Captured pixels:", pixelNum, "Sensations", newSensations, 'Variances', newVariances,
                          "Pos:", position,
                          "Line:", i, "Dir:", moveDir, "Offsets", newOffsets, "New score", newScore)
                    '''
                    # print("subtle dif", score - newScore)
                    if newScore >= score - 0.001:
                        break
                    score = newScore
                print("Current score:", score, "Best score", bestScore)
                if score < bestScore:
                    print("Best score", score, "Hexagram", hexName, "Sensations", newSensations, "Offsets",
                          newOffsets)
                    bestScore = score
                    bestPixelSums = newPixelSums
                    bestVariances = newVariances
                    bestPixelLens = newPixelLens
                    bestMaxSums = newMaxSums
                    bestSensations = newSensations
                    bestOffsets[orient] = newOffsets
            pixelSums = bestPixelSums
            variances = bestVariances
            pixelLens = bestPixelLens
            maxSums = bestMaxSums
            sensations = bestSensations
            offsets = bestOffsets[orient]
    else:
        bestOffsets[orient] = [0, 0]

    rays = []
    dynamicName = '動態' if dynamic else '靜態'
    if orient == 0:
        rays += [[(origin[0], origin[1] + height / 3 * i + bestOffsets[orient][i - 1]),
                  (origin[0] + width, origin[1] + height / 3 * i + bestOffsets[orient][i - 1])]
                 for i in [1, 2]]
        label = [(origin[0] + width / 2, origin[1] + height / 2), hexName + dynamicName, orient]
    else:
        rays += [[(origin[0] + width / 3 * i + bestOffsets[orient][i - 1], origin[1]),
                  (origin[0] + width / 3 * i + bestOffsets[orient][i - 1], origin[1] + height)] for i
                 in [1, 2]]
        label = [(origin[0] + width / 2, origin[1] + height / 2), hexName + dynamicName, orient]
    totalScore += bestScore

    return rays, label


def analyzeSeries(pixels, origin, guideTree, orient=1, flexible=False):
    compNum = len(guideTree[1:])  # the number of components connected in series
    width = len(pixels[0])
    height = len(pixels)
    if orient == 0:  # tripartite using horizontal lines
        measureA = height
    elif orient == 1:  # tripartite using vertical lines
        measureA = width

    tPixels = np.array(pixels).T.tolist()
    division = measureA / compNum

    rayLst = []
    labelLst = []
    for i in range(0, compNum):
        if orient == 0:
            subPixels = pixels[round(division * i):round(division * (i + 1))]
            subOrigin = (origin[0], origin[1] + height / compNum * i)
            rayLst.append([subOrigin, (subOrigin[0] + width, subOrigin[1])])
        elif orient == 1:
            subPixels = np.array(tPixels[round(division * i):round(division * (i + 1))]).T.tolist()
            subOrigin = (origin[0] + width / compNum * i, origin[1])
            rayLst.append([subOrigin, (subOrigin[0], subOrigin[1] + height)])
        if guideTree[1 + i][0] == 'parallel':
            rays, labels = splitImage(subPixels, subOrigin, guideTree[1 + i], orient=orient, flexible=flexible)
            rayLst += rays
            labelLst += labels
        else:
            rays, label = analyzeElem(subPixels, subOrigin, guideTree[1 + i], flexible=flexible)
            rayLst += rays
            labelLst.append(label)

    return rayLst, labelLst


def splitImage(pixels, origin, guideTree, orient=1, flexible=True):
    rayLst = []
    labelLst = []
    alterOrient = orient
    if guideTree[0] == 'series':
        print("analyzing series", guideTree)
        rays, labels = analyzeSeries(pixels, origin, guideTree, orient=orient, flexible=flexible)
        rayLst += rays
        labelLst += labels
    elif guideTree[0] == 'parallel':
        print("analyzing parallel", guideTree[1:])
        for branch in guideTree[1:]:
            if len(branch) == 2:
                alterOrient = branch[1][2]
        for branch in guideTree[1:]:
            rays, labels = splitImage(pixels, origin, branch, orient=alterOrient, flexible=flexible)
            rayLst += rays
            labelLst += labels
    return rayLst, labelLst

def produceDense():
    # Define image properties
    image_size = (1000, 1000)

    # Generate random noise image
    # random_image = np.random.randint(0, 256, image_size, dtype=np.uint8)
    random_image = np.random.choice([0, 255], image_size)
    # Save the random image
    cv2.imwrite('./images/randomImage.png', random_image)
    # detectEdge(random_image, useCanny=True)


def drawRays(image, rays):
    global colorA
    global colorB
    rgbImage = image.convert('RGB')
    rayImage = ImageDraw.Draw(rgbImage)
    for ray in rays:
        if ray[0][0] == ray[1][0]:
            color = colorB
        if ray[0][1] == ray[1][1]:
            color = colorA
        rayImage.line(ray, fill=color, width=0)
    return rgbImage


def drawLabels(image, labelLst):
    global colorA
    global colorB
    rgbImage = image.convert('RGB')
    labelImage = ImageDraw.Draw(rgbImage)
    size = 64
    unicodeFont = ImageFont.truetype("Dengl.ttf", size)
    for tup in labelLst:
        shift = -1 if tup[2] == 0 else 1
        if not tup[2]:
            color = colorA
        else:
            color = colorB
        coord = (tup[0][0] - size * 2.5 / 2, tup[0][1] + size * shift)
        labelImage.text(coord, tup[1], font=unicodeFont, fill=color)
    return rgbImage


def drawTree(tree, dynamic=False):
    depth = countDepth(tree)
    print(depth)
    width = height = int(math.pow(3, depth + 1))
    pixels = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append([0, 0])
        pixels.append(row)
    pixels = treeHelper(tree, (0, 0), width, height, pixels, dynamic=dynamic)
    print(pixels)

    for i in range(height):
        for j in range(width):
            pixels[i][j] = pixels[i][j][0] / pixels[i][j][1] * 255
    array = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(array, 'L')
    return image


def treeHelper(tree, origin, width, height, pixels, dynamic, orient=1):
    alterOrient = orient
    if tree[0] == 'series':
        print("e")
    else:
        if orient == 0:
            for i in range(3):
                if tree[0] != 'parallel':
                    for k in range(height // 3):
                        for j in range(width):
                            pixels[origin[1] + i * height // 3 + k][origin[0] + j][0] += hexMap[exHex(tree[0])[0]][i]
                            pixels[origin[1] + i * height // 3 + k][origin[0] + j][1] += 1
                else:
                    for branch in tree[1:]:
                        if len(branch) == 2:
                            alterOrient = branch[1][2]
                    for branch in tree[1:]:
                        pixels = treeHelper(branch, (origin[0], origin[1] + i * height // 3), width, height // 3,
                                            pixels, dynamic=dynamic, orient=alterOrient)
        elif orient == 1:
            for i in range(3):
                if tree[0] != 'parallel':
                    for k in range(width // 3):
                        for j in range(height):
                            pixels[origin[1] + j][origin[0] + i * width // 3 + k][0] += hexMap[exHex(tree[0])[0]][i]
                            pixels[origin[1] + j][origin[0] + i * width // 3 + k][1] += 1
                else:
                    for branch in tree[1:]:
                        if len(branch) == 2:
                            alterOrient = branch[1][2]
                    for branch in tree[1:]:
                        pixels = treeHelper(branch, (origin[0] + i * width // 3, origin[1]), width // 3, height,
                                            pixels, dynamic=dynamic, orient=alterOrient)
    return pixels


def readVideo(name, jump=1):
    video = cv2.VideoCapture('./images/videos/' + name + '.mp4')
    success, image = video.read()
    count = 0
    while success:
        cv2.imwrite("./images/frames/frame%d.png" % count, image)  # save frame as JPEG file
        for i in range(jump):
            success, image = video.read()
        print('Read a new frame: ', success)
        count += jump
    print("Finished reading the video.", count, "frames in total.")


def frameDif(pixels1, pixels2):
    difPixels = []
    for i in range(len(pixels1)):  # iterates through y
        difPixels.append([])
        for j in range(len(pixels1[0])):  # iterates through x
            val = (pixels2[i][j] - pixels1[i][j] + 255) / 2
            # print(pixels2[i][j], pixels1[i][j])
            difPixels[i].append(val)
    return difPixels


def main():
    guideTree = ['series', ['c', 1, 0], ['parallel', ['series', ['c', 0, 0]], ['series', ['z', 0, 0]]], ['t', 0, 0]]
    # readVideo('damage proof', jump=10)
    pixels = None
    frameJump = 10
    for frameNum in range(0, 1):
        lastPixels = pixels
        image = Image.open(r"./images/frames/room" + str(frameNum * frameJump) + ".png")
        # sourceImages = extractChannels(image)
        sourceImages = [detectEdge(image, useCanny=True)]
        # sourceImages = [image.convert('L')]
        resultImages = []
        for i, image in enumerate(sourceImages):
            if len(sourceImages) == 3:
                image.save('./images/results/' + ['red', 'green', 'blue'][i] + 'Image.png')
            # image.rotate(-90)
            pixels = list(image.getdata())
            width, height = image.size
            pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
            if staticMode:
                rayLst, labelLst = splitImage(pixels, (0, 0), guideTree, flexible=True)
                print(rayLst)
                print(labelLst)
                #treeImage = drawTree(guideTree)
                #treeImage.save('./images/results/tree' + str(frameNum * frameJump) + '-' + str(i) + '.png')
                #resultImages.append(treeImage)
                image = drawRays(image, rayLst)
                image = drawLabels(image, labelLst)
                image.save(r'./images/results/result' + str(frameNum * frameJump) + '-' + str(i) + '.png')
            if dynamicMode:
                if lastPixels is not None:
                    difPixels = frameDif(lastPixels, pixels)
                    difArray = np.array(difPixels, dtype=np.uint8)
                    difImage = Image.fromarray(difArray, 'L')
                    difImage.save('./images/results/difference' + str(frameNum * frameJump) + '-' + str(i) + '.png')
                    rays = splitImage(difPixels, (0, 0), guideTree)
                    print(rays)
                    #treeImage = drawTree(tree, dynamic=True)
                    #treeImage.save('./images/results/treeDyn' + str(frameNum * frameJump) + '-' + str(i) + '.png')
                    #resultImages.append(treeImage)
                    difImage = drawRays(difImage, rayLst)
                    difImage = drawLabels(difImage, labelLst)
                    difImage.save(r'./images/results/resultDyn' + str(frameNum * frameJump) + '-' + str(i) + '.png')
        if len(resultImages) == 3:
            mergedImage = Image.merge("RGB", (resultImages[0], resultImages[1], resultImages[2]))
            mergedImage.save('./images/results/mergedTree.png')
        elif len(resultImages) == 6:
            mergedStaticImage = Image.merge("RGB", (resultImages[0], resultImages[2], resultImages[4]))
            mergedStaticImage.save('./images/results/mergedStaticTree.png')
            mergedDynamicImage = Image.merge("RGB", (resultImages[1], resultImages[3], resultImages[5]))
            mergedDynamicImage.save('./images/results/mergedDynamicTree.png')


if __name__ == "__main__":
    main()
