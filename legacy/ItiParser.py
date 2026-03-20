import Iti2IPA
import re

text = """áce cơz õt. s"""


def split(txt):
    acc = 0
    for pos, char in enumerate(txt):
        if char == '(':
            txt = txt[:pos + acc] + '.#.' + txt[pos + acc:]
            acc += 3
    # arr = re.split('    |- |, |,\n|,$|\. |\.\n|\.$|: |:\n|:$|; |;\n|;$|\? |\?\n|\?$|! |!\n|!$|\n|\(|\) ', txt)
    arr = re.split('    |- |,|\.|:|;|\?|!\n|\(|\)', txt)
    arr = [i.strip() for i in arr]
    for pos, str in enumerate(arr):
        if str == '#':
            arr[pos - 1] += arr[pos + 3]
            arr[pos + 3] = ''
    return [i for i in arr if i and not i.isspace() and i != '#' and not i.isdigit()]


expandDict = {
    'á': 'a`', 'à': 'a-', 'ā': 'aa',  # vowels
    'é': 'e`', 'è': 'e-', 'ē': 'ee', 'ẻ': 'a`e`', 'ț': 'o`e`', 'ȟ': 'a`o`e`',
    'í': 'i`', 'ì': 'i-', 'ī': 'ii', 'ỉ': 'e`i`', 'ȍ': 'o`i`',
    'ó': 'o`', 'ò': 'o-', 'ō': 'oo', 'ỏ': 'a`o`', 'ớ': 'ơ`', 'ờ': 'ơ-',
    'ú': 'u`', 'ù': 'u-', 'ū': 'uu', 'ủ': 'o`u`', 'ứ': 'e`u`',
    'ĩ': 'ei',
    'ị': 'oi',
    'ẹ': 'oe',
    'ẽ': 'ae',
    'ȡ': 'aoe',
    'õ': 'ao',
    'ư': 'eu',
    'ũ': 'ou',
}


def expand(txt):
    for letter in expandDict:
        txt = txt.replace(letter, expandDict[letter])
    return txt


def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def exLayer(v):
    if v in ['i', 'í', 'ĩ', 'ỉ', 'ị', 'ȍ']:
        return -1
    elif v in ['a', 'á', 'ẽ', 'ẻ', 'õ', 'ỏ']:
        return 0
    elif v in ['u', 'ú', 'ư', 'ứ', 'ũ', 'ủ']:
        return 1
    else:
        return ''


def countLayer(string):
    ultimate = string[-1]
    count = 1
    if ultimate in ['i', 'í', 'ĩ', 'ỉ', 'ị', 'ȍ']:
        options = ['i', 'í', 'ĩ', 'ỉ', 'ị', 'ȍ']
        sign = -1
    elif ultimate in ['u', 'ú', 'ư', 'ứ', 'ũ', 'ủ']:
        options = ['u', 'ú', 'ư', 'ứ', 'ũ', 'ủ']
        sign = 1
    elif ultimate in ['a', 'á', 'ẽ', 'ẻ', 'õ', 'ỏ']:
        return 0
    else:
        return ''

    for c in reversed(string):
        if c == "'":
            count += 1
        elif c not in options:
            break
    return sign * count


def countAccent(string):  # return the number of accents in the vowel
    accentSum = 0
    for c in string:
        if c in expandDict:
            accentSum += c.replace(c, expandDict[c]).count('`')
        elif c not in ['a', 'i', 'u', 'e', 'o', 'ơ', "'"]:
            print('Error: ' + v + " is not a vowel or a quotation mark")
            return None
    return accentSum


def remAccent(v):  # return the vowel after its accent(s) has/have been removed
    if v in expandDict:
        return expandDict[v].replace('`', '')
    elif v in ['a', 'i', 'u', 'e', 'o', 'ơ']:
        return v
    else:
        print('Error: ' + v + ' is not a vowel')
        return None


c = "[żṡzspbtdmnfvlrcgċhyw']"
ve = "(e|é|(ĩ|ỉ)('(i|í))*|ẽ|ẻ|(ư|ứ)('(ư|ứ))*)"
vo = "(o|ó|(ị|ȍ)('(i|í))*|õ|ỏ|(ũ|ủ)('(u|ú))*)"
ven = "(?:é|(ĩ|ỉ)('(i|í))*|ẽ|ẻ|(ư|ứ)('(ư|ứ))*)"
veni = "(?:é|(ĩ|ỉ)('(i|í))*|ẽ|ẻ|(ư|ứ)('(u|ú))*|e)"
von = "(?:ó|(ị|ȍ)('(i|í))*|õ|ỏ|(ũ|ủ)('(u|ú))*)"
voni = "(?:ó|(ị|ȍ)('(i|í))*|õ|ỏ|(ũ|ủ)('(u|ú))*|o)"
vl = "(?:a|á|(i|í)('(i|í))*|(u|ú)('(u|ú))*)"
# branch = '(' + vl + '*' + c + '(?:' + vl + '*' + c + ')*)'  # V*C[V*C] (needs to check the first of V* is layer)
# branchSP = '((?:ớ|' + vl + '*)' + c + '(?:' + vl + '*' + c + ')*)'  # only for sp12sp's 2
start = ''  # '' = nothing, 'e/o' = normal, '[ĩẽư]/[ịõũ]' = special
end = ''  # '' = nothing, 'e/o' = normal, '[ĩẽư]/[ịõũ]' = special
prev = -1  # -1 = start, 0 = not fuseau & lacks e, 1 = fuseau & lacks o, 2 = fuseau & has o, 3 = not fuseau & has e
layer = 0
accent = 0
cur = ''
lst = []


def branch(seg, tmpLayer, tmpAccent, sp, inner=False, guideMode=False):
    global start
    compNum = 0
    if guideMode:
        newLst = ["series"]
    else:
        newLst = []
    print("branch now", seg, sp)

    if sp > 1:  # 2 = allows ớ, 1 = first branch, 0 = else
        unit = re.match('(?P<check>ớ|' + vl + '*)', seg)  # ớ|v*
    else:
        unit = re.match('(?P<check>' + vl + '*)', seg)  # v*

    if unit is not None:
        check = unit.group('check')
        if check != '':
            print(check, start, sp, inner)

            if (check == 'ớ' or exLayer(check[0]) != sign(tmpLayer)) and (
                    start != '' or sp != 1 or inner):  # if the branch begins with vowel, first
                # vowel has to be different from layer. If it is the first branch, make sure start contains vowel
                if check != 'ớ':
                    tmpLayer = countLayer(check)
                for v in check:
                    if v == "'":
                        continue
                    tmpAccent += countAccent(v)
                seg = seg[unit.span()[1]:]
            else:
                return None  # not an independent branch (vowel may be continuing the previous branch)
        unit = re.match('(?P<br>' + c + ')', seg)
        if unit is not None:
            br = unit.group('br')
            seg = seg[unit.span()[1]:]
            if br != "'":
                newLst.append([br, tmpAccent, tmpLayer])
                compNum += 1
                tmpAccent = 0
                print("anịs", tmpAccent)
            else:  # not c then maybe an inner fuseau
                print("analyzing dual fuseau", seg)
                analyses = [dualFuseau(seg, tmpLayer, tmpAccent, inner=True, guideMode=guideMode)]
                print("analyzing complex fuseau", seg, analyses)
                analyses.append(complexFuseau(seg, tmpLayer, tmpAccent, inner=True, guideMode=guideMode))
                print("analyses", analyses, seg)
                result = next((i for i, tup in enumerate(analyses) if tup is not None), '¡All are Nones!')
                if result == '¡All are Nones!':
                    return None
                else:
                    newLst.append(analyses[result][0])
                    print("newLst", newLst)
                    seg = analyses[result][1]
                    tmpLayer = analyses[result][2]
                    tmpAccent = analyses[result][3]
        else:
            return None
    else:
        return None

    while 1:
        unit = re.match('(?P<check>' + vl + '+)', seg)
        print("checking continue vowel", unit, tmpLayer)
        if unit is not None:
            check = unit.group('check')
            if check != '':
                if exLayer(check[0]) == sign(tmpLayer):
                    tmpLayer = countLayer(check)
                    for v in check:
                        tmpAccent += countAccent(v)
                    seg = seg[unit.span()[1]:]
                else:
                    break
            unit = re.match('(?P<br>' + c + ')', seg)
            print("continue", unit, "current layer", tmpLayer)
            if unit is not None:
                br = unit.group('br')
                seg = seg[unit.span()[1]:]
                print("deleting", seg)
                if br != "'":
                    newLst.append([br, tmpAccent, tmpLayer])
                    compNum += 1
                    tmpAccent = 0
                else:  # not c then maybe an inner fuseau
                    print("analyzing dual fuseau", seg)
                    analyses = [dualFuseau(seg, tmpLayer, tmpAccent, inner=True, guideMode=guideMode)]
                    print("analyzing complex fuseau", seg, analyses)
                    analyses.append(complexFuseau(seg, tmpLayer, tmpAccent, inner=True, guideMode=guideMode))
                    print("analyses", analyses, seg)
                    result = next((i for i, tup in enumerate(analyses) if tup is not None), '¡All are Nones!')
                    if result == '¡All are Nones!':
                        return None
                    else:
                        newLst.append(analyses[result][0])
                        seg = analyses[result][1]
                        tmpLayer = analyses[result][2]
                        tmpAccent = analyses[result][3]
            else:
                break
        else:
            break
    print("branch now returning newLst", newLst, "seg", seg)
    return newLst, seg, tmpLayer, tmpAccent, compNum


def simpleFuseau(seg, tmpLayer, tmpAccent, guideMode=False):  # 1ơ2
    global prev
    global lst
    compNum = 0
    if guideMode:
        newLst = ['parallel']
    else:
        newLst = []
    unit = re.match('(?P<br1>' + c + ')ơ(?P<br2>' + c + ')(?:\s+|$)', seg)
    if unit is not None:
        prev = 1
        print('Simple Fuseau (1ơ2). Start: , end: , prev: 1')
        seg = seg[unit.span()[1]:]  # eats unit from seg
        br1 = unit.group('br1')
        br2 = unit.group('br2')
        if guideMode:
            newLst.append(['series', [br1, tmpAccent, tmpLayer]])
        else:
            newLst.append([[br1, tmpAccent, tmpLayer]])
        tmpAccent = 0
        if guideMode:
            newLst.append(['series', [br2, tmpAccent, tmpLayer]])
        else:
            newLst.append([[br2, tmpAccent, tmpLayer]])
        compNum += 2
        return newLst, seg, tmpLayer, tmpAccent, compNum
    else:
        return None


def dualFuseau(seg, tmpLayer, tmpAccent, inner=False, guideMode=False):  # sp1(ớ)2sp
    global prev
    global start
    compNum = 0
    if guideMode:
        newLst = ['parallel']
    else:
        newLst = []

    if not inner and seg[0] == "'" and prev != 3:
        print("Error in parsing the dual fuseau. The consonant ' must precede e.")
        return None
    if inner or len(seg) > 1 and seg[0] in veni and seg[1] == "'":
        unit = re.match('(?P<start>' + veni + ')', seg)
    else:
        unit = re.match('(?P<start>' + ven + '?)', seg)
        if ' ' in seg:
            hasV = re.search(veni + '|' + voni + '|' + vl + '|' + 'ớ', seg[:seg.index(' ')])
        else:
            hasV = re.search(veni + '|' + voni + '|' + vl + '|' + 'ớ', seg)
        if not hasV:
            print("Error in parsing the dual fuseau. It must contain at least one vowel.")
            return None

    if unit is not None:
        start = unit.group('start')
        if start != '' and (prev == -1 or prev == 1 or prev == 2 or inner):
            count = countLayer(start)
            if count != '':
                tmpLayer = count
            tmpAccent += countAccent(start)
        elif not (start == '' and (prev == -1 or prev == 1 or prev == 2 or prev == 3)):
            # print('Error in parsing dual fuseau segment start: ' + start + ' prev: ' + str(prev))
            return None
        seg = seg[unit.span()[1]:]
    else:
        return None

    for sp in [1, 2]:
        print("dual fuseau branch", sp, seg)
        res = branch(seg, tmpLayer, tmpAccent, sp, inner, guideMode=guideMode)  # return pair (newLst, seg, tmpLayer, tmpAccent)
        print("dual fuseau branch over", sp, seg, res)
        if res is None:
            return None
        else:
            newLst.append(res[0])
            seg = res[1]
            tmpLayer = res[2]
            tmpAccent = res[3]
            compNum += res[4]
    if inner:
        unit = re.match("(?P<end>" + voni + ")'", seg)
        if unit is not None:
            print("checking veni:", seg, seg[unit.span()[1]:])
            checkVeni = re.match(veni, seg[unit.span()[1]:])  # if followed by veni then definitely a new fuseau
            if checkVeni:
                print("veni checked")
                return None
    else:
        unit = re.match('(?P<end>' + von + '?)(?:\s+|$)', seg)
    print("unit", unit, seg)
    if unit is not None:
        end = unit.group('end')
        if end != '':
            count = countLayer(end)
            if count != '':
                tmpLayer = count
            tmpAccent += countAccent(end)
            if not inner:
                prev = 2
        else:
            if not inner:
                prev = 1

        print('Dual Fuseau (sp1(ớ)2sp). Start: ' + start + ', end: ' + end + ', prev: ' + str(prev))
        seg = seg[unit.span()[1]:]
        print("returning", newLst, seg)
        return newLst, seg, tmpLayer, tmpAccent, compNum
    else:
        print("returning none")
        return None


def complexFuseau(seg, tmpLayer, tmpAccent, inner=False, guideMode=False):  # sp1(ớ|[e2o])3o4sp
    global prev
    global start
    compNum = 0
    if guideMode:
        newLst = ['parallel']
    else:
        newLst = []
    if not inner and seg[0] == "'" and prev != 3:
        print("Error in parsing the complex fuseau. The consonant ' must precede e.")
        return None
    if inner or len(seg) > 1 and seg[0] in veni and seg[1] == "'":
        unit = re.match('(?P<start>' + veni + ')', seg)
    else:
        unit = re.match('(?P<start>' + ven + '?)', seg)
    if unit is not None:
        start = unit.group('start')
        if start != '' and (prev == -1 or prev == 1 or prev == 2 or inner):
            count = countLayer(start)
            if count != '':
                tmpLayer = count
            tmpAccent += countAccent(start)
        elif not (start == '' and (prev == -1 or prev == 1 or prev == 2 or prev == 3)):
            # print('Error in parsing complex fuseau segment start: ' + start + ' prev: ' + str(prev))
            return None
        seg = seg[unit.span()[1]:]
    else:
        return None
    print("complex fuseau branch 1", seg)
    res = branch(seg, tmpLayer, tmpAccent, 1, inner, guideMode=guideMode)
    if res is None:
        return None
    else:
        newLst.append(res[0])
        seg = res[1]
        tmpLayer = res[2]
        tmpAccent = res[3]
        compNum += res[4]
    res = branch(seg, tmpLayer, tmpAccent, 2, inner, guideMode=guideMode)  # checks (ớ)3

    if res is None:  # if fails, check (e2o*)3
        tmp2 = seg
        while 1:  # feeds as branch continuously, stop if fails to identify
            unit = re.match('(?P<check>' + ve + ')', tmp2)
            if unit is None:
                break
            else:
                check = unit.group('check')
                count = countLayer(check)
                if count != '':
                    tmpLayer = count
                tmpAccent += countAccent(check)
                tmp2 = tmp2[unit.span()[1]:]
                res = branch(tmp2, tmpLayer, tmpAccent, 0, inner, guideMode=guideMode)
                if res is None:
                    break
                else:
                    tmp2 = res[1]
                    unit = re.match('(?P<check>' + vo + ')', tmp2)
                    if unit is None:
                        break
                    else:
                        newLst.append(res[0])
                        tmpLayer = res[2]
                        tmpAccent = res[3]
                        check = unit.group('check')
                        count = countLayer(check)
                        if count != '':
                            tmpLayer = count
                        print('anịs2', tmpAccent, countAccent(check))
                        tmpAccent += countAccent(check)
                        tmp2 = tmp2[unit.span()[1]:]
                        seg = tmp2  # updates seg only if the entire loop succeeds
                        compNum += res[4]
        res = branch(seg, tmpLayer, tmpAccent, 0, inner, guideMode=guideMode)
        if res is None:
            return None
        else:
            newLst.append(res[0])
            seg = res[1]
            tmpLayer = res[2]
            tmpAccent = res[3]
            compNum += res[4]
    else:
        newLst.append(res[0])
        seg = res[1]
        tmpLayer = res[2]
        tmpAccent = res[3]
        compNum += res[4]

    unit = re.match('(?P<check>' + vo + ')', seg)  # vo
    if unit is not None:
        check = unit.group('check')
        count = countLayer(check)
        if count != '':
            tmpLayer = count
        tmpAccent += countAccent(check)
        seg = seg[unit.span()[1]:]
    else:
        return None
    res = branch(seg, tmpLayer, tmpAccent, 0, inner, guideMode=guideMode)  # branch
    if res is None:
        return None
    else:
        newLst.append(res[0])
        seg = res[1]
        tmpLayer = res[2]
        tmpAccent = res[3]
        compNum += res[4]
    if inner:
        unit = re.match("(?P<end>" + voni + ")'", seg)
    else:
        unit = re.match('(?P<end>' + von + '?)(?:\s+|$)', seg)
    if unit is not None:
        end = unit.group('end')
        if end != '':
            count = countLayer(end)
            if count != '':
                tmpLayer = count
            tmpAccent += countAccent(end)
            if not inner:
                prev = 2
        else:
            if not inner:
                prev = 1
        print('Complex Fuseau (sp1(ớ|[e2o])3o4sp). Start: ' + start + ', end: ' + end + ', prev: ' + str(prev))
        seg = seg[unit.span()[1]:]
        return newLst, seg, tmpLayer, tmpAccent, compNum
    else:
        return None


def parseByChar(seg):
    global layer
    global accent
    global lst
    compNum = 0
    for char in seg:
        if char in c and char != "'":
            lst.append([char, accent, layer])
            compNum += 1
            layer = sign(layer)
            accent = 0
        if char in vl:
            res = exLayer(char)
            if res != '' and res != sign(layer):
                layer = res
            accent += countAccent(char)
        if char == "'" and layer != 0:
            layer += sign(layer)
    return compNum


def parse(guideMode=False):
    global layer
    global accent
    global prev
    global cur
    global lst
    global start
    global end
    compNum = 0
    if guideMode:
        lst.append("series")
    while cur:
        analyses = [simpleFuseau(cur, layer, accent, guideMode=guideMode),
                    dualFuseau(cur, layer, accent, guideMode=guideMode),
                    complexFuseau(cur, layer, accent, guideMode=guideMode)]
        result = next((i for i, tup in enumerate(analyses) if tup is not None), '¡All are Nones!')
        print("here are the results:", analyses)
        if result == '¡All are Nones!':
            unit = re.match('(?P<start>' + vo + '?)(?P<middle>(' + c + '|' + vl + ')+)(?P<end>'
                            + ve + '?)(?:\s+|$)', cur)  # non-fuseau
            if unit is not None:
                start = unit.group('start')
                if start != '' and prev == 1:  # we have an o now, and the previous segment is a fuseau without o
                    count = countLayer(start)
                    if count != '':
                        layer = count
                    accent += countAccent(start)
                elif not (start == '' and (prev == 0 or prev == -1)):  # (we don't have an o, and the previous segment
                    # is a non-fuseau without e or there is no previous segment)
                    print('Error in parsing the non-fuseau.')
                    break
                middle = unit.group('middle')
                if re.search(c, middle) is None:
                    print("Error in parsing the non-fuseau. It must contain a consonant.")
                compNum += parseByChar(middle)
                end = unit.group('end')
                if end != '':
                    count = countLayer(end)
                    if count != '':
                        layer = count
                    accent += countAccent(end)
                    prev = 3
                else:
                    prev = 0
                print('Non-fuseau. Start: ' + start + ', end: ' + end + ', prev: ' + str(prev))
                cur = cur[unit.span()[1]:]
                continue
            else:
                print("Error in parsing the sentence. Remainder: " + cur)
                break
        else:
            lst.append(analyses[result][0])
            cur = analyses[result][1]
            layer = analyses[result][2]
            accent = analyses[result][3]
            compNum += analyses[result][4]
    return compNum


def convert2Tree(guideTree):  # convert guide trees into drawable trees
    if guideTree[0] in ['parallel', 'series']:
        newTree = []
        for comp in guideTree[1:]:
            newTree.append(convert2Tree(comp))
    else:
        newTree = guideTree
    return newTree

def read(txt, guideMode=False):
    global prev
    global layer
    global accent
    global cur
    global lst
    passageLst = []
    splitArr = split(Iti2IPA.lower(txt))
    # expandArr = list(map(expand, splitArr))
    compNum = 0
    print(splitArr)
    for sent in splitArr:
        prev = -1
        layer = 0
        accent = 0
        cur = sent
        lst = []
        compNum += parse(guideMode=guideMode)
        passageLst.append(lst)
    print(passageLst, compNum)
    return passageLst, compNum


def main():
    read(text)


if __name__ == "__main__":
    main()
