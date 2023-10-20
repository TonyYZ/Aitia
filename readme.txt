Aitia is one of the three core sections making up the Ítí project, the other two being Nguasach and Ete. Its purpose is to implement the semantics of Ítí, a conlang inspired by I-Ching trigrams.
The name Aitia comes from the Greek word αιτία, which sounds like Ítí and means "cause, reason", thus the origin of "etiology". It is named after this concept because it serves to construct the meaning of linguistic forms. It cannot exist without the life energy (~ goal, desire) that motivates the expression of linguistic forms.
The most important files in this project are imageProcessing and bayesianLearning.
ImageProcessing is mainly designed for translating an image or a video into a series of syntax trees in Ítí. However, it is also capable of doing the operation in reverse: generating a video that simulates the possible meaning indicated by a syntax tree.
BayesianLearning sees Ítí semantics as a cognitive model for learning meaning from videos. It tries to find the syntax tree (the hypothesis) that suits the video the most using a tree similarity algorithm. A series of trees are created based on each frame of the video using imageProcessing and are compared with the hypothesis syntax tree.


imageProcessing.py 将图像划分格子，横三分竖三分往复分割，格线为了提高卦象概率可以灵活变动，最后得出卦象分布。还可以反向输出图像/视频，bayesianLearning.py会经常用到其中的显示方法。
*interpreter.py 用iti作为可变动格线的框架去套图像，用动态元素按照扩散方程去演化，计算该框架能产出该图像序列的概率
*diffusionSimulation.py 用扩散方程的差分方法模拟卦象演化
bayesianLearning.py 用贝叶斯学习方法随机生成句子树，衡量其作为语义模型能够生成目标图像树的程度，不断筛选生成新树。

*为已弃用
