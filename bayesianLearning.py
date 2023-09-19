# -*- coding: utf-8 -*-
import copy
import math

from zss import simple_distance, Node
import networkx as nx
import matplotlib.pyplot as plt
import imageProcessing as improc
import numpy as np
import Levenshtein
import demoBackups

# sourceTree = ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]], ['parallel', ['series', ['s', 1, 0]], ['series', ['b', 0, 0]]], ['parallel', ['series', ['ṡ', 1, 0]], ['series', ['b', 0, 0]]]]], ['parallel', ['series', ['b', 0, 0]], ['series', ['parallel', ['series', ['s', 1, 0]], ['series', ['s', 0, 0]]], ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]], ['parallel', ['series', ['s', 1, 0]], ['series', ['v', 0, 0]]]]]], ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['parallel', ['series', ['ṡ', 1, 0]], ['series', ['b', 0, 0]]], ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]], ['parallel', ['series', ['d', 1, 0]], ['series', ['r', 0, 0]]]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['parallel', ['series', ['g', 1, 0]], ['series', ['s', 0, 0]]], ['parallel', ['series', ['b', 1, 0]], ['series', ['n', 0, 0]]], ['parallel', ['series', ['g', 1, 0]], ['series', ['v', 0, 0]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['parallel', ['series', ['n', 1, 0]], ['series', ['r', 0, 0]]], ['parallel', ['series', ['s', 1, 0]], ['series', ['r', 0, 0]]], ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]], ['parallel', ['series', ['v', 1, 0]], ['series', ['s', 0, 0]]], ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]]]]]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['parallel', ['series', ['s', 1, 0]], ['series', ['d', 0, 0]]], ['parallel', ['series', ['g', 1, 0]], ['series', ['s', 0, 0]]], ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]], ['parallel', ['series', ['v', 1, 0]], ['series', ['s', 0, 0]]], ['parallel', ['series', ['v', 1, 0]], ['series', ['v', 0, 0]]]]]], ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]], ['parallel', ['series', ['b', 1, 0]], ['series', ['v', 0, 0]]], ['parallel', ['series', ['v', 1, 0]], ['series', ['s', 0, 0]]]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['parallel', ['series', ['s', 1, 0]], ['series', ['n', 0, 0]]], ['parallel', ['series', ['d', 1, 0]], ['series', ['s', 0, 0]]], ['parallel', ['series', ['s', 1, 0]], ['series', ['v', 0, 0]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['parallel', ['series', ['s', 1, 0]], ['series', ['v', 0, 0]]], ['parallel', ['series', ['ṡ', 1, 0]], ['series', ['v', 0, 0]]], ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]]]], ['parallel', ['series', ['v', 0, 0]], ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['s', 0, 0]]], ['parallel', ['series', ['v', 1, 0]], ['series', ['v', 0, 0]]], ['parallel', ['series', ['h', 1, 0]], ['series', ['h', 0, 0]]]]]]]]]

sourceTree = [[['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['d', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['g', 0, 0]], ['series', ['t', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['d', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['v', 0, 0]], ['series', ['z', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['p', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['p', 0, 0]]]], ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['p', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['l', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['b', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['c', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]]], [['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['d', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['g', 0, 0]], ['series', ['t', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['v', 0, 0]], ['series', ['z', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['l', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['b', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]]], [['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['g', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['d', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['m', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['m', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['c', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['v', 0, 0]], ['series', ['c', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['b', 1, 0]], ['series', ['l', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['t', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]]], [['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['d', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['z', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['n', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['z', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['d', 0, 0]], ['series', ['z', 0, 0]]]], ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['l', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['v', 0, 0]], ['series', ['c', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['l', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['p', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['l', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['s', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['v', 1, 0]], ['series', ['p', 1, 0]]], ['parallel', ['series', ['s', 0, 0]], ['series', ['c', 0, 0]]]]]], ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['f', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['c', 0, 0]]]], ['parallel', ['parallel', ['series', ['b', 1, 0]], ['series', ['l', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['p', 0, 0]]]], ['parallel', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]], ['parallel', ['series', ['n', 0, 0]], ['series', ['z', 0, 0]]]]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]], ['series', ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]], ['parallel', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]], ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]]]]
#sourceTree = [[['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['p', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['p', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['p', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]], [['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['l', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['l', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]]]]]], [['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['f', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['t', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['z', 0, 0]]]]]]]]]], [['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['f', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['v', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['z', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['l', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['z', 1, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['t', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['c', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['r', 1, 0]], ['series', ['c', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['c', 0, 0]]]]]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]], ['series', ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['d', 1, 0]], ['series', ['l', 1, 0]]]], ['series', ['parallel', ['series', ['r', 0, 0]], ['series', ['p', 0, 0]]]]], ['parallel', ['series', ['parallel', ['series', ['h', 1, 0]], ['series', ['ċ', 1, 0]]]], ['series', ['parallel', ['series', ['h', 0, 0]], ['series', ['ċ', 0, 0]]]]]]]]]]]
#sourceTree = [frameTree[1] for frameTree in sourceTree][:2]  # green channel
#sourceTree = improc.reverseGuideTree(sourceTree)  # applies to raw sensory trees, not hand-written trees
sourceTree = []
for i in range(5):
    sourceTree += [['parallel', ['series', ['h', 0, 0], ['ṡ', 0, 0]]]]

# testTree = ['parallel', ['series', ['parallel', ['series', ['b', 0, 0]], ['series', ['f', 0, 0]]], ['parallel', ['series', ['c', 1, 0], ['n', 1, 0]]]], ['series', ['parallel', ['series', ['p', 1, 0], ['n', 1, 0]], ['series', ['t', 0, 0], ['r', 0, 0]]]], ['series', ['parallel', ['series', ['s', 0, 0]], ['series', ['t', 0, 0]]], ['parallel', ['series', ['g', 1, 0]], ['series', ['g', 1, 0]]]]]
testTree = demoBackups.notOp
#testTree = improc.reverseGuideTree(testTree)
# testTree = ['parallel', ['series', ['parallel', ['series', ['h', 0, -1]], ['series', ['n', 0, -1]]], ['parallel', ['series', ['m', 0, -2], ['f', 0, -1]], ['series', ['d', 1, 0]]]], ['series', ['ż', 1, 0], ['parallel', ['series', ['n', 0, 0]], ['series', ['g', 0, 0]]]]]
#sourceTree = [['parallel', ['series', ['g', 0, 0], ['z', 0, 0]]]]
#testTree = ['parallel', ['series', ['g', 0, 0], ['z', 0, 0]]]
leafSymbols = improc.leafSymbols
nodeSymbols = improc.nodeSymbols
replaceDict = improc.replaceDict
print('Leaf Symbols', leafSymbols)
nodeLst = [[], []]
samplePercentage = 1
prunePercentage = 0.5  # pursued tree completeness
global axes
initCompare = True


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
        label = improc.lst2tup(label)
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
    if (maxDepth <= 0 or np.random.rand() < 0.3) and tree[0] == nodeSymbols[1]:
        tree.append([np.random.choice(leafSymbols), np.random.choice([0, 1]),
                     np.random.randint([-2, 1])])  # random lowercase letter
        return
    else:
        numBranches = np.random.randint(1, 4)  # random number of branches
        if len(tree[1:]) < numBranches:
            for i in range(numBranches - len(tree[1:])):
                if tree[0] == nodeSymbols[0]:
                    tree.append([nodeSymbols[1]])
                else:
                    tree.append([np.random.choice(nodeSymbols)])
        elif len(tree[1:]) > numBranches:
            indices2Remove = np.random.choice(len(tree[1:]), size=len(tree[1:]) - numBranches,
                                              replace=False)
            tree[1:] = [child for i, child in enumerate(tree[1:]) if i not in indices2Remove]

        for child in tree[1:]:
            if child[0] in nodeSymbols:
                generateRandomTree(child, maxDepth - 1)


def getRandomLayer():
    shot = np.random.choice([1] * 15 + [0] * 45 + [-1] * 25 + [-2] * 15)
    # shot = np.random.choice([0] * 60 + [-1] * 25 + [-2] * 15)
    return shot


def getRandomBranch():  # random distribution based on frequency statistics
    shot = np.random.choice([1] * 35 + [2] * 60 + [3] * 5)
    return shot


def generateRandomNode(node, maxDepth):
    global leafSymbols
    global nodeSymbols

    if (maxDepth <= 0 or np.random.rand() < 0.8
        and node.label != nodeSymbols[0]) \
            and node.children == []:  # terminate tree if exceeds depth or randomly agrees
        # print("add one")
        node.addkid(Node(
            [np.random.choice(leafSymbols), np.random.choice([0, 1]), getRandomLayer()]))  # random lowercase letter
    else:  # expand/shrink tree
        numBranches = getRandomBranch()
        # print('len comparison', len(node.children), numBranches)
        if len(node.children) < numBranches:
            for i in range(numBranches - len(node.children)):
                if node.label == nodeSymbols[0]:
                    node.addkid(Node(nodeSymbols[1]))
                else:
                    if maxDepth - 1 <= 0:  # prevent parallel at the penultimate
                        node.addkid(Node([np.random.choice(leafSymbols), np.random.choice([0, 1]), getRandomLayer()]))
                    else:
                        node.addkid(Node(nodeSymbols[0]))
        elif len(node.children) > numBranches:
            indices2Remove = np.random.choice(len(node.children), size=len(node.children) - numBranches,
                                              replace=False)
            node.children = [child for i, child in enumerate(node.children) if i not in indices2Remove]

        for i, child in enumerate(node.children):
            if child.label not in nodeSymbols:  # the child is an element
                if np.random.rand() < 0.8:  # alter element pattern
                    currentPattern = list(improc.reversedCharMapDbl[child.label[0]])
                    # print('current pattern', currentPattern, child.label)
                    if currentPattern == [0.5, 0.5, 0.5] and np.random.rand() < 8 / 9:
                        currentPattern = [np.random.choice([0, 1]) for i in range(3)]
                    else:
                        if np.random.rand() < 1 / 9:
                            currentPattern = [0.5, 0.5, 0.5]  # accidentally falls into middle zone
                        else:
                            for j in range(3):
                                if np.random.rand() < 0.3:
                                    currentPattern[j] = 1 - currentPattern[j]
                    isDynamic = child.label[0] in improc.reversedCharMapDyn
                    if np.random.rand() < 0.3:  # flip the dynamic nature
                        isDynamic = 1 - isDynamic
                    newChar = improc.charMapDyn[tuple(currentPattern)] if isDynamic else improc.charMap[
                        tuple(currentPattern)]
                    # print('new char', newChar)
                    node.children[i] = Node([newChar, child.label[1], child.label[2]])
                if np.random.rand() < 0.3:  # alter element dimension
                    node.children[i] = Node([child.label[0], 1 - child.label[1], child.label[2]])
                if np.random.rand() < 0.2:  # alter element layer
                    node.children[i] = Node([child.label[0], child.label[1], getRandomLayer()])
            else:
                generateRandomNode(child, maxDepth - 1)


def treeSubstitutionMCMC(trainTrees, numIterations, verbose=0, showTree=True, showImage=True,
                         replaceText=False,
                         useTotal=False, useDepth=False, keyLen=6, showZero=True, showSource=True, useSymbols=False,
                         fillDyn=True, mirrorMode=False, precision=5):
    global nodeLst
    bestTree = None
    bestLikelihood = 0.0
    currentLikelihood = 0.0
    currentPrior = 0.0
    currentTree = Node('parallel')
    successIter = 0
    graphs = [None, None]

    labelCounter = {}
    # sortGuideTree(trainTree)
    sourceVideos = []
    sourceGraphs = []
    for i, trainTree in enumerate(trainTrees):
        if verbose > 0:
            print('Train Tree', i, trainTree)
        if showImage and showSource:
            videos, layerLst = improc.calcLayered(trainTree, normalize=False, reverse=False, keyLen=keyLen, layerMode=False, verbose=verbose > 2, fillDyn=fillDyn)
            sourceVideos += videos

        if showTree and showSource:
            graphs[0] = nx.DiGraph()
        trainTree = convert2NodeTree(trainTree, labelCounter=labelCounter, graph=graphs[0])
        nodeLst[0].append(flattenList(extractNodes(trainTree, needTotal=useTotal, needDepth=useDepth)))
        if showTree and showSource:
            sourceGraphs.append(graphs[0])
    if showSource:
        improc.playLayered(sourceVideos, isSource=True, mirrorMode=mirrorMode)
        showPlot(graph=sourceGraphs, replaceText=replaceText, videos=sourceVideos, layerLst=[0], isSource=True, useSymbols=useSymbols, mirrorMode=mirrorMode)
    for iteration in range(numIterations):
        if verbose > 0:
            print(f"Iteration: {iteration + 1}/{numIterations}, Accuracy: {bestLikelihood}")
        generateRandomNode(currentTree, 3)
        graphs[1] = nx.DiGraph()
        labelCounter = {}
        currentGuideTree = convert2GuideTree(currentTree, labelCounter=labelCounter, graph=graphs[1])

        # sortGuideTree(currentGuideTree)
        currentTree = convert2NodeTree(currentGuideTree)
        candidateTree = currentTree

        if verbose > 0:
            print("Current tree", convert2GuideTree(currentTree))
        nodeTotal = countNode(candidateTree)
        # candidatePrior = 2 - 2 / (1 + pow(math.e, -countNode(candidateTree) / 10))
        candidatePrior = 20 / nodeTotal if nodeTotal >= 20 else 1

        videos, layerLst, zeroLayerTrees = improc.calcLayered(currentGuideTree, normalize=False, reverse=True, needZero=True, keyLen=len(trainTrees), verbose=verbose > 2, fillDyn=fillDyn, precision=precision)
        print("zero tree len", len(zeroLayerTrees))
        if showZero:
            zeroGraphs = []
            for i in range(len(zeroLayerTrees)):
                zeroGraphs.append(nx.DiGraph())
            nodeLst[1] = [convert2NodeTree(tree, graph=zeroGraphs[i]) for i, tree in enumerate(zeroLayerTrees)]
        else:
            zeroGraphs = None
            nodeLst[1] = [convert2NodeTree(tree) for i, tree in enumerate(zeroLayerTrees)]
        candidateLikelihood = getLikelihood(useTotal=useTotal, useDepth=useDepth, verbose=verbose > 1)
        print('Node total', countNode(candidateTree), 'prior', candidatePrior, '* likelihood', candidateLikelihood, '=',
              candidatePrior * candidateLikelihood)
        if currentLikelihood * currentPrior == 0:
            acceptanceProb = min(1, candidateLikelihood * candidatePrior)
        else:
            acceptanceProb = min(1, (candidateLikelihood * candidatePrior) / (currentLikelihood * currentPrior))
        if verbose > 0:
            print("Acceptance Prob:", acceptanceProb)

        if np.random.rand() < acceptanceProb:
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
                graph = graphs[1] if showTree else None
                if not showImage:
                    videos = None
                showPlot(graph=graph, replaceText=replaceText, videos=videos, layerLst=layerLst, zeroGraphs=zeroGraphs, useSymbols=useSymbols, mirrorMode=mirrorMode)

    return bestTree, successIter, bestLikelihood


def showPlot(graph=None, replaceText=False, videos=None, layerLst=None, zeroGraphs=None, isSource=False, useSymbols=False, mirrorMode=False):
    showMode = [videos is not None, zeroGraphs is not None, graph is not None]
    if not any(showMode):
        return None
    if not isSource:
        if videos is not None:
            improc.playLayered(videos, layerLst, showMode=showMode, isSource=isSource, mirrorMode=mirrorMode)
            axes = improc.axes
        else:
            fig = improc.initAxes(showMode=showMode, numAxes=1)
            axes = improc.axes
    if showMode[1] and showMode[2]:
        startAx = -4
    elif showMode[1] and not showMode[2]:
        startAx = -3
    elif not showMode[1] or not showMode[2]:
        startAx = -1
    if isSource:
        startAx = 1
    for i, curMode in enumerate(showMode[1:]):  # applies to both zero layer graphs and complete graph
        if not curMode:
            continue
        if i == 0:
            improc.initButtons(zeroGraphs, startAx, replaceText=replaceText, useSymbols=useSymbols)
            improc.updateGallery(0, zeroGraphs, startAx, replaceText=replaceText, useSymbols=useSymbols)
            # title = 'Zero Layer 1/' + str(len(zeroGraphs))
        elif i == 1:
            if isSource:
                improc.initButtons(graph, startAx, replaceText=replaceText, videos=videos, useSymbols=useSymbols)
                improc.updateGallery(0, graph, startAx, replaceText=replaceText, videos=videos, useSymbols=useSymbols)
            else:
                improc.drawNetwork(graph, startAx, replaceText, useSymbols)
                title = 'Guide Tree'
                axes[startAx].set_title(title)
        startAx += 3
    plt.show(block=True)


def getLabelSimilarity(label1, label2):
    isDynamic = [label[0] in improc.reversedCharMapDyn for label in [label1, label2]]
    #if isDynamic[0] != isDynamic[1] or label1[1] != label2[1] or label1[2] != label2[2]:
    #    return 0

    staticDifSum = 0
    staticPatterns = [None, None]
    dynamicDifSum = 0
    dynamicPatterns = [None, None]
    dynamicAbsDifSum = 0
    dynamicAbsPatterns = [None, None]
    for i, label in enumerate([label1, label2]):
        staticPatterns[i] = improc.reversedCharMapDbl[label[0]]
        if isDynamic[i]:
            dynamicPatterns[i] = improc.vecMap[improc.reversedHexMap[improc.reversedCharMapDyn[label[0]]]][0]
            dynamicAbsPatterns[i] = improc.vecMap[improc.reversedHexMap[improc.reversedCharMapDyn[label[0]]]][1]
        else:
            dynamicPatterns[i] = [0, 0]
            dynamicAbsPatterns[i] = [0, 0]

    for i in range(3):
        staticDifSum += abs(staticPatterns[0][i] - staticPatterns[1][i])
    staticDist = staticDifSum / 3
    for i in range(2):
        dynamicDifSum += abs(dynamicPatterns[0][i] - dynamicPatterns[1][i])
        dynamicAbsDifSum += abs(dynamicAbsPatterns[0][i] - dynamicAbsPatterns[1][i])
    dynamicDist = (dynamicDifSum + dynamicAbsDifSum) / 6
    if not isDynamic[0] and not isDynamic[1]:
        patternDistSum = staticDist
    elif isDynamic[0] and isDynamic[1]:
        patternDistSum = dynamicDist
    else:
        patternDistSum = staticDist * 0.7 + dynamicDist * 0.3
    dist = patternDistSum * 0.7\
                 + (abs(int(isDynamic[0]) - int(isDynamic[1])) * 0.3 +
                    abs(label1[1] - label2[1]) * 0.7) * 0.3
    if label2[0] in ['ż', 'm', 'ċ'] and label1[0] not in ['ż', 'm', 'ċ']:  # punishment for unknown tree
        dist += 0.3
    if dist > 1:
        dist = 1
    # dist = 0.4 * staticDifSum / 3 + 0.3 * abs(int(label1[0] in improc.reversedCharMap) - int(label2[0] in improc.reversedCharMap)) + 0.3 * abs(label1[1] - label2[1])
    # print('dist between', label1, 'and', str(label2) + ':', dist)
    # print('static', staticPatterns, staticDifSum, 'dynamic', dynamicPatterns, dynamicDifSum, 'abs', dynamicAbsPatterns, dynamicAbsDifSum)
    return 1 - dist


def flattenList(lst):
    stack = lst[::-1]  # Reverse the list to simulate a stack
    result = []

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
            result.append(extractNodes(child, depth=depth + 1, needTotal=needTotal, needDepth=needDepth))
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


def getLikelihood(useTotal=False, useDepth=False, verbose=False):  # tree1 is the source tree, nodeLst[0] is given ex-method
    global nodeLst
    global samplePercentage
    avgCoverage = 0
    for showI in range(len(nodeLst[0])):
        # print(len(nodeLst[0]), len(nodeLst[1]))
        tree1 = nodeLst[0][showI]
        tree2 = nodeLst[1][showI]
        tree2 = flattenList(extractNodes(tree2, needTotal=useTotal, needDepth=useDepth))
        if useDepth:
            maxDepth = max([tup[2] for tup in tree1])
            print("max depth", maxDepth)
        score = 0
        if not useDepth:
            total = 0
        for iI, i in enumerate(tree1):
            shortTotals = []
            shortScores = []
            for jI, j in enumerate(tree2):
                # print(iI, i[0].label, i[1], i[2], jI, j[0].label, j[1], j[2])
                if np.random.rand() < samplePercentage:
                    if useDepth:
                        score += checkEquivalence(i[0], j[0], verbose=verbose) * (maxDepth - i[2] + 1)
                    else:
                        prunedTree1, shortTotal1 = pruneTree(i[0])
                        prunedTree2, shortTotal2 = pruneTree(j[0])
                        shortTotals.append(shortTotal1 + shortTotal2)
                        # print(convert2GuideTree(prunedTree1), convert2GuideTree(prunedTree2))
                        result = checkEquivalence(prunedTree1, prunedTree2, verbose=verbose)
                        shortScores.append(result * shortTotals[-1])
                        # print('result', result, 'short total', shortTotal1)
                        # score += resultTup[0] * i[1]
            if verbose:
                print(i, 'scores', shortScores, 'totals', shortTotals)
            if shortScores:
                score += np.max(shortScores)  # satisfies this node as long as one node in j matches with it
                total += shortTotals[np.argmax(shortScores)]
                if verbose:
                    print("added", np.max(shortScores), shortTotals[np.argmax(shortScores)])

        if useDepth:
            total = round(sum([maxDepth - tup[2] + 1 for tup in tree1]) * samplePercentage, 1)

        coverage = score / total
        print('Trees ' + str(showI) + ':', score, '/', total, '=', coverage)
        avgCoverage += min(coverage, 1)
    avgCoverage /= len(nodeLst[0])
    print("Average Coverage", avgCoverage)
    return avgCoverage


def pruneTree(tree):
    global prunePercentage
    total = 1
    newTree = Node(tree.label)
    for child in tree.children:
        if np.random.rand() < prunePercentage:
            resultTup = pruneTree(child)
            newTree.addkid(resultTup[0])
            total += resultTup[1]
    return newTree, total


def checkEquivalence(tree1, tree2, verbose=False):
    if tree1.label != tree2.label and not (isinstance(tree1.label, list) and isinstance(tree2.label, list)) \
            or tree1.label is None or tree2.label is None:
        # print("unequal label", tree1.label, tree2.label)
        return 0
    elif tree1.label in nodeSymbols:
        if tree1.children == tree2.children == []:
            return 1
        treeTup = [tree1, tree2]
        pad = len(tree1.children) - len(tree2.children)
        treeTup[int(math.copysign(1, pad) / 2 + 0.5)].children += [Node(None) for i in range(abs(pad))]
        seq1 = [i for i in range(len(tree1.children))]
        availableChoices = seq1.copy()
        seq1 = ''.join([str(val) for val in seq1])
        seq2 = ''
        score = 0
        for child1 in tree1.children:
            comparison = []
            for child2 in tree2.children:
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
            seq2 += str(choice)
            score += np.max(comparison)
        if tree1.label in nodeSymbols[0]:
            score = score / len(tree1.children)
        else:
            score = score / len(tree1.children) * Levenshtein.ratio(seq1, seq2)
        if verbose:
            print('score', score, seq1, seq2)
        return score
    else:
        if verbose:
            print(tree1.label, tree2.label)
        labelSimilarity = getLabelSimilarity(tree1.label, tree2.label)
        if verbose:
            print("similarity", labelSimilarity)
        return labelSimilarity

def main():
    global nodeLst
    if initCompare:
        trees = [sourceTree, testTree]
        labelCounter = {}
        for i, trainTree in enumerate(sourceTree):
            print('Train Tree', i, trainTree)
            graph = nx.DiGraph()
            trainTree = convert2NodeTree(trainTree, labelCounter=labelCounter, graph=graph)
            nodeLst[0].append(flattenList(extractNodes(trainTree, needTotal=True, needDepth=False)))
        videos, layerLst, zeroLayerTrees = improc.calcLayered(testTree, normalize=False, reverse=True, needZero=True, keyLen=len(sourceTree), verbose=True, fillDyn=False, mirrorMode=True)
        nodeLst[1] = [convert2NodeTree(tree, graph=nx.DiGraph()) for i, tree in enumerate(zeroLayerTrees)]
        print('Likelihood', getLikelihood(useTotal=True, verbose=True))
        nodeLst = [[], []]
        graphs = [None, None]
    learnedTree, successIter, bestLikelihood = \
        treeSubstitutionMCMC(sourceTree, numIterations=10000, verbose=1, showTree=True,
                             showImage=True,
                             replaceText=True, useTotal=True, useDepth=False, keyLen=10, showZero=True,
                             showSource=True, useSymbols=False, mirrorMode=True, precision=5)
    # verbose: 1 = basic, 2 = label comparison, 3 = layer calculations
    print("Learned Tree: ", convert2GuideTree(learnedTree), "Best Likelihood", bestLikelihood,
          "Successful Iterations", successIter)


if __name__ == "__main__":
    main()