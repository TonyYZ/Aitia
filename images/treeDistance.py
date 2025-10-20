class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children is not None else []

def tree_edit_distance(tree1, tree2):
    n = len(tree1)
    m = len(tree2)
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if are_subtrees_similar(tree1[i - 1], tree2[j - 1]) else 1

            insertion_cost = dp[i][j - 1] + 1
            deletion_cost = dp[i - 1][j] + 1
            rename_cost = dp[i - 1][j - 1] + 1

            dp[i][j] = min(insertion_cost, deletion_cost, rename_cost)

    return dp[n][m]

def are_subtrees_similar(subtree1, subtree2):
    # Compare unordered subtrees using Jaccard similarity
    set1 = extract_nodes(subtree1)
    set2 = extract_nodes(subtree2)
    jaccard_sim = calculate_jaccard_similarity(set1, set2)

    # Decide whether the subtrees are similar based on Jaccard similarity
    similarity_threshold = 0.7  # Adjust as needed
    return jaccard_sim >= similarity_threshold

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0

def extract_nodes(node):
    nodes = set([node.label])
    for child in node.children:
        nodes.update(extract_nodes(child))
    return nodes

# Example trees
tree1 = TreeNode("A", [TreeNode("B"), TreeNode("C", [TreeNode("D"), TreeNode("E")])])
tree2 = TreeNode("A", [TreeNode("C", [TreeNode("E"), TreeNode("D")]), TreeNode("B")])

# Calculate modified Tree Edit Distance
ted = tree_edit_distance([tree1], [tree2])
print("Modified Tree Edit Distance:", ted)
