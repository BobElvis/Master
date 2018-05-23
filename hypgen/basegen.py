import heapq


def normalize(nodes):
    # Normalize. Sum all probabilities equals to one.
    sum = 0
    for node in nodes:
        sum += node.probability
    for node in nodes:
        node.probability /= sum


def nlargest(hypothesises, K):
    return heapq.nlargest(K, hypothesises, key=lambda x: x.probability)