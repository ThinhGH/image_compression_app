import numpy as np
import heapq
from collections import Counter

ZIGZAG_INDEX = [
 (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),
 (0,3),(1,2),(2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),
 (0,5),(1,4),(2,3),(3,2),(4,1),(5,0),
 (6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),
 (0,7),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1),(7,0),
 (7,1),(6,2),(5,3),(4,4),(3,5),(2,6),(1,7),
 (2,7),(3,6),(4,5),(5,4),(6,3),(7,2),
 (7,3),(6,4),(5,5),(4,6),(3,7),
 (4,7),(5,6),(6,5),(7,4),
 (7,5),(6,6),(5,7),
 (6,7),(7,6),
 (7,7)
]

def zigzag(block):
    return np.array([block[i, j] for i, j in ZIGZAG_INDEX])

def inverse_zigzag(arr):
    block = np.zeros((8,8))
    for k, (i, j) in enumerate(ZIGZAG_INDEX):
        block[i, j] = arr[k]
    return block
def rle_encode(ac):
    result = []
    zeros = 0
    for v in ac:
        if v == 0:
            zeros += 1
        else:
            result.append((zeros, int(v)))
            zeros = 0
    result.append((0, 0))  # EOB
    return result

def rle_decode(rle):
    ac = []
    for zeros, val in rle:
        if (zeros, val) == (0, 0):
            break
        ac.extend([0]*zeros)
        ac.append(val)
    while len(ac) < 63:
        ac.append(0)
    return np.array(ac)
class Node:
    def __init__(self, sym, freq):
        self.sym = sym
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq
def build_tree(data):
    freq = Counter(data)
    heap = [Node(k,v) for k,v in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        p = Node(None, a.freq + b.freq)
        p.left, p.right = a, b
        heapq.heappush(heap, p)

    return heap[0]
def build_codebook(node, prefix="", book=None):
    if book is None:
        book = {}
    if node.sym is not None:
        book[node.sym] = prefix
    else:
        build_codebook(node.left, prefix+"0", book)
        build_codebook(node.right, prefix+"1", book)
    return book
def encode_bits(data, codebook):
    return ''.join(codebook[x] for x in data)

def decode_bits(bits, tree):
    out = []
    node = tree
    for b in bits:
        node = node.left if b == '0' else node.right
        if node.sym is not None:
            out.append(node.sym)
            node = tree
    return out

