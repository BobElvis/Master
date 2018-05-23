

def leaf_node_tostr(leaf_node):
    hyp_str = []
    getHypothesisStr(leaf_node, hyp_str)
    tracks = []
    conf_tracks = []
    for track_leaf in leaf_node.track_nodes:
        tracks.append(getFullTrackFromLeaf(track_leaf))
    for track_leaf in leaf_node.track_nodes_del:
        conf_tracks.append(getFullTrackFromLeaf(track_leaf))
    return '{}{:.4g}{} CONF:{} - {}'.format(hyp_str, leaf_node.probability, tracks, conf_tracks, leaf_node.sol)


def getHypothesisStr(node, hypstr):
    if node is None:
        return
    getHypothesisStr(node.parent, hypstr)
    if type(node) is not HypScanJoin:
        #node_id = 0 if node.target is None else node.target.idx
        #hypstr.append('{}= {}'.format(node.measurement, node_id))
        pass
    else:
        getHypothesisStr(node.parent2, hypstr)


def printNodeFull(node, prev):
    if type(node) is HypScanJoin:
        node_id = 0 if node.target is None else node.target.idx
    else:
        node_id = 'J'
    prev.append(node_id)
    if len(node.children) == 0:
        print("{}: {:.4g}".format(prev, node.probability))
    for child in node.children:
        printNodeFull(child, prev.copy())


def printNode(node, depth, indent):
    if type(node) is HypScanJoin:
        node_id = 0 if node.target is None else node.target.idx
    else:
        node_id = 'J'
    print("{}{}: {:.4g}".format(" " * (indent * depth), node_id, node.probability))

    for child in node.children:
        printNode(child, depth + 1, indent)


def getFullTrackFromLeaf(track_leaf):
    track_str = []
    getFullTrackFromLeafAux(track_leaf, track_str)
    return '{}{}'.format(track_leaf.target, track_str)


def getFullTrackFromLeafAux(track_node, track_str):
    if track_node is None:
        return
    getFullTrackFromLeafAux(track_node.parent, track_str)
    track_str.append(track_node.measurement)


def printTrackTree(targets, indent):
    for target in sorted(targets, key=lambda x:x.idx):
        print('{}:'.format(target))
        printTrackTreeAux(target.source, 1, indent)


def printTrackTreeAux(node, depth, indent):
    print("{}{}:{} g:{}".format(" " * (indent * depth), node.measurement, node.isPosterior, node.gated_measurements))
    for child_node in sorted(node.children(), key=lambda x: 0 if x.measurement is None else x.measurement.idx):
        printTrackTreeAux(child_node, depth+1, indent)