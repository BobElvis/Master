from hypgen.mhtgen import MHTGen
from model.model import *


class MHTGenDel(MHTGen):
    def __init__(self, mht, prob_detect, prob_delete):
        super().__init__(mht, prob_detect)
        self.prob_delete = prob_delete

        self.prob_prior_base = (1 - self.prob_detect - self.prob_delete)
        self.existing_base = self.prob_detect/self.prob_prior_base
        self.prior_prob = self.existing_base
        self.delete_base = self.prob_delete/self.prob_prior_base

    def gen_hyp_scan_node(self, scan_parent, leaf_node):
        # For each track not innovated, create one keep and one delete node.

        # Create del node
        leaves = [HypDel(leaf_node.probability, leaf_node.parent, leaf_node.track_nodes, scan_parent.track_nodes_del.copy())]
        for idx, track_node in enumerate(leaf_node.track_nodes):
            if track_node.isPosterior:
                continue
            innovated_node = track_node.innovateTrack(None, self.mht.measurementModel)

            new_leaves = []
            for leaf_node in leaves:
                # 1. Keep:
                track_nodes = [t if t != track_node else innovated_node for t in leaf_node.track_nodes]
                track_nodes_del = leaf_node.track_nodes_del.copy()
                prob = leaf_node.probability
                new_leaves.append(HypDel(prob, leaf_node, track_nodes, track_nodes_del))

                # 2. Delete:
                track_nodes = [t for t in leaf_node.track_nodes if t != track_node]
                track_nodes_del = leaf_node.track_nodes_del.copy()
                track_nodes_del.append(track_node)
                prob = leaf_node.probability * self.delete_base
                new_leaves.append(HypDel(prob, leaf_node, track_nodes, track_nodes_del))

            leaves = new_leaves

        # Create scan nodes. All tracks should already be innovated or removed:
        scan_nodes = []
        for leaf_node in leaves:
            prob = leaf_node.probability * (self.prob_prior_base ** len(scan_parent.track_nodes))
            track_nodes = leaf_node.track_nodes.copy()  # NOT copied.
            track_nodes_del = leaf_node.track_nodes_del.copy()
            scan_nodes.append(HypScan(prob, scan_parent, track_nodes, track_nodes_del))
        return scan_nodes

    def createScanNode(self, prob, parent, track_nodes):
        return HypScan(prob, parent, track_nodes)


