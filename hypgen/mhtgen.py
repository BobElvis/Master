from multivariate import multi_gaussian


class MHTGen:
    def __init__(self, mht, prob_detect):
        self.mht = mht
        self.first_m = False
        self.prob_detect = prob_detect
        self.prior_prob = self.prob_detect / (1 - self.prob_detect)

    def gen_hyp_meas(self, measurement, leaves, new_target):
        new_leaves = []

        for hyp_node in leaves:
            # Case 1: Clutter node
            hypnode_clutter = self.createClutterNode(hyp_node)
            new_leaves.append(hypnode_clutter)

            # Case 2: Existing tracks (gated and not already used)
            hypnodes_existing = self.createExistingNodes(hyp_node, measurement)
            new_leaves.extend(hypnodes_existing)

            # Case 3: New target
            hypnode_new_target = self.createNewTargetNode(hyp_node, new_target.source)
            new_leaves.append(hypnode_new_target)

        return new_leaves

    def gen_hyp_scan_node(self, scan_parent, leaf_node):
        prob = leaf_node.probability * (1 - self.prob_detect) ** len(scan_parent.track_nodes)
        track_nodes = self.innovate_track_nodes(leaf_node.track_nodes, self.mht.measurementModel)
        return HypScan(prob, scan_parent, track_nodes, scan_parent.track_nodes_del.copy())

    def gen_hyps(self, cluster):
        # Using measurements, create new hyp scans.

        new_targets = []  # All new targets created by the measurements.
        measurements = sorted(cluster.gated_measurements, key=lambda x: x.idx)
        leaves = cluster.leaves

        for idx, measurement in enumerate(measurements):
            self.first_m = (idx == 0)

            # New target:
            new_target = self.mht.createNewTarget(measurement)
            new_targets.append(new_target)

            # Main hypothesis generation:
            leaves = self.gen_hyp_meas(measurement, leaves, new_target)

        # Create the new scan node leaves:
        get_parent = (lambda nx: nx) if len(measurements) == 0 else (lambda nx: nx.parent)
        cluster.leaves = []
        for node in leaves:
            scan_node = self.gen_hyp_scan_node(get_parent(node), node)
            if type(scan_node) is list:
                cluster.leaves.extend(scan_node)
            else:
                cluster.leaves.append(scan_node)

        # Update parent children list:
        for node in cluster.leaves:
            node.parent.children.append(node)

        self.normalize(cluster.leaves)
        return new_targets

    @staticmethod
    def innovate_track_nodes(track_nodes, measurement_model):
        # Returns a copy of the track_nodes where not posterior track_nodes are innovated with None.
        return [t if t.isPosterior else t.innovateTrack(None, measurement_model) for t in track_nodes]

    def createClutterNode(self, base_node):
        new_track_nodes = base_node.track_nodes.copy()
        return self.createNode(base_node.probability * self.mht.densityFalseReports, base_node, new_track_nodes)

    def createExistingNodes(self, base_node, measurement):
        newNodes = []
        ext_base_prob = base_node.probability * self.prior_prob
        for idx, trackNode in enumerate(base_node.track_nodes):
            if trackNode.isPosterior or not trackNode.isMeasurementGated(measurement):
                continue

            # Copy existing tracks and innovate the one assigned the measurement
            newTrackNodes = base_node.track_nodes.copy()
            innovatedNode = trackNode.innovateTrack(measurement, self.mht.measurementModel)
            newTrackNodes[idx] = innovatedNode

            # Calculate probability
            exist_prob = ext_base_prob * multi_gaussian(measurement.value - trackNode.z_hat, trackNode.B)
            new_node = self.createNode(exist_prob, parent_node=base_node, track_nodes=newTrackNodes)
            newNodes.append(new_node)
        return newNodes

    def createNewTargetNode(self, base_node, new_target_track_node):
        new_track_nodes = base_node.track_nodes.copy()
        new_track_nodes.append(new_target_track_node)
        return self.createNode(base_node.probability * self.mht.densityUnknownTargets, base_node, new_track_nodes)

    def createNode(self, prob, parent_node, track_nodes):
        if not self.first_m:
            parent_node = parent_node.parent
        return HypMeas(prob, parent_node, track_nodes)
