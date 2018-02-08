from __future__ import print_function
import malis
import numpy as np
from scipy.ndimage.measurements import center_of_mass

def find_cc_2d(ids):

    slices = []

    max_id = np.uint64(0)
    for z in range(ids.shape[0]):
        # ids need to be 3D for malis...
        frame = np.array([ids[z]])
        affs = malis.seg_to_affgraph(frame, malis.mknhood3d())
        nodes = malis.connected_components_affgraph(
                affs,
                malis.mknhood3d())
        nodes = nodes[0][0]
        bg = nodes==0
        max_slice_id = nodes.max()
        nodes += max_id
        nodes[bg] = 0
        max_id += max_slice_id
        slices.append(nodes)

    return np.array(slices)

def find_centers(ids):

    all_ids = np.unique(ids)
    coms = center_of_mass(np.ones_like(ids), ids, all_ids)

    return { i: l for i, l in zip(all_ids, coms) if i != 0 }

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def find_possible_edges(ids_prev, ids_next, nodes_prev, nodes_next):

    overlay = np.array([
        ids_prev.flatten(),
        ids_next.flatten(),
        nodes_prev.flatten(),
        nodes_next.flatten()])
    uniques = np.unique(overlay, axis=1)

    possible_edges = {}
    for id_p, id_n, node_p, node_n in zip(uniques[0], uniques[1], uniques[2], uniques[3]):
        if id_p == id_n:
            if id_p not in possible_edges:
                possible_edges[id_p] = []
            possible_edges[id_p].append((node_p, node_n))

    return possible_edges

def find_edges_between(ids_prev, ids_next, nodes_prev, nodes_next):

    edges = []

    possible_edges = find_possible_edges(
        ids_prev,
        ids_next,
        nodes_prev,
        nodes_next)

    # get center of masses of nodes
    locations = find_centers(nodes_prev)
    locations.update(find_centers(nodes_next))

    # print("Locations:")
    # print(locations)

    # for each id
    for i, candidates in possible_edges.iteritems():

        # continuation
        if len(candidates) == 1:

            # print("%d continues"%i)
            edges.append(candidates[0])

        else:

            # print("%d does something complex"%i)
            prev_nodes = set([p for (p, n) in candidates])
            next_nodes = set([p for (p, n) in candidates])

            pairs = []
            for (p, n) in candidates:
                distance = dist(locations[p], locations[n])
                pairs.append((distance, p, n))
            pairs.sort()
            # print("all possible continuations: %s"%pairs)

            # greedily match closest continuations
            for (d, pn, nn) in pairs:
                if pn in prev_nodes and nn in next_nodes:
                    # print("pick %s"%([d, pn, nn]))
                    edges.append((pn, nn))
                    prev_nodes.remove(pn)
                    next_nodes.remove(nn)

            # left over next nodes are splits, assign to closest prev
            for (d, pn, nn) in pairs:
                if nn in next_nodes:
                    # print("pick %s"%([d, pn, nn]))
                    edges.append((pn, nn))
                    next_nodes.remove(nn)

    return edges

def find_edges(ids, nodes):

    edges = []

    for z in range(ids.shape[0] - 1):
        print("Searching for edges out of frame ", z)
        edges.append(
            find_edges_between(
                ids[z], ids[z+1], nodes[z], nodes[z+1]))

    return edges

class Track:

    def __init__(self, start, end, label, parent):
        self.start = start
        self.end = end
        self.label = label
        self.parent = parent
        self.nodes = []

    def __repr__(self):

        parent = None
        if self.parent:
            parent = self.parent.label
        return "%d: [%d, %s], nodes %s, parent track: %s"%(self.label,
                self.start, self.end, self.nodes, parent)

def contract(edges, nodes):

    tracks = []
    node_to_track = {}

    # for each frame
    for z in range(len(edges) + 1):

        # print("Contracting in z=%d"%z)

        in_nodes = {}
        out_nodes = {}

        # for all edges leaving the current frame
        if z < len(edges):
            for p, n in edges[z]:
                if p in out_nodes:
                    out_nodes[p].append(n)
                else:
                    out_nodes[p] = [n]
        # for all edges entering the current frame
        if z > 0:
            for p, n in edges[z - 1]:
                if n in in_nodes:
                    in_nodes[n].append(p)
                else:
                    in_nodes[n] = [p]

        # for each node in the current frame
        frame_nodes = list(np.unique(nodes[z]))
        if 0 in frame_nodes:
            frame_nodes.remove(0)
        for node in frame_nodes:
            if node not in in_nodes:
                in_nodes[node] = []
            if node not in out_nodes:
                out_nodes[node] = []

        for node in frame_nodes:

            if len(in_nodes[node]) == 0:

                # start of new track with label 'node'
                # print("Start of %d"%node)

                track = Track(z, None, node, None)
                tracks.append(track)
                node_to_track[node] = track

            elif len(in_nodes[node]) == 1 and len(out_nodes[node]) == 1:

                # continuation of previous track or right after split

                # there is already a track for this node it was created by a
                # split right before, then there is nothing to do
                if node not in node_to_track:
                    # print("Continuation of %d"%node)
                    prev_node = in_nodes[node][0]
                    track = node_to_track[prev_node]
                    node_to_track[node] = track
                # else:
                    # print("%d is first after split"%node)

            elif len(in_nodes[node]) == 1 and len(out_nodes[node]) > 1:

                # split -> end of track
                # print("Split of %d"%node)

                prev_node = in_nodes[node][0]
                track = node_to_track[prev_node]
                node_to_track[node] = track
                track.end = z
                # print("Ending track %s"%track)
                # create new tracks for subsequent nodes
                for nn in out_nodes[node]:
                    new_track = Track(z + 1, None, nn, track)
                    tracks.append(new_track)
                    node_to_track[nn] = new_track
                # print("Have tracks: %s"%tracks)

            if len(out_nodes[node]) == 0:

                # end of track
                # print("End of %d"%node)

                if node not in node_to_track:
                    prev_node = in_nodes[node][0]
                    track = node_to_track[prev_node]
                    node_to_track[node] = track
                else:
                    track = node_to_track[node]
                track.end = z

    for node, track in node_to_track.items():
        track.nodes.append(node)

    return tracks

def replace(array, old_values, new_values):

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]

def relabel(nodes, tracks):

    old_values = []
    new_values = []
    for track in tracks:
        for node in track.nodes:
            old_values.append(node)
            new_values.append(track.label)

    old_values = np.array(old_values, dtype=nodes.dtype)
    new_values = np.array(new_values, dtype=nodes.dtype)

    return replace(nodes, old_values, new_values)

def ids_to_track_graph(ids):

    # uniquely label each component in each frame, let's call them "nodes"
    print(ids.shape)
    nodes = find_cc_2d(ids)
    print(nodes.shape)

    # print("Original IDs:")
    # print(ids)
    # print("Nodes:")
    # print(nodes)

    # based on original ids, create a tracking graph by introducing directed
    # edges between nodes
    edges = find_edges(ids, nodes)

    # print("Edges:")
    # print(edges)

    # contract the graph by replacing each chain with a single node, remember
    # the original nodes
    tracks = contract(edges, nodes)

    ids = relabel(nodes, tracks)

    return tracks, ids
