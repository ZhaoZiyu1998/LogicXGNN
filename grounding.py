import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
import torch
import numpy
from networkx.algorithms import isomorphism
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt
import re
import os
def double_check_used_iso_predicate_node(used_alone_predicates, used_iso_predicate_node, idx_predicates_mapping, filter_num=2):
    keys_to_delete = []  # Collect keys to delete to avoid modifying dict during iteration
    
    for k, v in used_iso_predicate_node.items():
        keys = list(v.keys())
        if len(keys) != 2:
            raise ValueError(f"Expected exactly two keys for used_iso_predicate_node[{k}], got {keys}")
        
        other_k = keys[0] if keys[1] == k else keys[1]
        
        if len(v[other_k]) <= filter_num:
            wl_hash = idx_predicates_mapping[k][0]
            used_alone_predicates[wl_hash] = k, used_iso_predicate_node[k][k]
            keys_to_delete.append(k)
    
    # Delete keys after iteration to avoid RuntimeError
    for k in keys_to_delete:
        del used_iso_predicate_node[k]

    return used_alone_predicates, used_iso_predicate_node





def analyze_used_predicate_nodes(predicate_node, predicate_to_idx, used_predicates):
    """
    Analyze used predicate nodes and separate them into alone and iso categories.
    
    Args:
        predicate_node (dict): Dictionary with keys of form (str, binary) and list values
        predicate_to_idx (dict): Dictionary mapping (str, binary) tuples to indices
        used_predicates (list): List of used predicate indices
        
    Returns:
        tuple: (used_alone_predicates, used_iso_predicate_node)
            - used_alone_predicates: dict with keys that appear only once and are used
            - used_iso_predicate_node: dict where key is used predicate idx that has binary counterpart,
                                      value is dict with idx keys for (str,0) and (str,1)
    """
    # Convert used_predicates to set for faster lookup
    used_set = set(used_predicates)
    
    # Create reverse mapping from idx to (str, binary)
    idx_to_predicate = {idx: key for key, idx in predicate_to_idx.items()}
    
    # Count occurrences of each string key in predicate_node
    str_counts = {}
    for key in predicate_node.keys():
        str_part = key[0]  # Extract string part
        str_counts[str_part] = str_counts.get(str_part, 0) + 1
    
    # Initialize result dictionaries
    used_alone_predicates = {}
    used_iso_predicate_node = {}
    
    # Process each used predicate
    for idx in used_predicates:
        if idx not in idx_to_predicate:
            continue  # Skip if idx not in predicate_to_idx
            
        key = idx_to_predicate[idx]
        str_part, binary_part = key[0], key[1]
        
        # Check if this key exists in predicate_node
        if key not in predicate_node:
            continue
            
        if str_counts[str_part] == 1:
            # String appears only once - add to used_alone_predicates
            used_alone_predicates[key[0]] = predicate_to_idx[key], predicate_node[key]
        else:
            # String appears multiple times - add to used_iso_predicate_node
            if idx not in used_iso_predicate_node:
                used_iso_predicate_node[idx] = {}
            
            # Find both binary variants for this string
            key_0 = (str_part, 0)
            key_1 = (str_part, 1)
            
            if key_0 in predicate_to_idx and key_0 in predicate_node:
                idx_0 = predicate_to_idx[key_0]
                used_iso_predicate_node[idx][idx_0] = predicate_node[key_0]
                
            if key_1 in predicate_to_idx and key_1 in predicate_node:
                idx_1 = predicate_to_idx[key_1]
                used_iso_predicate_node[idx][idx_1] = predicate_node[key_1]

    used_alone_predicates, used_iso_predicate_node = double_check_used_iso_predicate_node(used_alone_predicates, used_iso_predicate_node, idx_to_predicate, filter_num=10)
    
    return used_alone_predicates, used_iso_predicate_node


def plot_k_hop_subgraph(
    graph_idx, center_index,
    x_dict, edge_dict, atom_type_dict,
    nodes_attr=0, title_=None, k=0,
    plot=1, show_node_idx=1,
    ax=None, title_position="top", grounding=False
):
    """
    Extract and (optionally) plot a k-hop subgraph around a center node.

    Args:
        graph_idx (int): Graph index in dataset
        center_index (int): Center node index
        x_dict (dict): Node features per graph
        edge_dict (dict): Edge index per graph
        atom_type_dict (dict): Mapping atom type index -> name
        nodes_attr (bool): Whether to include node attributes in return
        title_ (str): Optional title for the plot
        k (int): Hop distance
        plot (bool): Whether to plot or not
        show_node_idx (bool): Show node indices + atom names if True
        ax (matplotlib.axes.Axes): Optional axis to draw on (for subplots)
        title_position (str): "top" or "bottom"

    Returns:
        tuple: (edges, nodes_attributes, center_index, wl_hash)
    """
    edge_tensor = edge_dict[graph_idx]
    x_tensor = x_dict[graph_idx]

    # Extract k-hop subgraph
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        center_index, k, edge_tensor, relabel_nodes=True
    )

    # Create PyG Data object
    sub_x = x_tensor[subset]

    sub_data = Data(x=sub_x, edge_index=sub_edge_index)

    # Convert to NetworkX
    G = to_networkx(sub_data, to_undirected=True)

    nodes_attributes = {}
    if nodes_attr:
        for node in G.nodes:
            original_node_idx = subset[node].item()

            one_hot_vec = sub_x[node]
            atom_type_idx = torch.argmax(one_hot_vec).item()
            
            atom_name = atom_type_dict.get(atom_type_idx, "UNK")
            G.nodes[node]["original_idx"] = original_node_idx
            G.nodes[node]["atom_name"] = atom_name
            nodes_attributes[original_node_idx] = [atom_name, one_hot_vec.tolist()]

    # Extract edges with original indices
    edges = [(subset[u].item(), subset[v].item()) for u, v in sub_edge_index.t().tolist()]

    if grounding:
        wl_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr='atom_name')
    else:
        wl_hash = nx.weisfeiler_lehman_graph_hash(G)

    if plot:
        pos = nx.spring_layout(G, seed=42)
        target_ax = ax if ax is not None else plt.gca()
        target_ax.clear()

        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, ax=target_ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", ax=target_ax)

        if nodes_attr:
            for node, (x, y) in pos.items():
                if show_node_idx:
                    label = f"{G.nodes[node]['original_idx']}\n{G.nodes[node]['atom_name']}"
                    target_ax.text(x, y - 0.05, label, fontsize=8, ha="center", color="blue")
                else:
                    label = f"{G.nodes[node]['atom_name']}"
                    target_ax.text(x, y - 0.02, label, fontsize=8, ha="center", color="blue")

        if title_:
            if title_position == "top":
                target_ax.set_title(title_, fontsize=12)
            elif title_position == "bottom":
                target_ax.set_title("")
                target_ax.text(0.5, -0.05, title_, fontsize=12,
                               ha="center", transform=target_ax.transAxes)
        else:
            target_ax.set_title(f"{k}-hop subgraph around node {center_index}")
        #fig= target_ax.get_figure()
        #fig.savefig(f"subgraph_{graph_idx}_center_{center_index}_k_{k}.png")
        target_ax.axis("off")

    return edges, nodes_attributes, center_index, wl_hash

def stable_orbits_with_anchor(edge_list, anchor):
    """
    Compute stable orbits and reconstruction blueprint for isomorphic graphs.
    Uses multiple criteria for stable ordering including distance from anchor.
    
    Args:
        edge_list: List of (node1, node2) tuples representing edges
        anchor: Anchor node (unique, always goes in first orbit by itself)
             
    Returns:
        tuple: (label_map, orbits_sorted)
        - label_map: dict mapping node -> orbit label
        - orbits_sorted: list of frozensets representing sorted orbits
    """
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Precompute distances from anchor to all nodes
    try:
        distances_from_anchor = nx.single_source_shortest_path_length(G, anchor)
    except nx.NetworkXError:
        # If anchor is isolated or graph is disconnected, handle gracefully
        distances_from_anchor = {anchor: 0}
        for node in G.nodes():
            if node != anchor:
                try:
                    distances_from_anchor[node] = nx.shortest_path_length(G, anchor, node)
                except nx.NetworkXNoPath:
                    distances_from_anchor[node] = float('inf')
    
    # Compute all automorphisms (not restricting to anchor-fixing ones)
    GraphMatcher = isomorphism.GraphMatcher
    gm = GraphMatcher(G, G, node_match=lambda x, y: True)
    automorphisms = list(gm.isomorphisms_iter())
    
    # Compute orbits for all nodes under all automorphisms
    orbit_dict = {node: set([node]) for node in G.nodes()}  # Initialize each node in its own orbit
    
    for mapping in automorphisms:
        for node, image in mapping.items():
            orbit_dict[node].add(image)
    
    # Find the orbit containing the anchor
    anchor_orbit_nodes = orbit_dict[anchor]
    anchor_orbit = frozenset(anchor_orbit_nodes)
    
    # Convert orbit sets to frozensets for remaining nodes
    orbits = [anchor_orbit]  # Anchor orbit always first
    visited = set(anchor_orbit_nodes)  # Mark all nodes in anchor orbit as visited
    
    for node in G.nodes():
        if node not in visited:
            orb = frozenset(orbit_dict[node])
            orbits.append(orb)
            visited.update(orb)
    
    # Sort non-anchor orbits stably by multiple criteria:
    # 1. Size ascending
    # 2. Sorted list of node degrees inside orbit
    # 3. Sorted list of distances from anchor inside orbit
    # 4. Sorted node IDs for ultimate stability
    def orbit_key(orb):
        if anchor in orb:
            return (0,)  # Anchor orbit always first
        
        degrees = sorted([G.degree(n) for n in orb])
        distances = sorted([distances_from_anchor.get(n, float('inf')) for n in orb])
        node_ids = sorted(orb)
        
        return (1, len(orb), degrees, distances, node_ids)
    
    orbits_sorted = sorted(orbits, key=orbit_key)
    
    # Build label map: anchor orbit = 'anchor', others labeled by order
    label_map = {}
    orbit_counter = 1
    
    for i, orb in enumerate(orbits_sorted):
        if anchor in orb:
            label = 'anchor'
        else:
            label = f'orbit_{orbit_counter}'
            orbit_counter += 1
        
        for node in orb:
            label_map[node] = label
    
    return label_map, orbits_sorted



def merge_orbits_by_distance_with_paths(edge_list, orbits_sorted, label_map):
    """
    Merge orbits based on distance to unique nodes (single-node orbits).
    For distant orbits (distance > 1), merge each node with its immediate predecessor
    on the path to the unique node to create edge-based representations like (predecessor, distant_node).
    
    NEW: If there are no unique nodes (all orbits have length > 1), merge nodes to edges
    based on degree analysis - higher degree nodes become "starts" and connect to their neighbors.
    
    Args:
        edge_list: List of (node1, node2) tuples representing edges
        orbits_sorted: List of frozensets representing sorted orbits from stable_orbits_with_anchor
        label_map: Dict mapping node -> orbit label from stable_orbits_with_anchor
    
    Returns:
        tuple: (new_orbits_sorted, new_label_map)
        - new_orbits_sorted: Updated list of merged orbits (may contain tuples for path-merged nodes)
        - new_label_map: Updated label mapping after merging
    """
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Separate unique nodes (single-node orbits) from multi-node orbits
    unique_orbits = [orbit for orbit in orbits_sorted if len(orbit) == 1]
    multi_orbits = [orbit for orbit in orbits_sorted if len(orbit) > 1]
    
    # NEW LOGIC: Handle case where there are no unique nodes
    if not unique_orbits and multi_orbits:
        return handle_no_unique_nodes_case(edge_list, multi_orbits, label_map, G)
    
    def get_orbit_distance_to_unique_node(orbit, unique_node):
        """Get minimum distance from any node in orbit to unique_node"""
        min_dist = float('inf')
        for node in orbit:
            try:
                dist = nx.shortest_path_length(G, node, unique_node)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
        return min_dist
    
    def get_immediate_predecessor(node, unique_node):
        """Get the immediate predecessor of node on the path to unique_node"""
        try:
            path = nx.shortest_path(G, node, unique_node)
            if len(path) >= 2:
                # Return the node right before the current node (immediate predecessor)
                return path[-2]  # Second to last node in path (predecessor of unique_node)
            return None
        except nx.NetworkXNoPath:
            return None
    
    # Classify each multi-orbit by distance to closest unique node
    distant_orbits = []  # orbits with distance > 1
    close_orbits = []    # orbits with distance = 1
    
    for orbit in multi_orbits:
        min_dist_to_any_unique = float('inf')
        closest_unique = None
        
        for unique_orbit in unique_orbits:
            unique_node = list(unique_orbit)[0]
            dist = get_orbit_distance_to_unique_node(orbit, unique_node)
            if dist < min_dist_to_any_unique:
                min_dist_to_any_unique = dist
                closest_unique = unique_node
        
        if min_dist_to_any_unique == 1:
            close_orbits.append((orbit, closest_unique))
        elif min_dist_to_any_unique > 1:
            distant_orbits.append((orbit, closest_unique))
    
    # Start building new orbits
    new_orbits = list(unique_orbits)  # Keep all unique orbits
    used_pathway_nodes = set()  # Track nodes that become pathway nodes
    
    # Process distant orbits - merge with immediate predecessors
    for orbit, closest_unique in distant_orbits:
        path_merged_orbit = set()
        
        for node in orbit:
            predecessor = get_immediate_predecessor(node, closest_unique)
            if predecessor is not None:
                path_merged_orbit.add((predecessor, node))
                used_pathway_nodes.add(predecessor)
        
        if path_merged_orbit:
            new_orbits.append(frozenset(path_merged_orbit))
    
    # Process close orbits - keep only those that don't have nodes used as pathway nodes
    for orbit, closest_unique in close_orbits:
        # Check if this orbit should be dropped (any of its nodes are used as pathway nodes)
        if not any(node in used_pathway_nodes for node in orbit):
            new_orbits.append(orbit)
        # If orbit has nodes used as pathway nodes, it gets dropped (merged into path representations)
    
    # Create new label map
    new_label_map = {}
    orbit_counter = 1
    
    for orbit in new_orbits:
        if len(orbit) == 1 and not isinstance(list(orbit)[0], tuple):
            # Single node orbit - check if anchor
            node = list(orbit)[0]
            if node in label_map and label_map[node] == 'anchor':
                label = 'anchor'
            else:
                label = f'orbit_{orbit_counter}'
                orbit_counter += 1
            new_label_map[node] = label
        else:
            # Multi-node orbit or path-merged orbit
            label = f'orbit_{orbit_counter}'
            orbit_counter += 1
            for element in orbit:
                if isinstance(element, tuple):
                    # For path-merged elements, use the tuple as key
                    new_label_map[element] = label
                else:
                    new_label_map[element] = label
    
    return new_orbits, new_label_map


def handle_no_unique_nodes_case(edge_list, multi_orbits, label_map, G):
    """
    Handle the case where there are no unique nodes (all orbits have length > 1).
    Merge nodes to edges based on degree analysis - nodes with higher degrees become "starts"
    and connect to their neighbors in other orbits.
    
    Args:
        edge_list: List of (node1, node2) tuples representing edges
        multi_orbits: List of frozensets with length > 1
        label_map: Original label mapping
        G: NetworkX graph object
    
    Returns:
        tuple: (new_orbits_sorted, new_label_map)
    """
    # Calculate degrees for all nodes
    node_degrees = dict(G.degree())
    
    # Find the orbit with the highest degree nodes
    orbit_max_degrees = {}
    for i, orbit in enumerate(multi_orbits):
        max_degree_in_orbit = max(node_degrees[node] for node in orbit)
        orbit_max_degrees[i] = max_degree_in_orbit
    
    # Find the orbit with the highest maximum degree - this becomes the "start" orbit
    start_orbit_idx = max(orbit_max_degrees.keys(), key=lambda x: orbit_max_degrees[x])
    start_orbit = multi_orbits[start_orbit_idx]
    
    # Get the highest degree nodes from the start orbit
    start_orbit_degrees = [(node, node_degrees[node]) for node in start_orbit]
    start_orbit_degrees.sort(key=lambda x: x[1], reverse=True)  # Sort by degree descending
    
    # Create edge mappings from start orbit nodes to nodes in other orbits
    merged_edges = set()
    used_target_nodes = set()
    
    for start_node, _ in start_orbit_degrees:
        # Find all neighbors of this start node
        neighbors = list(G.neighbors(start_node))
        
        # Group neighbors by their orbits
        neighbor_orbit_map = {}
        for neighbor in neighbors:
            for i, orbit in enumerate(multi_orbits):
                if i != start_orbit_idx and neighbor in orbit:
                    if i not in neighbor_orbit_map:
                        neighbor_orbit_map[i] = []
                    neighbor_orbit_map[i].append(neighbor)
        
        # Create edges from start_node to neighbors in other orbits
        for orbit_idx, neighbor_nodes in neighbor_orbit_map.items():
            for neighbor_node in neighbor_nodes:
                if neighbor_node not in used_target_nodes:
                    merged_edges.add((start_node, neighbor_node))
                    used_target_nodes.add(neighbor_node)
    
    # Create the merged orbit containing all the edge tuples
    if merged_edges:
        new_orbits = [frozenset(merged_edges)]
    else:
        # Fallback: if no cross-orbit connections found, keep original orbits
        new_orbits = list(multi_orbits)
    
    # Create new label map
    new_label_map = {}
    orbit_counter = 1
    
    for orbit in new_orbits:
        label = f'orbit_{orbit_counter}'
        orbit_counter += 1
        
        for element in orbit:
            if isinstance(element, tuple):
                # For edge-based elements, use the tuple as key
                new_label_map[element] = label
            else:
                # For regular nodes
                new_label_map[element] = label
    
    return new_orbits, new_label_map


def create_expanded_encoding(x_tensor, node_indices_or_tuple, num_features):
    """
    Create expanded one-hot encoding for either single nodes or node pairs (edges).
    
    Args:
        x_tensor: Input tensor of shape (#nodes, #features)
        node_indices_or_tuple: Either a single node index or tuple (node_i, node_j)
        num_features: Number of original features
        
    Returns:
        torch.Tensor: Expanded one-hot encoding of length (num_features + num_features^2)
    """
    expanded_size = num_features + num_features * num_features  # 14 + 14*14 = 210
    expanded_encoding = torch.zeros(expanded_size)
    
    if isinstance(node_indices_or_tuple, tuple):
        # Handle edge case (i, j)
        node_i, node_j = node_indices_or_tuple
        
        # Find which positions are active in each node's one-hot encoding
        pos_i = torch.argmax(x_tensor[node_i]).item()  # Position of 1 in node_i's encoding
        pos_j = torch.argmax(x_tensor[node_j]).item()  # Position of 1 in node_j's encoding
        
        # Map to expanded encoding: edges start at position num_features
        # Edge (pos_i, pos_j) maps to position: num_features + pos_i * num_features + pos_j
        edge_pos = num_features + pos_i * num_features + pos_j
        expanded_encoding[edge_pos] = 1.0
        
    else:
        # Handle single node case
        node_idx = node_indices_or_tuple
        # Copy the original one-hot encoding to the first num_features positions
        expanded_encoding[:num_features] = x_tensor[node_idx]
    
    return expanded_encoding



# do cts feature encoding as well, now use frequencey encoding for discrete featutres, what encoding should be used for cts features??? maybe meaning ?? build edges for orbits that have more nodes, => making edges orbit seems work in discrete case? I guess it is useless for cts features? since (a,b) now a, b are cts are just dont tell u anything, in discrete case, we can use frequency encode, it does tell more info, am i right？





def predicate_features_z(x_tensor, merged_orbits):
    """
    Create new features Z by processing merged_orbits based on their element types.
    
    Args:
        x_tensor: Input tensor of shape (#nodes, #features) where each row is a one-hot encoding
        merged_orbits: List of frozensets containing node indices or composite terms (edges)
        
    Returns:
        torch.Tensor: Concatenated features based on merged_orbits processing
    """
    
    num_nodes, num_features = x_tensor.shape
    
    # Check if we need expanded encoding (if any composite terms exist)
    has_composite_terms = any(
        any(isinstance(elem, tuple) for elem in orbit) 
        for orbit in merged_orbits
    )
    
    if has_composite_terms:
        expanded_size = num_features + num_features * num_features
    else:
        expanded_size = num_features
    
    # Process each orbit in merged_orbits
    result_features = []
    
    for orbit in merged_orbits:
        orbit_list = list(orbit)
        
        if len(orbit_list) == 1:
            # Single element case
            elem = orbit_list[0]
            
            if has_composite_terms:
                # Use expanded encoding
                feature = create_expanded_encoding(x_tensor, elem, num_features)
            else:
                # Use original encoding
                if isinstance(elem, tuple):
                    raise ValueError("Composite terms found but expanded encoding not initialized")
                feature = x_tensor[elem]
                
        else:
            # Multiple elements - frequency-based encoding (sum)
            expanded_features = []
            
            for elem in orbit_list:
                if has_composite_terms:
                    elem_feature = create_expanded_encoding(x_tensor, elem, num_features)
                else:
                    if isinstance(elem, tuple):
                        raise ValueError("Composite terms found but expanded encoding not initialized")
                    elem_feature = x_tensor[elem]
                
                expanded_features.append(elem_feature)
            
            # Sum all features in this orbit (frequency-based encoding)
            feature = torch.stack(expanded_features).sum(dim=0)
        
        result_features.append(feature)
    
    # Concatenate all features in the order specified by merged_orbits
    result = torch.cat(result_features, dim=0)
    
    return result








def get_the_rule_iso_z(p_idx, used_iso_predicate_node, train_x_dict, atom_type_dict, train_edge_dict,  one_hot = 0,  k_hops=0,  depth= 5, adds = 0, verbose=1):
    keys_list = list(used_iso_predicate_node[p_idx].keys())
    p_idx_position = keys_list.index(p_idx) # tell if p_idx is 0 or 1

    b_0, b_1 = used_iso_predicate_node[p_idx].keys()
    # ----- For input z on class 0  -----
    z_res_0 = []
    instance_0 = []
    for graph_idx, center_index in used_iso_predicate_node[p_idx][b_0]:
        edges, nodes_attributes, center_index, wl_hash = plot_k_hop_subgraph(graph_idx, center_index, train_x_dict, train_edge_dict, atom_type_dict,  nodes_attr = one_hot, title_ = None, k=k_hops, plot=1, show_node_idx = 0)    
        original_label_map, original_orbits = stable_orbits_with_anchor(edges, center_index)
        merged_orbits, merged_label_map = merge_orbits_by_distance_with_paths(
            edges, original_orbits, original_label_map
        )
        z = predicate_features_z(train_x_dict[graph_idx],  merged_orbits)
        z_res_0.append(z)
        instance_0.append((graph_idx, center_index))

    # ----- For input z on class 1  -----
    z_res_1 = []
    instance_1 = []
    for graph_idx, center_index in used_iso_predicate_node[p_idx][b_1]:
        edges, nodes_attributes, center_index, wl_hash = plot_k_hop_subgraph(graph_idx, center_index, train_x_dict, train_edge_dict, atom_type_dict,  nodes_attr = one_hot, title_ = None, k=k_hops, plot=1, show_node_idx = 0)    
        original_label_map, original_orbits = stable_orbits_with_anchor(edges, center_index)
        merged_orbits, merged_label_map = merge_orbits_by_distance_with_paths(
            edges, original_orbits, original_label_map
        )
        z = predicate_features_z(train_x_dict[graph_idx], merged_orbits)
        z_res_1.append(z)
        instance_1.append((graph_idx, center_index))


    # ----- Create dataset -----
    z_concat = np.vstack([np.array(z_res_0), np.array(z_res_1)])
    z_label = np.array([0] * len(z_res_0) + [1] * len(z_res_1))
    instance = instance_0 + instance_1
    instance = np.array(instance)
    # ----- Train decision tree -----
    
    n_samples = len(z_concat)

    max_depth = depth if depth is not None else None



    # --- Attempt 1: Train with your preferred "Hybrid" (stricter) parameters ---

    

    # Define a threshold for what constitutes a "very small" dataset.

    if n_samples < 10:

        min_split = 2

        min_leaf = 1

    else:

        # Your preferred adaptive logic for the general case

        min_split = max(5, n_samples // 20)

        min_leaf = max(3, n_samples // 50)

    

    # Ensure logical consistency

    if min_split < 2 * min_leaf:

        min_split = 2 * min_leaf



    print("Attempt 1: Training with stricter (hybrid) parameters...")

    clf = DecisionTreeClassifier(

        max_depth=max_depth, 

        min_samples_split=min_split,

        min_samples_leaf=min_leaf,

        ccp_alpha=0.001,

        class_weight='balanced',  # ADD THIS LINE

        random_state=42

    )
    clf.fit(z_concat, z_label)
    if clf.tree_.node_count == 1 and n_samples > 1:

        print("Attempt 1 resulted in a stump. Falling back to lenient parameters...")

        

        # --- Attempt 2: Fallback to the "Simple" (most lenient) parameters ---

        clf = DecisionTreeClassifier(

            max_depth=max_depth, 

            random_state=42

        )

        clf.fit(z_concat, z_label)

    else:

        print("Attempt 1 was successful. Using the hybrid-parameter tree.")

    # get the ground rules for the class 1
    used_feature_indices = set(clf.tree_.feature[clf.tree_.feature != _tree.TREE_UNDEFINED])
    feature_names = [f'f_{i}' for i in range(z_concat.shape[1])]
    # used_features = [feature_names[i] for i in sorted(used_feature_indices)]

    # Step 4: Extract rules and sample indices per leaf
    def get_leaf_rules_and_samples(tree, feature_names, adds = adds):
        tree_ = tree.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else None for i in tree_.feature]
        leaf_rules_samples_0 = {}
        leaf_rules_samples_1 = {}

        def recurse(node, conditions, samples):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                feature_idx = tree_.feature[node]
                threshold = tree_.threshold[node]
                left_samples = samples[z_concat[ samples, feature_idx] <= threshold]
                right_samples = samples[z_concat[ samples, feature_idx] > threshold]

                name = feature_name[node]
                # Fixed: Use actual threshold comparisons instead of negation symbols
                recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold}"], left_samples)
                recurse(tree_.children_right[node], conditions + [f"{name} > {threshold}"], right_samples)
            else:
                # Find features with same value across all samples in this leaf
                additional_conditions = []
                if len(samples) > 1:  # Only check if there are multiple samples
                    leaf_data = z_concat[samples]
                    for feature_idx in range(z_concat.shape[1]):
                        feature_values = leaf_data[:, feature_idx]
                        # Check if all values are the same for this feature
                        if len(np.unique(feature_values)) == 1:
                            # Skip if this feature is already used in the path conditions
                            feature_already_used = any(f"f_{feature_idx}" in cond for cond in conditions)
                            if not feature_already_used:
                                unique_value = feature_values[0]
                                # Only report features with values greater than 0
                                if unique_value > 0:
                                    additional_conditions.append(f"f_{feature_idx} = {unique_value}")
                
                # Combine path conditions with additional constant features
                if adds:
                    all_conditions = conditions + additional_conditions
                else:
                    all_conditions = conditions
                rule = f'({" and ".join(all_conditions)})'
                class_value = tree_.value[node].argmax()
                if class_value == 0:
                    leaf_rules_samples_0[rule] = instance[samples.tolist()]
                else:
                    leaf_rules_samples_1[rule] = instance[samples.tolist()]

        recurse(0, [], np.arange(z_concat.shape[0]))

        return leaf_rules_samples_0, leaf_rules_samples_1

    leaf_rules_samples_0, leaf_rules_samples_1 = get_leaf_rules_and_samples(clf, feature_names)
     

    leaf_rules_samples_1 = dict(sorted(leaf_rules_samples_1.items(), key=lambda item: len(item[1]), reverse=True))

    if False:
        # Report accuracy
        acc = clf.score(z_concat, z_label)
        print(f"Decision Tree accuracy (train): {acc:.4f}")

        # Plot tree
        plt.figure(figsize=(12, 8))
        plot_tree(clf, filled=True, feature_names=[f"f{i}" for i in range(z_concat.shape[1])])
        #plt.show()


    if p_idx_position == 0:
        leaf_rules_samples = leaf_rules_samples_0
    if p_idx_position == 1:
        leaf_rules_samples = leaf_rules_samples_1


    return clf, z_concat, z_label, leaf_rules_samples
     




import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_graph_with_orbits(p_idx, all_refined_results, edge_list, eq_sets, show_node=1, save_dir="./plots"):
    """
    Visualize a graph with orbit labels for equivalent nodes and edges.
    
    Parameters:
    p_idx: pattern index to look up in all_refined_results
    all_refined_results: dict containing pattern rules
    edge_list: list of tuples representing edges
    eq_sets: list of frozensets containing equivalent elements (nodes or edges)
    show_node: int, 1 to show node labels, 0 to hide them
    """
    
    # Create the graph
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Calculate required height for rule text
    rule_text = ""
    rule_height = 0.15  # default minimum height for rule area
    
    if all_refined_results:
        all_pattern_rules = []
        
        # Process all patterns in all_refined_results
        for pattern_idx in sorted(all_refined_results.keys()):
            pattern_rules = all_refined_results[pattern_idx]
            orbit_conditions = []
            
            # Sort by orbit number for consistent ordering
            for orbit_num in sorted(pattern_rules.keys()):
                condition = pattern_rules[orbit_num]
                orbit_conditions.append(f"orbit {orbit_num}: '{condition}'")
            
            pattern_text = f"Pattern {pattern_idx}: {' and '.join(orbit_conditions)}"
            all_pattern_rules.append(pattern_text)
        
        rule_text = "\n".join(all_pattern_rules)
        
        # Estimate required height based on text length and number of patterns
        num_patterns = len(all_pattern_rules)
        max_line_length = max(len(line) for line in all_pattern_rules) if all_pattern_rules else 0
        estimated_lines = sum(max(1, len(line) // 90) for line in all_pattern_rules)  # 90 chars per line
        
        # Calculate rule area height (minimum 0.15, scale with content)
        rule_height = max(0.15, min(0.4, 0.05 + estimated_lines * 0.025))
    
    # Set up the plot with calculated dimensions
    fig = plt.figure(figsize=(12, 8))
    
    # Define regions clearly:
    # - Top region for graph: from rule_height to 1.0
    # - Bottom region for rules: from 0 to rule_height
    # - Leave small margins
    
    margin = 0.0
    graph_bottom = rule_height + margin
    graph_top = 0.9
    graph_height = graph_top - graph_bottom
    
    # Create main graph axes in the upper region
    ax = fig.add_axes([0.02, graph_bottom, 0.98, graph_height])  # [left, bottom, width, height]
    
    # Try different layouts for better edge separation
    num_nodes = len(G.nodes())
    
    if num_nodes <= 10:
        # For small graphs, use circular layout with shorter edges
        pos = nx.circular_layout(G, scale=0.8)
    elif num_nodes <= 20:
        # For medium graphs, use shell layout with shorter edges
        pos = nx.shell_layout(G, scale=0.8)
    else:
        # For larger graphs, use spring layout with shorter edges
        pos = nx.spring_layout(G, k=1.5/np.sqrt(num_nodes), iterations=100, seed=42)
    
    # If edges are still too crowded, try kamada_kawai for better spacing
    if len(edge_list) > num_nodes * 1.5:  # Dense graph
        try:
            pos = nx.kamada_kawai_layout(G, scale=0.8)
        except:
            # Fallback to spring layout with shorter spacing
            pos = nx.spring_layout(G, k=2/np.sqrt(num_nodes), iterations=150, seed=42)
    
    # Colors for different orbits
    orbit_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                   '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    # Draw all edges first (in light gray) with better spacing and stronger lines
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.6, ax=ax)
    
    # Draw all nodes first (in light color) with bigger size
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=600, alpha=0.7, ax=ax)
    
    # Process equivalence sets
    orbit_labels = []
    
    for orbit_idx, eq_set in enumerate(eq_sets):
        color = orbit_colors[orbit_idx % len(orbit_colors)]
        orbit_label = f"Orbit {orbit_idx}"  # Starting from 0
        
        # Convert frozenset to list for easier handling
        elements = list(eq_set)
        
        # Check if this orbit contains nodes (single integers) or edges (tuples)
        if len(elements) > 0:
            first_element = elements[0]
            
            if isinstance(first_element, tuple):
                # This is an edge orbit
                # Highlight the edges in this orbit with curved edges and stronger lines
                nx.draw_networkx_edges(G, pos, edgelist=elements, 
                                     edge_color=color, width=4, alpha=0.9,
                                      ax=ax)  # Curved edges
                
                # Add labels near the edges with better positioning
                for i, edge in enumerate(elements):
                    # Calculate midpoint of edge for label placement
                    x1, y1 = pos[edge[0]]
                    x2, y2 = pos[edge[1]]
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # Calculate perpendicular offset for better label placement
                    dx, dy = x2 - x1, y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        perp_x, perp_y = -dy/length, dx/length
                    else:
                        perp_x, perp_y = 0, 0
                    
                    # Smaller offset to bring labels closer to edges
                    offset_distance = 0.08 + (i % 3) * 0.03
                    final_x = mid_x + perp_x * offset_distance
                    final_y = mid_y + perp_y * offset_distance
                    
                    # Create a bigger text box for the orbit label
                    bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.95, edgecolor='white', linewidth=2)
                    ax.text(final_x, final_y, orbit_label, 
                           fontsize=10, fontweight='bold', ha='center', va='center',
                           bbox=bbox, color='white')
                
                orbit_labels.append((orbit_label, color, 'Edge'))
                
            else:
                # This is a node orbit
                # Highlight the nodes in this orbit with bigger size
                nx.draw_networkx_nodes(G, pos, nodelist=elements, 
                                     node_color=color, node_size=800, alpha=0.9,
                                     edgecolors='white', linewidths=2, ax=ax)
                
                # Add labels for the nodes
                for node in elements:
                    x, y = pos[node]
                    # Create a bigger text box for the orbit label, closer to node
                    bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.95, edgecolor='white', linewidth=2)
                    ax.text(x, y + 0.12, orbit_label, fontsize=10, fontweight='bold', 
                           ha='center', va='center', bbox=bbox, color='white')
                
                orbit_labels.append((orbit_label, color, 'Node'))
    
    # Add node labels (node IDs) only if show_node is 1
    if show_node:
        node_labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold', ax=ax)

    # Create legend for orbits using the original approach
    if orbit_labels:
        legend_elements = []
        for label, color, orbit_type in orbit_labels:
            if orbit_type == 'Node':
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=f"{label} ({orbit_type})"))
            else:
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, 
                                                label=f"{label} ({orbit_type})"))
        
        # Position legend on the best available spot within the graph area
        ax.legend(handles=legend_elements, loc='best',
                 frameon=True, fancybox=True, shadow=True)
    
    # Add rule text box in the bottom region (completely separate)
    if rule_text:
        # Create a separate axes for the rule text at bottom
        rule_ax = fig.add_axes([0.02, 0.02, 0.96, rule_height - 0.04])  # Full width, dedicated height
        rule_ax.axis('off')  # Hide axes
        
        # Split long text into multiple lines for better display
        max_chars_per_line = 120  # Increased since we have full width
        lines = rule_text.split('\n')  # Already split by pattern
        formatted_lines = []
        
        for line in lines:
            if len(line) > max_chars_per_line:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) > max_chars_per_line:
                        if current_line:
                            formatted_lines.append(current_line.strip())
                            current_line = word + " "
                        else:
                            formatted_lines.append(word)
                            current_line = ""
                    else:
                        current_line += word + " "
                if current_line:
                    formatted_lines.append(current_line.strip())
            else:
                formatted_lines.append(line)
        
        final_rule_text = "\n".join(formatted_lines)
        
        # Create text box with better styling
        rule_bbox = dict(boxstyle="round,pad=0.5", facecolor='lightyellow', 
                        alpha=0.95, edgecolor='darkblue', linewidth=2)
        
        # Add the text to the rule axes, centered vertically in the available space
        rule_ax.text(0.02, 0.98, final_rule_text, fontsize=11, fontweight='bold',
                    ha='left', va='top', bbox=rule_bbox,
                    transform=rule_ax.transAxes, color='darkblue')
    
    # Set title for the graph
    ax.set_title(f"Grounding rule explanation for p_{p_idx}", fontsize=14, fontweight='bold')
    ax.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"orbits_explanation_p{p_idx}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)



def check_use_edge(eq_sets):
    """Check if there exists any tuple in eq_sets to determine use_edge"""
    for eq_set in eq_sets:
        for item in eq_set:
            if isinstance(item, tuple):
                return True
    return False

def parse_conditions(condition_str):
    """Parse condition string and extract feature indices, operators, and thresholds"""
    # Remove parentheses and split by 'and'
    condition_str = condition_str.strip('()')
    conditions = condition_str.split(' and ')
    
    parsed_conditions = []
    for cond in conditions:
        # Extract feature index, operator, and threshold using regex
        # Handle <=, >=, <, >, =
        match = re.match(r'f_(\d+)\s*(<=|>=|<|>|=)\s*([\d.]+)', cond.strip())
        if match:
            feature_idx = int(match.group(1))
            operator = match.group(2)
            threshold = float(match.group(3))
            parsed_conditions.append((feature_idx, operator, threshold))
    
    return parsed_conditions







def process_feature_conditions(eq_sets, atom_type_dict, condition):
    """Process feature conditions based on equivalent sets and atom types"""
    
    use_edge = check_use_edge(eq_sets)
    
    if use_edge:
        len_ = len(atom_type_dict) + len(atom_type_dict) * len(atom_type_dict)
        node_len = len(atom_type_dict)
    else:
        len_ = len(atom_type_dict)
    
    # Parse the condition string

    parsed_conditions = parse_conditions(condition)
    print(f"Parsed conditions: {parsed_conditions}")
    print("Condition:", condition)
    results = []
    
    for feature_idx, operator, threshold in parsed_conditions:
        
        if use_edge:
            orbit_idx = feature_idx // len_
            mid_idx = feature_idx % len_
            edge_0 = mid_idx // len(atom_type_dict)
            
            if edge_0 == 0:
                # Node feature case
                atom_type_key = mid_idx % len(atom_type_dict)
                atom_type = atom_type_dict[atom_type_key]
                
                # Convert threshold and operator to discrete condition
                if operator == '<=':
                    discrete_value = 0 if threshold <= 0.5 else int(threshold)
                    condition_op = '=' if discrete_value == 0 else '<='
                elif operator == '>':
                    # For > threshold, we want >= (threshold + 0.5) rounded up
                    discrete_value = int(threshold) + 1 if threshold != int(threshold) else int(threshold) + 1
                    if threshold == int(threshold):  # exact integer
                        discrete_value = int(threshold) + 1
                        condition_op = '>='
                    else:
                        discrete_value = int(threshold + 1)
                        condition_op = '>='
                elif operator == '>=':
                    discrete_value = int(threshold) if threshold == int(threshold) else int(threshold) + 1
                    condition_op = '>='
                elif operator == '<':
                    discrete_value = int(threshold) - 1 if threshold == int(threshold) else int(threshold)
                    condition_op = '<='
                else:  # operator == '='
                    discrete_value = int(threshold)
                    condition_op = '='
                
                result = {
                    'orbit_idx': orbit_idx,
                    'type': 'node',
                    'atom_type': atom_type,
                    'discrete_value': discrete_value,
                    'operator': condition_op
                }
            else:
                # Edge feature case
                edge_0 = edge_0 - 1
                edge_1 = mid_idx % len(atom_type_dict)
                atom_type_0 = atom_type_dict.get(edge_0, edge_0)
                atom_type_1 = atom_type_dict.get(edge_1, edge_1)
                
                # Similar logic for edges
                if operator == '<=':
                    discrete_value = 0 if threshold <= 0.5 else int(threshold)
                    condition_op = '=' if discrete_value == 0 else '<='
                elif operator == '>':
                    discrete_value = int(threshold) + 1 if threshold != int(threshold) else int(threshold) + 1
                    condition_op = '>='
                elif operator == '>=':
                    discrete_value = int(threshold) if threshold == int(threshold) else int(threshold) + 1
                    condition_op = '>='
                elif operator == '<':
                    discrete_value = int(threshold) - 1 if threshold == int(threshold) else int(threshold)
                    condition_op = '<='
                else:  # operator == '='
                    discrete_value = int(threshold)
                    condition_op = '='
                
                result = {
                    'orbit_idx': orbit_idx,
                    'type': 'edge',
                    'edge_pair': (atom_type_0, atom_type_1),
                    'discrete_value': discrete_value,
                    'operator': condition_op
                }
        
        else:
            # Node-only case
            orbit_idx = feature_idx // len_
            node_idx = feature_idx % len_
            
            atom_type = atom_type_dict[node_idx] if node_idx < len(atom_type_dict) else str(node_idx)
            
            # Handle different operators
            if operator == '<=':
                discrete_value = 0 if threshold <= 0.5 else int(threshold)
                condition_op = '=' if discrete_value == 0 else '<='
            elif operator == '>':
                # For f_70 > 2.5, we want #C >= 3
                if threshold == int(threshold):  # exact integer like 2.0
                    discrete_value = int(threshold) + 1
                else:  # decimal like 2.5
                    discrete_value = int(threshold + 0.5)
                condition_op = '>='
            elif operator == '>=':
                discrete_value = int(threshold) if threshold == int(threshold) else int(threshold) + 1
                condition_op = '>='
            elif operator == '<':
                discrete_value = int(threshold) - 1 if threshold == int(threshold) else int(threshold)
                condition_op = '<='
            else:  # operator == '='
                discrete_value = int(threshold)
                condition_op = '='
            
            result = {
                'orbit_idx': orbit_idx,
                'type': 'node',
                'node_idx': node_idx,
                'atom_type': atom_type,
                'discrete_value': discrete_value,
                'operator': condition_op
            }
        
        results.append(result)
    
    return results

def merge_results_by_orbit(results):
    """Merge results by orbit_idx and format as requested"""
    orbit_groups = {}
    
    for result in results:
        orbit_idx = result['orbit_idx']
        if orbit_idx not in orbit_groups:
            orbit_groups[orbit_idx] = []
        orbit_groups[orbit_idx].append(result)
    
    merged_results = {}
    for orbit_idx, group in orbit_groups.items():
        conditions = []
        for result in group:
            if result['type'] == 'edge':
                edge_pair = result['edge_pair']
                discrete_val = int(result['discrete_value'])
                op = result['operator']
                
                if op == '=' and discrete_val == 0:
                    condition = f"#({edge_pair[0]}, {edge_pair[1]}) = 0"
                elif op == '<=':
                    condition = f"#({edge_pair[0]}, {edge_pair[1]}) <= {discrete_val}"
                elif op == '>=':
                    condition = f"#({edge_pair[0]}, {edge_pair[1]}) >= {discrete_val}"
                else:
                    condition = f"#({edge_pair[0]}, {edge_pair[1]}) {op} {discrete_val}"
                
                conditions.append(condition)
            elif result['type'] == 'node':
                atom_type = result['atom_type']
                discrete_val = int(result['discrete_value'])
                op = result['operator']
                
                if op == '=' and discrete_val == 0:
                    condition = f"#{atom_type} = 0"
                elif op == '<=':
                    condition = f"#{atom_type} <= {discrete_val}"
                elif op == '>=':
                    condition = f"#{atom_type} >= {discrete_val}"
                else:
                    condition = f"#{atom_type} {op} {discrete_val}"
                
                conditions.append(condition)
        
        merged_results[orbit_idx] = ' and '.join(conditions)
    
    return merged_results




def refine_conditions_with_orbit_sizes(merged_results, eq_sets):
    """
    Refine conditions based on the actual size of orbits from eq_sets.
    If a condition like '#O >= k' is applied to an orbit with size n where k >= n,
    then refine it to '#O = n' (the maximum possible value).
    
    Args:
        merged_results: Dictionary mapping orbit_idx to condition strings
        eq_sets: List of frozensets representing the equivalent sets for each orbit
    
    Returns:
        Dictionary with refined condition strings
    """
    refined_results = {}
    
    for orbit_idx, condition_str in merged_results.items():
        # Get the size of this orbit
        if orbit_idx < len(eq_sets):
            orbit_size = len(eq_sets[orbit_idx])
        else:
            # If orbit_idx is out of range, keep original condition
            refined_results[orbit_idx] = condition_str
            continue
        
        # Parse and refine each condition in the string
        refined_conditions = []
        conditions = condition_str.split(' and ')
        
        for cond in conditions:
            cond = cond.strip()
            refined_cond = refine_single_condition(cond, orbit_size)
            refined_conditions.append(refined_cond)
        
        refined_results[orbit_idx] = ' and '.join(refined_conditions)
    
    return refined_results

def refine_single_condition(condition, orbit_size):
    """
    Refine a single condition based on orbit size.
    
    Examples:
    - '#O >= 1' with orbit_size=1 becomes '#O = 1'
    - '#O >= 2' with orbit_size=2 becomes '#O = 2'
    - '#O >= 1' with orbit_size=3 stays '#O >= 1'
    - '#(C, O) >= 2' with orbit_size=2 becomes '#(C, O) = 2'
    """
    import re
    
    # Pattern to match conditions like '#O >= 1', '#(C, O) >= 2', etc.
    # This handles both node and edge patterns
    pattern = r'(#\([^)]+\)|#\w+)\s*(>=|<=|=|>|<)\s*(\d+)'
    match = re.match(pattern, condition)
    
    if not match:
        return condition  # Return original if pattern doesn't match
    
    element_part = match.group(1)  # '#O' or '#(C, O)'
    operator = match.group(2)      # '>=', '<=', '=', '>', '<'
    value = int(match.group(3))    # The numeric value
    
    # Apply refinement logic
    if operator == '>=' and value > 0:
        # If the condition requires >= k and orbit size is exactly k or less,
        # then the maximum possible value is orbit_size
        if value >= orbit_size:
            return f"{element_part} = {orbit_size}"
        else:
            return condition  # Keep original condition
    
    elif operator == '>' and value >= 0:
        # For '>' conditions, if value+1 >= orbit_size, refine to max
        required_min = value + 1
        if required_min >= orbit_size:
            return f"{element_part} = {orbit_size}"
        else:
            return condition
    
    elif operator == '<=' and value < orbit_size:
        # For '<=' conditions, no refinement needed based on orbit size
        return condition
    
    elif operator == '<' and value <= orbit_size:
        # For '<' conditions, no refinement needed based on orbit size  
        return condition
    
    elif operator == '=':
        # For '=' conditions, check if the value is possible
        if value > orbit_size:
            # This condition is impossible, but we'll keep it as is
            # (might want to flag as inconsistent in a real application)
            return condition
        else:
            return condition
    
    return condition










def parse_ground_rules_pattern_refined(p_idx, ground_rules,  train_x_dict, train_edge_dict, atom_type_dict, one_hot = 0, k_hops = 0, stat=0, save_dir="./plots"):
    """
    Enhanced version of parse_ground_rules_pattern that includes orbit size refinement.
    """
    pattern_counter = 1
    all_merged_results = {}
    all_refined_results = {}

    for k, v in ground_rules.items():
        graph_idx, center_index = v[0]
        edges, nodes_attributes, center_index, wl_hash = plot_k_hop_subgraph(graph_idx, int(center_index), train_x_dict, train_edge_dict, atom_type_dict, nodes_attr = one_hot, title_ = None, k=k_hops, plot=1, show_node_idx = 0)
        original_label_map, original_orbits = stable_orbits_with_anchor(edges, center_index)
        merged_orbits, merged_label_map = merge_orbits_by_distance_with_paths(edges, original_orbits, original_label_map)
        results = process_feature_conditions(merged_orbits, atom_type_dict, k)
        merged_results = merge_results_by_orbit(results)
        
        # Apply orbit size refinement
        refined_results = refine_conditions_with_orbit_sizes(merged_results, merged_orbits)
        
        orbit_conditions = []
        for orbit_idx, conditions in sorted(refined_results.items()):
            orbit_conditions.append(f"orbit {orbit_idx}: '{conditions}'")
        
        num_instances = len(v)
        
        # if pattern_counter == 1:
        #     visualize_graph_with_orbits(p_idx, edges, merged_orbits, show_node=0)
        
        # if stat:
        #     print(f"Pattern {pattern_counter} (# of instances: {num_instances}): {' and '.join(orbit_conditions)}")
        # else:
        #     print(f"Pattern {pattern_counter}: {' and '.join(orbit_conditions)}")
        
        # Store both original and refined results
        all_merged_results[pattern_counter] = merged_results
        all_refined_results[pattern_counter] = refined_results
        pattern_counter += 1

    # visualize_graph_with_orbits(p_idx, all_refined_results, edges, merged_orbits, show_node=0, save_dir=save_dir)

    return all_refined_results

def explain_predicate_with_rules(
    p_idx,
    used_iso_predicate_node,
    train_x_dict,
    train_edge_dict,
    atom_type_dict,
    idx_predicates_mapping,
    iso_predicates_inference,   # shared dict, updated in place
    one_hot,
    k_hops=2,
    depth=5,
    adds=0,
    verbose=1,
    top_k=5,
    save_dir="./plot",
    plot=True   # <--- switch for coverage plots
):
    """
    Extract rules and explanations for a single predicate, 
    update iso_predicates_inference in place, and always return subgraph hash mapping.
    """
    # === Step 1: get rules ===
    clf, z_concat, z_label, ground_rules = get_the_rule_iso_z(
        p_idx, used_iso_predicate_node,
        train_x_dict, atom_type_dict, train_edge_dict,
        one_hot=one_hot, k_hops=k_hops, depth=None,
        adds=adds, verbose=verbose
    )
    if not ground_rules:
        if verbose:
            print(f"⚠️ No ground rules found for predicate {p_idx}, skipping.")
        return {}  # always return a dict

    hash_, res = idx_predicates_mapping[p_idx]
    iso_predicates_inference[hash_] = (clf, res, p_idx)   # update shared dict

    # === Step 2: refine rules ===
    parse_ground_rules_pattern_refined(
        p_idx, ground_rules,
        train_x_dict, train_edge_dict, atom_type_dict,
        one_hot=one_hot, k_hops=k_hops, stat=0, save_dir=save_dir
    )

    # === Step 3: collect subgraph hashes (always) ===
    hash_dict = {}
    subgraph_hashes = None
    for graph_idx, center_index in used_iso_predicate_node[p_idx][p_idx]:
        edges, nodes_attributes, center_index, wl_hash = plot_k_hop_subgraph(
            graph_idx, center_index,
            train_x_dict, train_edge_dict, atom_type_dict,
            title_=None, k=k_hops, plot=1, show_node_idx=0, nodes_attr=one_hot, grounding=False
        )
        if verbose:
            print(f"Graph {graph_idx}, center {center_index} → WL hash {wl_hash}")

        hash_dict.setdefault(wl_hash, []).append((graph_idx, center_index))
        hash_dict_sorted = dict(sorted(hash_dict.items(), key=lambda x: len(x[1]), reverse=True))
        total_nodes = len(used_iso_predicate_node[p_idx][p_idx])
        k_ = min(top_k, len(hash_dict_sorted))
        top_k_keys = list(hash_dict_sorted.keys())[:k_]

    # === Step 4: coverage plots (only if enabled) ===
    if plot:


        fig, axes = plt.subplots(1, k_, figsize=(4 * k_, 4))
        if k_ == 1:
            axes = [axes]

        for ax, key in zip(axes, top_k_keys):
            node_list = hash_dict_sorted[key]
            if node_list:
                # coverage_ratio = len(node_list) / total_nodes
                # title_str = f"Coverage: {coverage_ratio:.2%}"
                title_str = None
                graph_idx, center_index = node_list[0]
                plot_k_hop_subgraph(
                    graph_idx, center_index,
                    train_x_dict, train_edge_dict, atom_type_dict,
                    k=2, show_node_idx=0,
                    title_=title_str, ax=ax, title_position='bottom',
                    nodes_attr=one_hot
                )

        fig.suptitle(f"Top {k_} explanation graphs for p_{p_idx}", fontsize=14, y=1.01)
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"coverage_p{p_idx}.png")
        fig.savefig(filename, bbox_inches="tight", dpi=150)
        plt.close(fig)

        if verbose:
            print(f"✅ Saved coverage plot for p_{p_idx} at {filename}")

    # ✅ Always return one subgraph hashes
    return top_k_keys[0]

def plot_alone_predicate_explanation(p, graph_idx, center_index,
                                     x_dict, edge_dict,
                                     dataset, seed, arch, k=2):
    """
    Save the k-hop subgraph around a center node for a given predicate
    into ./plot/<dataset>/<seed>/

    Args:
        p (int/str): Predicate ID
        graph_idx (int): Graph index in dataset
        center_index (int): Center node index
        x_dict (dict): Node features per graph
        edge_dict (dict): Edge index per graph
        dataset (str): Dataset name
        seed (int): Random seed
        k (int): Number of hops
    """
    # Directory: ./plot/dataset/seed/
    save_dir = os.path.join("plot", dataset, str(seed), arch, "alone")
    os.makedirs(save_dir, exist_ok=True)

    edge_tensor = edge_dict[graph_idx]
    x_tensor = x_dict[graph_idx]

    # Extract k-hop subgraph
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        center_index, k, edge_tensor, relabel_nodes=True
    )

    # Subgraph Data
    sub_x = x_tensor[subset]
    sub_data = Data(x=sub_x, edge_index=sub_edge_index)

    # Convert to NetworkX for plotting
    G = to_networkx(sub_data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    # Draw
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    plt.title(f"Explanation graph for predicate p_{p}", fontsize=14, y=1.01)
    plt.axis("off")

    # Save to file
    filename = os.path.join(
        save_dir, f"predicate_{p}_graph{graph_idx}_center{center_index}.png"
    )
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()

    return filename


def save_all_alone_predicates(used_alone_predicates, x_dict, edge_dict,
                              dataset, seed, arch, k=2):
    """
    Save explanation plots for all used-alone predicates into ./plot/<dataset>/<seed>/

    Args:
        used_alone_predicates (dict): {wl_hash: (predicate, [(graph_idx, center_index), ...])}
        x_dict (dict): Node features
        edge_dict (dict): Edge indices
        dataset (str): dataset name
        seed (int): seed value
        k (int): k-hop size
    """
    saved_files = []
    for wl, v in used_alone_predicates.items():
        p, node_list = v
        graph_idx, center_index = node_list[0]

        # print(f"Saving {dataset}_{seed} → predicate {p} (graph {graph_idx}, center {center_index})")
        fname = plot_alone_predicate_explanation(
            p, graph_idx, center_index,
            x_dict, edge_dict,
            dataset, seed, arch, k=k
        )
        

def plot_alone_predicate_explanation(p, graph_idx, center_index, train_x_dict, train_edge_dict, k, ax=None):
    # Extract k-hop subgraph
    edge_tensor = train_edge_dict[graph_idx]
    x_tensor = train_x_dict[graph_idx]

    # Extract k-hop subgraph
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        center_index, k, edge_tensor, relabel_nodes=True
    )

    # Create PyG Data object for subgraph
    sub_x = x_tensor[subset]
    sub_data = Data(x=sub_x, edge_index=sub_edge_index)

    # Convert to NetworkX graph
    G = to_networkx(sub_data, to_undirected=True)

    pos = nx.spring_layout(G, seed=42)
    target_ax = ax if ax is not None else plt.gca()
    target_ax.clear()

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=target_ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=target_ax)

    target_ax.set_title(f"The explanation graphs for p_{p}", fontsize=14, y=1.01)
    target_ax.axis('off')
    
def grounded_graph_predictions(
    test_dataset,
    test_x_dict,
    test_edge_dict,
    atom_type_dict,
    iso_predicates_inference,
    used_alone_predicates,
    predicates,
    clf_graph,
    k_hops=2,
    one_hot=True
):
    """
    Compute grounded predictions for each graph in the dataset using
    WL-hash predicates, orbit features, and a global classifier.

    Args:
        test_dataset: PyG Dataset of graphs
        test_x_dict (dict): node features per graph
        test_edge_dict (dict): edge index per graph
        atom_type_dict (dict): atom type mapping
        iso_predicates_inference (dict): {wl_hash: (clf_pred, act, predicate)}
        used_alone_predicates (dict): {wl_hash: (predicate, ...)}
        predicates (list): full predicate list
        clf_graph: global graph-level classifier
        k_hops (int): neighborhood size for subgraph extraction
        one_hot (bool): whether node attributes are one-hot

    Returns:
        np.ndarray: predicted class labels for all graphs
    """
    data_grounded_pred = []

    for graph_idx in range(len(test_dataset)):
        edges_graph = test_dataset[graph_idx].edge_index
        matched_predicate_graph = []

        # Collect all unique nodes in the graph
        unique_nodes = torch.unique(edges_graph).tolist()

        for center in unique_nodes:
            # k-hop subgraph + WL-hash
            edges, nodes_attributes, center_index, wl_hash = plot_k_hop_subgraph(
                graph_idx, center, test_x_dict, test_edge_dict,
                atom_type_dict, nodes_attr=one_hot,
                title_=None, k=k_hops, plot=0, show_node_idx=1
            )

            # Case 1: predicate with classifier
            if wl_hash in set(iso_predicates_inference.keys()):
                clf_pred, act, predicate = iso_predicates_inference[wl_hash]

                # Compute orbits & features
                original_label_map, original_orbits = stable_orbits_with_anchor(edges, center)
                merged_orbits, merged_label_map = merge_orbits_by_distance_with_paths(
                    edges, original_orbits, original_label_map
                )
                z = predicate_features_z(test_x_dict[graph_idx], merged_orbits)

                pred_res = clf_pred.predict(z.reshape(1, -1))[0]
                if pred_res == act:
                    matched_predicate_graph.append(predicate)

            # Case 2: standalone predicate
            if wl_hash in set(used_alone_predicates.keys()):
                matched_predicate_graph.append(used_alone_predicates[wl_hash][0])

        # Deduplicate predicates
        idx_true = list(set(matched_predicate_graph))

        # Build predicate indicator vector
        graph_infernece_array = np.zeros((len(predicates)))
        graph_infernece_array[idx_true] = 1

        # Graph-level classifier prediction
        pred_class = int(clf_graph.predict(graph_infernece_array.reshape(1, -1))[0])
        data_grounded_pred.append(pred_class)

    return np.array(data_grounded_pred)
