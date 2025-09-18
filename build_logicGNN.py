from collections import defaultdict
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
def process_graph_predicates(graph_idx, train_x_dict, train_edge_dict, train_activations_dict, val_idx, threshold, use_embed = 1, k_hops=2):
    node_act = (train_activations_dict['conv2'][graph_idx][:, val_idx] > threshold).int().numpy()
    edges = train_edge_dict[graph_idx]
    unique_nodes = torch.unique(edges)
    
    # Use defaultdict for faster insertion
    predicate_node_dict_graph = defaultdict(list)
    
    for node in unique_nodes:
        node_int = node.item()
        # Convert single node to tensor
        node_tensor = torch.tensor([node_int], dtype=torch.long)
        
        # Extract k-hop subgraph
        node_idx, edge_index_subgraph, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_tensor,
            num_hops=k_hops,
            edge_index=edges,
            relabel_nodes=True
        )
        
        # Convert subgraph to NetworkX graph for WL hash
        subgraph_data = Data(edge_index=edge_index_subgraph, num_nodes=len(node_idx))
        nx_subgraph = to_networkx(subgraph_data, to_undirected=True)
        
        # Compute Weisfeiler-Lehman hash
        wl_hash = nx.weisfeiler_lehman_graph_hash(nx_subgraph)
        
        # Directly add to predicate_node_dict_graph (skip intermediate dict)
        if use_embed:
            predicate = (wl_hash, node_act[node_int])
        else:
            predicate = (wl_hash, 1) # only use graph pattern

        predicate_node_dict_graph[predicate].append((graph_idx, node_int))
    
    # Convert defaultdict back to regular dict
    return dict(predicate_node_dict_graph)
def get_predicates_bin_one_pass(index_0_correct,index_1_correct,train_x_dict, train_edge_dict, train_activations_dict, val_idx, threshold, use_embed = 0 , k_hops = 2):

    predicate_graph_class_0 = {}  # key is graph_idx, value is set of predicates for that graph
    predicate_node_class_0 = defaultdict(list)  # key is predicate, value is list of (graph_idx, node) pairs

    predicate_graph_class_1 = {}
    predicate_node_class_1 = defaultdict(list)

    # Process class 0 graphs
    for graph_idx in index_0_correct:
        graph_idx = int(graph_idx)
        predicate_node_dict_graph = process_graph_predicates(graph_idx, train_x_dict, train_edge_dict, train_activations_dict, val_idx, threshold, use_embed = use_embed , k_hops = k_hops)
        
        # Store predicates for this graph
        predicate_graph_class_0[graph_idx] = set(predicate_node_dict_graph.keys())
        
        # Merge predicate_node_dict_graph into predicate_node_class_0
        for predicate, node_list in predicate_node_dict_graph.items():
            predicate_node_class_0[predicate].extend(node_list)

    # Process class 1 graphs
    for graph_idx in index_1_correct:
        graph_idx = int(graph_idx)
        predicate_node_dict_graph = process_graph_predicates(graph_idx, train_x_dict, train_edge_dict, train_activations_dict, val_idx, threshold, use_embed = use_embed , k_hops = k_hops)
        
        # Store predicates for this graph
        predicate_graph_class_1[graph_idx] = set(predicate_node_dict_graph.keys())
        
        # Merge predicate_node_dict_graph into predicate_node_class_1
        for predicate, node_list in predicate_node_dict_graph.items():
            predicate_node_class_1[predicate].extend(node_list)

    # Convert defaultdicts back to regular dicts if needed
    predicate_node_class_0 = dict(predicate_node_class_0)
    predicate_node_class_1 = dict(predicate_node_class_1)

    predicate_node = defaultdict(list)

    # Merge both dictionaries
    for predicate_dict in [predicate_node_class_0, predicate_node_class_1]:
        for predicate, node_list in predicate_dict.items():
            predicate_node[predicate].extend(node_list)

    predicate_node = dict(predicate_node)
    predicates = list(predicate_node.keys())  # Use list to maintain order for indexing

    # Create mapping from predicate to index
    predicate_to_idx = {predicate: idx for idx, predicate in enumerate(predicates)}

    # Initialize matrices with correct shape
    rules_matrix_0 = np.zeros((len(predicates), len(index_0_correct)))
    rules_matrix_1 = np.zeros((len(predicates), len(index_1_correct)))

    # Create mapping from graph_idx to column index for class 0
    graph_to_col_0 = {int(graph_idx): col_idx for col_idx, graph_idx in enumerate(index_0_correct)}

    # Fill rules_matrix_0
    for graph_idx, predicate_set in predicate_graph_class_0.items():
        col_idx = graph_to_col_0[graph_idx]
        for predicate in predicate_set:
            if predicate in predicate_to_idx:  # Check if predicate exists in merged set
                row_idx = predicate_to_idx[predicate]
                rules_matrix_0[row_idx, col_idx] = 1

    # Create mapping from graph_idx to column index for class 1
    graph_to_col_1 = {int(graph_idx): col_idx for col_idx, graph_idx in enumerate(index_1_correct)}

    # Fill rules_matrix_1
    for graph_idx, predicate_set in predicate_graph_class_1.items():
        col_idx = graph_to_col_1[graph_idx]
        for predicate in predicate_set:
            if predicate in predicate_to_idx:  # Check if predicate exists in merged set
                row_idx = predicate_to_idx[predicate]
                rules_matrix_1[row_idx, col_idx] = 1


    return  predicates, predicate_to_idx, predicate_node, predicate_graph_class_0, predicate_graph_class_1, rules_matrix_0, rules_matrix_1
def get_predicate_graph(index_graph, predicates, predicate_to_idx, x_dict, edge_dict, activations_dict, val_idx, threshold, use_embed = 0 , k_hops = 1): #found it !!!!!!!
    predicate_graph = {}
        # Process class 0 graphs
    for graph_idx in index_graph:
        graph_idx = int(graph_idx)
        predicate_node_dict_graph = process_graph_predicates(graph_idx, x_dict, edge_dict, activations_dict, val_idx, threshold, use_embed = use_embed , k_hops = k_hops)
        
        # Store predicates for this graph
        predicate_graph[graph_idx] = set(predicate_node_dict_graph.keys())

    rule_matrix = np.zeros((len(predicates), len(index_graph)))     

    graph_to_col = {int(graph_idx): col_idx for col_idx, graph_idx in enumerate(index_graph)}

    # Fill rules_matrix_0
    for graph_idx, predicate_set in predicate_graph.items():
        col_idx = graph_to_col[graph_idx]
        for predicate in predicate_set:
            if predicate in predicate_to_idx:  # Check if predicate exists in merged set
                row_idx = predicate_to_idx[predicate]
                rule_matrix[row_idx, col_idx] = 1   


    return predicate_graph, rule_matrix
def get_discriminative_rules_with_samples(rules_matrix_0, rules_matrix_1, index_0_correct, index_1_correct, max_depth=3, plot=0, text=0):
    print("Used max depth:", max_depth)

    # Step 1: Combine matrices
    X = np.concatenate((rules_matrix_0, rules_matrix_1), axis=1)  # Shape (num_predicates, num_instances)

    # Step 2: Labels
    y = np.concatenate((np.zeros(rules_matrix_0.shape[1]), np.ones(rules_matrix_1.shape[1])))

    # Combine indices
    mapping_idx = torch.cat((index_0_correct, index_1_correct))
    # Step 3: Train decision tree
    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    clf.fit(X.T, y)

    # sample_weights = compute_sample_weight(class_weight=class_weights_map, y=y)
    #weights = torch.tensor(sample_weights, dtype=torch.float32) 
    # Step 4: Predictions and accuracy
    y_pred = clf.predict(X.T)
    acc = accuracy_score(y, y_pred)
    #weighted_acc = accuracy_score(y, y_pred, sample_weight=weights)
    print(f"Decision Tree Accuracy: {acc:.4f}")
    #print(f"Decision Tree Weighted Accuracy: {weighted_acc:.4f}")

    # Get used predicates
    used_feature_indices = set(clf.tree_.feature[clf.tree_.feature != _tree.TREE_UNDEFINED])
    predicate_names = [f'p_{i}' for i in range(X.shape[0])]
    used_predicates = [predicate_names[i] for i in sorted(used_feature_indices)]


    used_predicates_ = [int(p.split('_')[1]) for p in used_predicates]

    # Step 4: Extract rules and sample indices per leaf
    def get_leaf_rules_and_samples(tree, predicate_names):
        tree_ = tree.tree_
        feature_name = [predicate_names[i] if i != _tree.TREE_UNDEFINED else None for i in tree_.feature]
        leaf_rules_samples_0 = {}
        leaf_rules_samples_1 = {}

        def recurse(node, conditions, samples):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                feature_idx = tree_.feature[node]
                threshold = tree_.threshold[node]
                left_samples = samples[X[feature_idx, samples] <= threshold]
                right_samples = samples[X[feature_idx, samples] > threshold]

                name = feature_name[node]
                recurse(tree_.children_left[node], conditions + [f"¬{name}"], left_samples)
                recurse(tree_.children_right[node], conditions + [f"{name}"], right_samples)
            else:
                rule = f'({" and ".join(conditions)})'
                class_value = tree_.value[node].argmax()
                if class_value == 0:
                    leaf_rules_samples_0[rule] = mapping_idx[samples.tolist()]
                else:
                    leaf_rules_samples_1[rule] = mapping_idx[samples.tolist()]

        recurse(0, [], np.arange(X.shape[1]))

        return leaf_rules_samples_0, leaf_rules_samples_1

    leaf_rules_samples_0, leaf_rules_samples_1 = get_leaf_rules_and_samples(clf, predicate_names)

    if False:
        print("Leaf Rules and Sample Indices for Class 0:")
        for rule, samples in leaf_rules_samples_0.items():
            print(f"Rule: {rule}")
            print(f"Samples: {samples}")
            print()

        print("Leaf Rules and Sample Indices for Class 1:")
        for rule, samples in leaf_rules_samples_1.items():
            print(f"Rule: {rule}")
            print(f"Samples: {samples}")
            print()

    if plot:
        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=predicate_names, class_names=['0', '1'], filled=True, rounded=True)
        plt.title("Decision Tree Classifier")
        plt.show()

    return leaf_rules_samples_0, leaf_rules_samples_1, used_predicates_, clf 