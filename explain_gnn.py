import torch
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
def get_all_activations_graph(t_loader, model, device, optimizer=None):
    """
    Extract activations for every graph in a loader.
    Works for both models with and without conv3.
    """
    x_dict = {}
    edge_dict = {}
    activations_dict = {
        'conv1': {},
        'relu1': {},
        'conv2': {},
        'relu2': {},
        # conv3 will be added dynamically if present
        'global_pool': {},
        'fc': {}
    }

    pred_li = []
    y_li = []
    graph_idx = 0

    for data in t_loader:
        data.x = data.x.float().to(device)
        data.y = data.y.long().squeeze().to(device)
        data = data.to(device)

        if optimizer:  # usually in eval we don't need optimizer.zero_grad()
            optimizer.zero_grad()

        # Forward pass
        out, act = model(data.x, data.edge_index, data.batch)
        _, pred = out.max(dim=1)

        pred_li.append(pred)
        y_li.append(data.y)

        # Convert batched Data back into individual graphs
        data_list = data.to_data_list()
        batch_size = len(data_list)

        # Split activations by node for each graph
        conv1_split, relu1_split, conv2_split, relu2_split, conv3_split = [], [], [], [], []
        node_ptr = 0
        for i in range(batch_size):
            num_nodes = data_list[i].x.shape[0]
            conv1_split.append(act['conv1'][node_ptr:node_ptr + num_nodes])
            relu1_split.append(act['relu1'][node_ptr:node_ptr + num_nodes])
            conv2_split.append(act['conv2'][node_ptr:node_ptr + num_nodes])
            relu2_split.append(act['relu2'][node_ptr:node_ptr + num_nodes])
            if "conv3" in act:  # ✅ only if present
                conv3_split.append(act['conv3'][node_ptr:node_ptr + num_nodes])
            node_ptr += num_nodes

        # Store per-graph activations
        for i in range(batch_size):
            x_dict[graph_idx] = data_list[i].x
            edge_dict[graph_idx] = data_list[i].edge_index

            activations_dict['conv1'][graph_idx] = conv1_split[i]
            activations_dict['relu1'][graph_idx] = relu1_split[i]
            activations_dict['conv2'][graph_idx] = conv2_split[i]
            activations_dict['relu2'][graph_idx] = relu2_split[i]
            if "conv3" in act:
                # Add conv3 slot if not already present
                if 'conv3' not in activations_dict:
                    activations_dict['conv3'] = {}
                activations_dict['conv3'][graph_idx] = conv3_split[i]

            activations_dict['global_pool'][graph_idx] = act['global_pool'][i].unsqueeze(0)
            activations_dict['fc'][graph_idx] = act['fc'][i].unsqueeze(0)
            graph_idx += 1

    pred_tensor = torch.cat(pred_li, dim=0)
    y_tensor = torch.cat(y_li, dim=0)

    gnn_graph_embed = torch.cat(
        [activations_dict['global_pool'][i] for i in range(len(activations_dict['global_pool']))],
        dim=0
    ).cpu().numpy()

    return pred_tensor.cpu(), y_tensor.cpu(), x_dict, edge_dict, activations_dict, gnn_graph_embed


def decision_tree_explainer(train_embed, train_preds, test_embed, test_preds, max_depth=1):
    """
    Train a Decision Tree as a surrogate model to explain GNN predictions.
    
    Args:
        train_embed (np.ndarray): Graph-level embeddings (train).
        train_preds (Tensor/np.ndarray): GNN predictions for training graphs.
        test_embed (np.ndarray): Graph-level embeddings (test).
        test_preds (Tensor/np.ndarray): GNN predictions for test graphs.
        max_depth (int): Maximum depth of the decision tree.

    Returns:
        clf: Trained decision tree classifier.
    """
    # Ensure numpy arrays
    if torch.is_tensor(train_preds):
        y_train = train_preds.cpu().numpy()
    else:
        y_train = train_preds

    if torch.is_tensor(test_preds):
        y_test = test_preds.cpu().numpy()
    else:
        y_test = test_preds

    X_train, X_test = train_embed, test_embed

    # Train DT
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Fidelity on train
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Decision Tree Fidelity (Train): {train_acc:.4f}")
    print("Train Classification Report:")

    labels = np.unique(y_train)  # detect which classes actually exist
    target_names = [f"Class {l}" for l in labels]
    print(classification_report(y_train, y_train_pred, labels=labels, target_names=target_names, zero_division=0))

    # Fidelity on test
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Decision Tree Fidelity (Test): {test_acc:.4f}")



    # Extract simple rule
    tree = clf.tree_
    val_idx = tree.feature[0]
    threshold = tree.threshold[0]
    #print(f"Extracted Rule: if Feature_{val_idx} <= {threshold:.4f} → Class 0 else Class 1")

    # Indices of graphs by predicted class
    idx_class0 = torch.nonzero(train_preds == 0).squeeze()
    idx_class1 = torch.nonzero(train_preds == 1).squeeze()


    return clf,idx_class0,idx_class1
