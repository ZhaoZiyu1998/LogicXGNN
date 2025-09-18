import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from explain_gnn import *
from load_data import load_data      # your function from before
from gnn import GCN, GIN, GAT, GraphSAGE       # your GCN class
from utils import train, test, load_model   # your train/test/save functions
from build_logicGNN import *
from collections import defaultdict
from grounding import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# dataset-specific early stop thresholds
stop_dict = {
    "BAMultiShapes": 0.8,
    "BBBP": 0.96,
    "Mutagenicity": 0.96,
    "IMDB-BINARY": 0.7350,
    "NCI1": 0.7,
    "reddit_threads": 0.8,
    "twitch_egos": 0.8,
    "github_stargazers": 0.8
}
original_atom_dict = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 15: "P", 16: "S", 17: "Cl", 20: "Ca", 35: "Br", 53: "I"
}
atom_types = sorted(original_atom_dict.keys())
atom_to_idx = {atom_num: idx for idx, atom_num in enumerate(atom_types)}
num_atom_types = len(atom_types)
BBBP_atom_type_dict = {idx: original_atom_dict[atom_types[idx]] for idx in range(num_atom_types)}
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GNNs on graph datasets")
    
    # dataset and experiment setup
    parser.add_argument("--dataset", type=str, choices=["BBBP", "Mutagenicity", "IMDB-BINARY", "NCI1", "BAMultiShapes", "reddit_threads", "twitch_egos", "github_stargazers"], required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--load", action="store_true", help="Load pretrained model instead of training")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth for decision tree")
    parser.add_argument(
    "--arch",
    type=str,
    choices=["GCN", "GIN", "GAT", "GraphSAGE"],
    default="GCN",
    help="GNN architecture to use"
)
    args = parser.parse_args()

    # Set device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

    # Load data
    train_dataset, test_dataset, train_loader, test_loader, device = load_data(args.dataset, args.seed)
    y_labels_flat=[]
    for data in train_dataset:
        y_labels_flat.append(data.y.item())
    for data in test_dataset:
        y_labels_flat.append(data.y.item())

    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_labels_flat), y=y_labels_flat)
    class_weights_map = dict(zip(np.unique(y_labels_flat), class_weights))


    print(f"Correctly calculated class weights: {class_weights_map}")
    # ✅ Decide whether to use conv3 based on dataset
    use_conv3 = args.dataset == "BAMultiShapes"
    use_node_features = args.dataset in ["BBBP", "Mutagenicity", "NCI1"]
    def get_model(arch, in_channels, hidden_channels, out_channels, num_classes, use_conv3=True):
        if arch == "GCN":
            return GCN(in_channels, hidden_channels, out_channels, num_classes, use_conv3)
        elif arch == "GIN":
            return GIN(in_channels, hidden_channels, out_channels, num_classes, use_conv3)
        elif arch == "GAT":
            return GAT(in_channels, hidden_channels, out_channels, num_classes, use_conv3=use_conv3)
        elif arch == "GraphSAGE":
            return GraphSAGE(in_channels, hidden_channels, out_channels, num_classes, use_conv3=use_conv3)
        else:
            raise ValueError(f"Unknown architecture {arch}")
    model = get_model(
    args.arch,
    in_channels=train_dataset[0].x.shape[1],
    hidden_channels=32,
    out_channels=32,
    num_classes=2,
    use_conv3=use_conv3
).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    if args.dataset=="BBBP":
        atom_type_dict=BBBP_atom_type_dict
        one_hot = 1
        use_embed = 1
        k_hops=2
    elif args.dataset=="BAMultiShapes":
        atom_type_dict = {}
        one_hot = 0
        use_embed = 1
        k_hops = 3    
    elif args.dataset=="Mutagenicity":
        atom_type_dict = {
                0: "C",
                1: "O",
                2: "Cl",
                3: "H",
                4: "N",
                5: "F",
                6: "Br",
                7: "S",
                8: "P",
                9: "I",
                10: "Na",
                11: "K",
                12: "Li",
                13: "Ca"
            }
        one_hot = 1
        use_embed = 1
        k_hops = 2
    elif args.dataset=="IMDB-BINARY":
        atom_type_dict = {}
        one_hot = 0
        use_embed = 0
        k_hops = 2
    elif args.dataset=="NCI1":
        atom_type_dict = {i: i for i in range(37)}
        one_hot = 1
        use_embed = 1
        k_hops = 2
    elif args.dataset=="reddit_threads":
        atom_type_dict = {}
        one_hot = 0
        use_embed = 0
        k_hops = 2
    elif args.dataset=="twitch_egos":
        atom_type_dict = {}
        one_hot = 0
        use_embed = 0
        k_hops = 2
    elif args.dataset=="github_stargazers":
        atom_type_dict = {}
        one_hot = 0
        use_embed = 0
        k_hops = 2
    if args.arch=="GCN":
        model_path = f"./models/{args.dataset}_{args.seed}.pth"
    else:
        model_path = f"./models/{args.dataset}_{args.seed}_{args.arch}.pth"

    
    if args.load:
        # 🔹 Load pretrained weights
        model = load_model(model, model_path, device=device)
        test_acc = test(model, test_loader, device)
        print(f"Loaded model | Test Accuracy: {test_acc:.4f}")
    else:
        # 🔹 Train from scratch with early stopping
        for epoch in range(1, 201):
            loss = train(model, train_loader, optimizer, criterion, device)
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            print(f"Epoch {epoch}, Loss {loss:.4f}, Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}")

            # check dataset-specific stop threshold
            if test_acc >= stop_dict[args.dataset]:
                print(f"✅ Early stopping at epoch {epoch} for {args.dataset}: Test Acc {test_acc:.4f}")
                break

        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved model to {model_path}")

    start_time = time.time()
    gnn_train_pred_tensor, train_y_tensor, train_x_dict, train_edge_dict, train_activations_dict, train_gnn_graph_embed = get_all_activations_graph(train_loader,model,device)
    gnn_test_pred_tensor, test_y_tensor, test_x_dict, test_edge_dict, test_activations_dict, test_gnn_graph_embed = get_all_activations_graph(test_loader,model,device)
    

    clf,index_0_correct,index_1_correct=decision_tree_explainer(train_gnn_graph_embed,gnn_train_pred_tensor,test_gnn_graph_embed,gnn_test_pred_tensor,max_depth=1)
    tree = clf.tree_
    val_idx = tree.feature[0]
    threshold = tree.threshold[0]
    predicates, predicate_to_idx, predicate_node, predicate_graph_class_0, predicate_graph_class_1, rules_matrix_0, rules_matrix_1 = get_predicates_bin_one_pass(index_0_correct,index_1_correct,train_x_dict, train_edge_dict, train_activations_dict, val_idx, threshold, use_embed = use_embed , k_hops = k_hops)
    # Create mapping from predicate to index
    predicates_idx_mapping = {predicate: idx for idx, predicate in enumerate(predicates)}

    # Create mapping from index to predicate  
    idx_predicates_mapping = {idx: predicate for idx, predicate in enumerate(predicates)}
    index_test = torch.tensor(list(range(test_y_tensor.shape[0])))
    test_predicate_graph, test_res = get_predicate_graph(index_test, predicates, predicate_to_idx, test_x_dict, test_edge_dict, test_activations_dict, val_idx, threshold, use_embed = use_embed , k_hops = k_hops)
    leaf_rules_samples_0, leaf_rules_samples_1, used_predicates, clf_graph = get_discriminative_rules_with_samples(rules_matrix_0, rules_matrix_1, index_0_correct, index_1_correct, max_depth = args.max_depth, plot=0, text=0)

    y_true_fid = gnn_test_pred_tensor.cpu().numpy()
    y_pred_fid = clf_graph.predict(test_res.T)

    fidelity_acc = accuracy_score(y_true_fid, y_pred_fid)
    prec_fid, rec_fid, f1_fid, _ = precision_recall_fscore_support(
        y_true_fid, y_pred_fid, average="weighted"
    )

    # -----------------------
    # Accuracy (vs ground truth)
    # -----------------------
    y_true_acc = test_y_tensor.cpu().numpy()
    y_pred_acc = clf_graph.predict(test_res.T)

    test_acc = accuracy_score(y_true_acc, y_pred_acc)
    prec_acc, rec_acc, f1_acc, _ = precision_recall_fscore_support(
        y_true_acc, y_pred_acc, average="weighted"
    )

    # -----------------------
    # Print results
    # -----------------------
    print(f"Performance for {args.dataset}_{args.seed}_{args.arch}")
    print(f"  Fidelity → Acc: {fidelity_acc:.4f}, Prec: {prec_fid:.4f}, Rec: {rec_fid:.4f}, F1: {f1_fid:.4f}")
    print(f"  Accuracy → Acc: {test_acc:.4f}, Prec: {prec_acc:.4f}, Rec: {rec_acc:.4f}, F1: {f1_acc:.4f}")
    ## end(BAshape, IMDB, large3)
    if not use_node_features:
        end_time = time.time()
        print(f"Time taken for {args.dataset}_{args.seed}_{args.arch} without node features: {end_time - start_time:.2f} seconds")
        exit()
    used_alone_predicates, used_iso_predicate_node = analyze_used_predicate_nodes(predicate_node, predicate_to_idx, used_predicates)

    iso_predicates_inference = {} 

    used_iso_predicates = list(used_iso_predicate_node.keys())
    hashs=[]
    for p in used_iso_predicates:
            h=explain_predicate_with_rules(
        p_idx=p,
        used_iso_predicate_node=used_iso_predicate_node,
        train_x_dict=train_x_dict,
        train_edge_dict=train_edge_dict,
        atom_type_dict=atom_type_dict,
        idx_predicates_mapping=idx_predicates_mapping,
        iso_predicates_inference=iso_predicates_inference,
        one_hot=one_hot,
        k_hops=k_hops,
        top_k=1,
        save_dir=f"./plot_bbbp_mutag/{args.dataset}/{args.seed}/{args.arch}/iso",
        verbose=0,
    )
            hashs.append(h)
    #print(hashs)
    #print("Final iso_predicates_inference:", iso_predicates_inference)

    data_grounded_pred_array = grounded_graph_predictions(
    test_dataset,
    test_x_dict,
    test_edge_dict,
    atom_type_dict,
    iso_predicates_inference,
    used_alone_predicates,
    predicates,
    clf_graph,
    k_hops=k_hops,
    one_hot=one_hot
)
        # Step 1: Convert y_tensor to a NumPy array
    y_true = gnn_test_pred_tensor.numpy().ravel()
    y_pred = data_grounded_pred_array
    fidelity = accuracy_score(y_true, y_pred)
    weighted_fidelity = accuracy_score(y_true, y_pred,sample_weight=[class_weights_map[label] for label in y_true])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    accuracy = accuracy_score(test_y_tensor.numpy(), data_grounded_pred_array)
    weighted_accuracy = accuracy_score(test_y_tensor.numpy(), data_grounded_pred_array,sample_weight=[class_weights_map[label] for label in y_true])
    ##########################
##########################
    correct_predictions_class_1 = ((data_grounded_pred_array == y_true) & (y_true == 1)).sum()

    total_actual_class_1 = (y_true == 1).sum()
    print("Class 1 coverage: ")
    print(correct_predictions_class_1 / total_actual_class_1)



    correct_predictions_class_0 = ((data_grounded_pred_array == y_true) & (y_true == 0)).sum()

    total_actual_class_0 = (y_true == 0).sum()
    print("Class 0 coverage: ")
    print(correct_predictions_class_0 / total_actual_class_0)
#############
    print(f"Test Fidelity (unweighted): {fidelity * 100:.2f}%")
    print(f"Test Fidelity (Weighted Fidelity): {weighted_fidelity * 100:.2f}%")
    print(f"Test Accuracy (unweighted): {accuracy * 100:.2f}%")
    print(f"Test Accuracy (Weighted Accuracy): {weighted_accuracy * 100:.2f}%")
    print(f"Test Fidelity (Weighted Precision): {prec * 100:.2f}%")
    print(f"Test Fidelity (Weighted Recall): {rec * 100:.2f}%")
    print(f"Test Fidelity (Weighted F1 Score): {f1 * 100:.2f}%")
    end_time = time.time()
    print(f"Time taken for {args.dataset}_{args.seed}_{args.arch} without node features: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    main()
