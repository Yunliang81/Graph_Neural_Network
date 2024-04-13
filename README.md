# Graph_Neural_Network
GNN used in microbiome research
# Build a Graph neural network model for identifing microbial biomarker (Reference: https://doi.org/10.1093/bib/bbad448)

## Generate graph data using R package SpiecEasi

**Loading two OTU files, one was from control sample, the other was from disease sample. The OTU data have been normalized using total sum scaling method.**

```R

# loading OTU files, OTU table with bacterial genera in rows, and samples in columns

micro_data_h <- read.csv("/Users/yunliangli/Desktop/GNN/control_crc1.csv", header = TRUE, row.names = 1)  
micro_data_d <- read.csv("/Users/yunliangli/Desktop/GNN/case_crc1.csv", header = TRUE, row.names = 1)

# convert the data into percentages

micro_data_d <- data.matrix(t(micro_data_d)) * 100  
micro_data_h <- data.matrix(t(micro_data_h)) * 100
```

**Generate graph data from microbial correlation matrix**

```R
#remotes::install_github("zdk123/SpiecEasi")
library(Matrix)
library(SpiecEasi) 

Graph_nums = 500 
edges_list_h <- data.frame(x1=numeric(), x2=numeric(), x3=numeric(), x4=numeric())
edges_list_d <- data.frame(x1=numeric(), x2=numeric(), x3=numeric(), x4=numeric())

for (i in 1:Graph_nums) {

  sparcc.h <- sparcc(micro_data_h[c(sample(1: nrow(micro_data_h), 30)), ]) # calculating the SparCC correlation coefficients for a random subset of 30 rows from 
micro_data_h.
  sparcc.h.graph <- abs(sparcc.h$Cor) >= ((sum(abs(sparcc.h$Cor))-81)/(81*80)) # Set the threshold to filter out edges with low correlation coefficients.
  diag(sparcc.h.graph) <- 0 # By setting them to 0, we ensure that each node (taxon) is not considered to be correlated with itself in the network.
  sparcc.h.graph <- Matrix(sparcc.h.graph, sparse = TRUE) #Sparse matrices store only the non-zero elements and their positions, which can significantly reduce memory usage 
and improve computational efficiency for operations involving large matrices.
  elist.sparcc.h <- summary(sparcc.h.graph * sparcc.h$Cor) #calculates the weighted edges for the graph
  names(elist.sparcc.h) <- c('source', 'target', 'weight') # start node, end node, and the weight of the edge
  elist.sparcc.h <- elist.sparcc.h[order(elist.sparcc.h$source, elist.sparcc.h$target), ]
  
  sparcc.d <- sparcc(micro_data_d[c(sample(1: nrow(micro_data_d), 30)), ])
  sparcc.d.graph <- abs(sparcc.d$Cor) >= ((sum(abs(sparcc.h$Cor))-81)/(81*80))
  diag(sparcc.d.graph) <- 0
  sparcc.d.graph <- Matrix(sparcc.d.graph, sparse = TRUE)
  elist.sparcc.d <- summary(sparcc.d.graph * sparcc.d$Cor)
  names(elist.sparcc.d) <- c('source', 'target', 'weight')
  elist.sparcc.d <- elist.sparcc.d[order(elist.sparcc.d$source, elist.sparcc.d$target), ]
  
  graph_h_id <- matrix(c(rep(i, times = nrow(elist.sparcc.h))), ncol = 1)
  elist.sparcc.h <- cbind(graph_h_id, elist.sparcc.h)
  temp_h <- elist.sparcc.h[, c(1,3,2,4)]
  colnames(temp_h) <- colnames(elist.sparcc.h)
  elist.sparcc.h <- rbind(elist.sparcc.h, temp_h)
  edges_list_h <- rbind(edges_list_h, elist.sparcc.h)
  
  graph_d_id <- matrix(c(rep(i, times = nrow(elist.sparcc.d))), ncol = 1)
  elist.sparcc.d <- cbind(graph_d_id, elist.sparcc.d)
  temp_d <- elist.sparcc.d[, c(1,3,2,4)]
  colnames(temp_d) <- colnames(elist.sparcc.d)
  elist.sparcc.d <- rbind(elist.sparcc.d, temp_d)
  edges_list_d <- rbind(edges_list_d, elist.sparcc.d) 
  
}
#By subtracting 1 from the 'source' and 'target' columns, the node indices are shifted down by 1. This is a common practice to align  the node indices with zero-based 
indexing, which may be required by certain graph algorithms or data structures.
edges_list_h$source <- edges_list_h$source - 1
edges_list_h$target <- edges_list_h$target - 1

edges_list_d$source <- edges_list_d$source - 1
edges_list_d$target <- edges_list_d$target - 1

write.csv(edges_list_h, '/Users/yunliangli/Desktop/GNN/Graph_ctrl_crc1.csv', row.names = FALSE)
write.csv(edges_list_d, '/Users/yunliangli/Desktop/GNN/Graph_case_crc1.csv', row.names = FALSE)
```

## Build weight signed graph neural network

 **modules required**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl # note version compatiblity
from dgl.nn.pytorch import JumpingKnowledge, SAGEConv, SortPooling
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torchmetrics.classification as tc
import pandas as pd
import numpy as np
import csv

import torch.optim as optim
from torch.utils.data import DataLoader
```
**Define a function _get_dataset_ to convert graph data into DGL graphs used for downstream analysis** 
```python
def get_dataset(health, disease):
    # loading graph data, and extracting source and target nodes as tensors and edge weight as floattensors 
    microbio_data_h = pd.read_csv(health)
    microbio_data_d = pd.read_csv(disease)

    G_h_dataset = {}
    G_d_dataset = {}

    u_h = microbio_data_h['source']
    u_h = torch.as_tensor(u_h)
    v_h = microbio_data_h['target']
    v_h = torch.as_tensor(v_h)
    edges_weight_h = torch.FloatTensor(microbio_data_h['weight'])
   
    # The following loop iterates over each unique graph ID in the control sample data, constructs a DGL graph for each graph ID, assigns edge weights, and labels each graph 
with 0 (indicating it belongs to the control group)
    ctrl_edge_nums = [0]
    for i in range(0, max(microbio_data_h['graph_id'])):
        edges_nums = np.sum(microbio_data_h['graph_id'] == (i + 1))
        ctrl_edge_nums.append(edges_nums)
        G_h_dataset[i] = dgl.graph((u_h[sum(ctrl_edge_nums[0:(i+1)]) : sum(ctrl_edge_nums[0:(i+2)])],
                                   v_h[sum(ctrl_edge_nums[0:(i+1)]) : sum(ctrl_edge_nums[0:(i+2)])]),
                                  num_nodes=81)
        G_h_dataset[i].edata['w'] = edges_weight_h[sum(ctrl_edge_nums[0:(i+1)]) : sum(ctrl_edge_nums[0:(i+2)])]
        label_h = 0
        G_h_dataset[i] = (G_h_dataset[i], label_h)

    u_d = microbio_data_d['source']
    u_d = torch.as_tensor(u_d)
    v_d = microbio_data_d['target']
    v_d = torch.as_tensor(v_d)
    edges_weight_d = torch.FloatTensor(microbio_data_d['weight'])

    case_edge_nums = [0]
    for j in range(0, max(microbio_data_d['graph_id'])):
        edges_nums = np.sum(microbio_data_d['graph_id'] == (j + 1))
        case_edge_nums.append(edges_nums)
        G_d_dataset[j] = dgl.graph((u_d[sum(case_edge_nums[0:(j+1)]) : sum(case_edge_nums[0:(j+2)])],
                                   v_d[sum(case_edge_nums[0:(j+1)]) : sum(case_edge_nums[0:(j+2)])]),
                                  num_nodes=81)
        G_d_dataset[j].edata['w'] = edges_weight_d[sum(case_edge_nums[0:(j+1)]) : sum(case_edge_nums[0:(j+2)])]
        label_d = 1
        G_d_dataset[j] = (G_d_dataset[j], label_d)
#the following code converts the dictionary G_h_dataset into a list of tuples Each tuple in the list represents a data sample.
    dataset_h = list(G_h_dataset.values())
    dataset_d = list(G_d_dataset.values())
    return dataset_h, dataset_d  
```

**Define function _get_batch_id_ to generate a tensor containing batch IDs for each node in a batched graph**
```python
def get_batch_id(num_nodes):
# Convert the num_nodes array obtained from batch graph to batch_id array
# for each node.

# num_nodes (torch.Tensor): The tensor whose element is the number of nodes
# in each graph in the batch graph.

    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)
```

**Define function _topk_, Top-k Selection Pooling, a pooling method used for GSAPooling (Graph Self-Adaptive Pooling)**
```python
def topk(x, ratio, batch_id, num_nodes):
# The top-k pooling method. Given a graph batch, this method will pool out some
# nodes from input node feature tensor for each graph according to the given ratio.
# x (torch.Tensor): The input node feature batch-tensor to be pooled.
# ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
# tensor will be pooled out.
# batch_id (torch.Tensor): The batch_id of each element in the input tensor.
# num_nodes (torch.Tensor): The number of nodes of each graph in batch.

# Returns:
# perm (torch.Tensor): The index in batch to be kept.
# k (torch.Tensor): The remaining number of nodes for each graph.

    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
    
    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)
    
    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)
    
    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    
    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + i * max_num_nodes 
        for i in range(batch_size)
    ]
    
    mask = torch.cat(mask, dim=0)
    perm = perm[mask]
    
    return perm, k
```

**Define a GSAPool class, implement a Structure-Feature based Graph Self-adaptive Pooling layer. In this class, function _topk_ and _get_batch_id_ were called in the defined 
_forward_ function**
```python
class GSAPool(nn.Module):
    # The Structure-Feature based Graph Self-adaptive Pooling layer
    #  in_dim (int): The dimension of node feature.
    #  ratio (float, optional): The pool ratio which determines the amount of nodes
    #  remain after pooling. (default: :obj:`0.5`)
    #  conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
    #  compute scale for each node. (default: :obj:`dgl.nn.SAGEConv`)
    #  non_linearity (Callable, optional): The non-linearity function, a pytorch function.
    #  (default: :obj:`torch.tanh`)
    
    def __init__(self, in_dim, ratio=0.5, alpha=0.6, conv_op=SAGEConv):
        super(GSAPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.alpha = alpha
        self.sbtl_layer = conv_op(in_dim, 1, 'lstm')
        self.fbtl_layer = nn.Linear(in_dim, 1)
        
    def forward(self, graph, feature):
        score_s = self.sbtl_layer(graph, feature).squeeze()
        score_f = self.fbtl_layer(feature).squeeze()
        score = score_s * self.alpha + score_f * (1 - self.alpha) # Both structure and feature information were considered
        perm, next_batch_num_nodes = topk(score,
                                          self.ratio,
                                          get_batch_id(graph.batch_num_nodes()),
                                          graph.batch_num_nodes()
                                         )

        graph = dgl.node_subgraph(graph, perm)
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm
```

**Define function _collate_ to customize the way to batch multiple graphs together**
```python
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batch_graphs = dgl.batch(graphs)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return batch_graphs, batch_labels
```

**Define function _split_train_test_ to split the data into training and testing sets**
```python
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = [data[i] for i in train_indices]
    test_set = [data[i] for i in test_indices]
    return train_set, test_set
```


**Define class _WSGConv_, a Weighted Signed Graph Convolutional layer, and it learns from positive and negative signed graphs separately. The update node representations 
capture information from the original graph as well as the positive and negative signed graphs**


```python
class WSGConv(nn.Module):
    
    def __init__(self, in_feats, out_feats, bias=True, activation=None):
        super(WSGConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation

        self.fc_layer = nn.Linear(self._in_feats * 3, out_feats, bias=True)
        self.coef_self = nn.Parameter(torch.FloatTensor([1.0]))
        self.coef_posi = nn.Parameter(torch.FloatTensor([1.0]))
        self.coef_nega = nn.Parameter(torch.FloatTensor([1.0]))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)
            
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            
            #  Split a weighted signed graph into positive and negative signed graph
            h_self = feat
            graph.ndata['h'] = h_self
            
            posiedges_index = torch.nonzero(graph.edata['w'] > 0).squeeze()
            negaedges_index = torch.nonzero(graph.edata['w'] < 0).squeeze()

            g_positive = dgl.graph((graph.edges()[0], graph.edges()[1]), num_nodes=graph.num_nodes())
            g_positive.ndata['h'] = feat
            g_positive.edata['w'] = graph.edata['w']
            g_positive.remove_edges(negaedges_index)

            g_negative = dgl.graph((graph.edges()[0], graph.edges()[1]), num_nodes=graph.num_nodes())
            g_negative.ndata['h'] = feat
            g_negative.edata['w'] = abs(graph.edata['w'])
            g_negative.remove_edges(posiedges_index)

            g_positive.edata['w'] = edge_softmax(g_positive, g_positive.edata['w'])
            g_negative.edata['w'] = edge_softmax(g_negative, g_negative.edata['w'])

            # Message Passing
            g_positive.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            g_negative.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            
            graph.ndata['h'] = self.coef_self * graph.ndata['h']
            g_positive.ndata['h'] = self.coef_posi * g_positive.ndata['h']
            g_negative.ndata['h'] = self.coef_nega * g_negative.ndata['h']
            
            feat_box = [graph.ndata['h'], g_positive.ndata['h'], g_negative.ndata['h']]
            feat_combined = torch.cat(feat_box, dim=1)
            h_new = feat_combined
            
            rst = self.fc_layer(h_new)

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)

            return rst
```

**Define a graph convolutional neural network (WSGCN) model for node classification tasks. Class WSGConv and GSAPool were called. It applies a fully connected layer (fc) to 
the flattened output of the convolutional layers.
ReLU activation and dropout are applied for regularization**

```python
class WSGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, n_classes):
        super(WSGCN, self).__init__()
        self.conv1 = WSGConv(in_dim, hid_dim, activation=torch.relu)
        self.conv2 = WSGConv(hid_dim, hid_dim, activation=torch.relu)
        
        self.jk = JumpingKnowledge()
        self.pool = GSAPool(64, ratio=1.0, alpha=0.5)
        self.conv1D_1 = nn.Conv1d(1, 16, kernel_size=64, stride=64)
        self.maxpool = nn.MaxPool1d(2)
        self.conv1D_2 = nn.Conv1d(16, 32, kernel_size=3, stride=3)
        
        self.fc = nn.Linear(32*13, 128)
        self.classify = nn.Linear(128, n_classes)
        
    def forward(self, graph):
        h = graph.in_degrees().view(-1, 1).float()
        h1 = self.conv1(graph, h)
        h1 = h1.flatten(1)
        h2 = self.conv2(graph, h1)
        h2 = h2.flatten(1)
        
        h = self.jk([h1, h2])
        _, h, _ = self.pool(graph, h)
        h = h.view(-1, 1, 64*81)
        
        h = self.conv1D_1(h)
        h = self.maxpool(h)
        h = self.conv1D_2(h)
        
        h = h.flatten(1)
        h = F.relu(self.fc(h))
        h = F.dropout(h, p=0.5)
        h = self.classify(h)
        
        with graph.local_scope():
            return F.log_softmax(h, dim=-1)
```

Define function _train_ to train the model. It takes the model, optimizer, training data loader, and device as input parameters. 
```python
def train(model, optimizer, trainloader, device):
    model.to(device)
    model.train()
    total_loss = 0.0
    train_correct = 0.0
    num_batches = len(trainloader)
    train_pred, train_label = [], []
    num_graphs = 0
    
    train_acc = tc.BinaryAccuracy()
    train_recall = tc.BinaryRecall()
    train_precision = tc.BinaryPrecision()
    train_auc = tc.BinaryAUROC(thresholds=None)
    train_specificity = tc.BinarySpecificity()
    
    for iter, (batch_graphs, batch_labels) in enumerate(trainloader):
        num_graphs += batch_labels.size(0)
        optimizer.zero_grad()
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        pred = out.argmax(dim=1)
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        train_pred += pred.detach().cpu().numpy().tolist()
        train_label += batch_labels.cpu().numpy().tolist()
        total_loss += loss.item()
    
    train_pred = torch.tensor(train_pred)
    train_label = torch.tensor(train_label)
    
    acc = train_acc(train_pred, train_label)
    auc = train_auc(train_pred, train_label)
    recall = train_recall(train_pred, train_label)
    precision = train_precision(train_pred, train_label)
    specificity = train_specificity(train_pred, train_label)
        
    return acc, auc, recall, precision, specificity, total_loss / num_batches

```

**Define function _test_ for evaluating the model's performance on the test dataset**
```python
@torch.no_grad() # @torch.no_grad() is a context manager in PyTorch used to specify that the code block inside it does not require gradient computation. When testing a 
model, gradients are typically not needed because we are only performing forward inference without backward propagation and parameter updates.
def test(model, testloader, device):
    model.to(device)
    model.eval()
    loss = 0.0
    num_graphs = 0
    test_pred, test_label = [], []
    
    test_acc = tc.BinaryAccuracy()
    test_recall = tc.BinaryRecall()
    test_precision = tc.BinaryPrecision()
    test_auc = tc.BinaryAUROC(thresholds=None)
    test_specificity = tc.BinarySpecificity()
    
    for iter, (batch_graphs, batch_labels) in enumerate(testloader):
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        pred = out.argmax(dim=1)
        test_pred += pred.detach().cpu().numpy().tolist()
        test_label += batch_labels.cpu().numpy().tolist()
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        
    test_pred = torch.tensor(test_pred)
    test_label = torch.tensor(test_label)
    
    acc = test_acc(test_pred, test_label)
    auc = test_auc(test_pred, test_label)
    recall = test_recall(test_pred, test_label)
    specificity = test_specificity(test_pred, test_label)
    precision = test_precision(test_pred, test_label)
    
    return acc, auc, recall, precision, specificity, loss / num_graphs
```

**_I concatenated above Python code into TRIAL.py_**

Give it a try.

```python
import TRIAL
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl # note version compatiblity
from dgl.nn.pytorch import JumpingKnowledge, SAGEConv, SortPooling
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torchmetrics.classification as tc
import pandas as pd
import numpy as np
import csv

import torch.optim as optim
from torch.utils.data import DataLoader


# Step 1: Prepare graph data and retrieve train/test index ============================= #  
health='/Users/yunliangli/Desktop/GNN/Graph_ctrl_crc1.csv' 
disease='/Users/yunliangli/Desktop/GNN/Graph_case_crc1.csv'  
dataset_h, dataset_d = TRIAL.get_dataset(health,disease)
train_set_h, test_set_h = TRIAL.split_train_test(dataset_h, 0.2)
train_set_d, test_set_d = TRIAL.split_train_test(dataset_d, 0.2)
train_set = train_set_h + train_set_d
test_set = test_set_h + test_set_d

train_loader = DataLoader(train_set, batch_size = 128, shuffle = True, collate_fn = TRIAL.collate)
test_loader = DataLoader(test_set, batch_size = 64, shuffle = True, collate_fn = TRIAL.collate)
device = torch.device("cpu") # I changed cuda:0 to cpu, as MacOS M3 GPU is not comptible with DGL

# Step 2: Create model =================================================================== #
model = TRIAL.WSGCN(1, 32, 2)

# Step 3: Create training components ===================================================== #
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Step 4: training epoches =============================================================== #
best_test_acc = 0.0
best_epoch = 0
epochs = 6

for epoch in range(epochs):
    train_acc, train_auc, train_recall, train_precision, train_specificity, train_loss = TRIAL.train(model, optimizer, train_loader, device)
    test_acc, test_auc, test_recall, test_precision, test_specificity, test_loss = TRIAL.test(model, test_loader, device)
    if best_test_acc < test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1
    if (epoch + 1) % 2 == 0:
        log_format = ("Epoch {}: TRIAL.Train loss={:.4f}, Train acc={:.4f}; Test loss={:.4f}, " "Test acc={:.4f}, Test auc={:.4f}, Test recall={:.4f}, Test 
specifity={:.4f}")
        print(log_format.format(epoch + 1, train_loss, train_acc, test_loss, test_acc,
                                test_auc, test_recall, test_specificity))

print("Best Epoch: {}, Best test acc: {:.4f}".format(best_epoch, best_test_acc))
```
It run successfully.

**To identify which nodes (genera) have significant effect on the model prediction ability, method of "kicking out" node in the graph data was applied. The nodes which have 
big change of model prediction accuracy were considered as biomarkers**

```python
for n in range(81):  # There were 81 nodes in total 
   
    Importance_score_lor_minus_abs = []
    Importance_score_lor_d = []
    
    Importance_score_minus_abs = []
    Importance_score_d = [] 
    
    for i in range(10): # The method run 10 times with 50 epoches
        
        dataset_h, dataset_d = get_dataset()
        train_set_h, test_set_h = split_train_test(dataset_h, 0.2)
        train_set_d, test_set_d = split_train_test(dataset_d, 0.2)
        train_set = train_set_h + train_set_d
        test_set = test_set_h + test_set_d
        
        # Step 1: Prepare graph data and retrieve train/test index ============================= #       
        train_loader = DataLoader(train_set, batch_size = 128, shuffle = True, collate_fn = collate)
        test_loader = DataLoader(test_set, batch_size = 64, shuffle = True, collate_fn = collate)
        device = torch.device("cpu")

        # Step 2: Create model =================================================================== #
        model = WSGCN(1, 32, 2)

        # Step 3: Create training components ===================================================== #
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        # Step 4: training epoches =============================================================== #
        epochs = 50
        for epoch in range(epochs):
            train_acc, _, _, _, _, train_loss = train(model, optimizer, train_loader, device)
        
        _, _, test_recall, _, test_specificity, _ = test(model, test_loader, device)
        
        test_nums = len(test_set)
        for j in range(test_nums):
            edge_ids_out = test_set[j][0].out_edges(torch.tensor([n]), form='eid')
            edge_ids_in = test_set[j][0].in_edges(torch.tensor([n]), form='eid')
            edge_ids = torch.cat((edge_ids_out, edge_ids_in), dim=-1)
            test_set[j][0].edata['w'][edge_ids] = 0
            
        test_drop_loader = DataLoader(test_set, batch_size = 64, shuffle = True, collate_fn = collate)
        _, _, test_recall_new, _, test_specificity_new, _ = test(model, test_drop_loader, device)
        
        
        #  Step 5: node importance score caculation =============================================================== #
        eps = 1e-6
        lor_1_1 = np.log(test_recall_new+eps) - np.log(1-test_recall_new+eps)
        lor_1_2 = np.log(test_recall+eps) - np.log(1-test_recall+eps)
        lor_2_1 = np.log(test_specificity_new+eps) - np.log(1-test_specificity_new+eps)
        lor_2_2 = np.log(test_specificity+eps) - np.log(1-test_specificity+eps)
        
        score_1_1 = (test_recall_new-test_recall) / test_recall
        score_1_2 = (test_specificity_new-test_specificity) / test_specificity
        
        score_lor_minus_abs = abs(abs(lor_1_1 - lor_1_2) - abs(lor_2_1 - lor_2_2))
        score_lor_d = abs(lor_1_1 - lor_1_2)
        
        score_minus_abs = abs(abs(score_1_1) - abs(score_1_2))
        score_d = abs(score_1_1)
        
        
        Importance_score_lor_d.append(score_lor_d)
        Importance_score_lor_minus_abs.append(score_lor_minus_abs)
        
        Importance_score_d.append(score_d)
        Importance_score_minus_abs.append(score_minus_abs)
    
    s_lor_d_mean = [np.mean(Importance_score_lor_d)]
    s_lor_minus_abs_mean = [np.mean(Importance_score_lor_minus_abs)]
    
    s_d_mean = [np.mean(Importance_score_d)]
    s_minus_abs_mean = [np.mean(Importance_score_minus_abs)]
    
    
    with open('score_dh_crc1.csv', 'a+', newline='') as f:
        csv_write = csv.writer(f)
        data_row = s_minus_abs_mean
        csv_write.writerow(data_row)

```

**Note:** the source code was from https://github.com/panshuheng/WSGMB/tree/master, I did some editing for my learning. 

