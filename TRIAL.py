# coding: utf-8

def get_dataset(health,disease):
    microbio_data_h = pd.read_csv(health)
    microbio_data_d = pd.read_csv(disease)

    G_h_dataset = {}
    G_d_dataset = {}

    u_h = microbio_data_h['source']
    u_h = torch.as_tensor(u_h)
    v_h = microbio_data_h['target']
    v_h = torch.as_tensor(v_h)
    edges_weight_h = torch.FloatTensor(microbio_data_h['weight'])

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

    dataset_h = list(G_h_dataset.values())
    dataset_d = list(G_d_dataset.values())
    return dataset_h, dataset_d


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
        score = score_s * self.alpha + score_f * (1 - self.alpha)
        perm, next_batch_num_nodes = topk(score,
                                          self.ratio,
                                          get_batch_id(graph.batch_num_nodes()),
                                          graph.batch_num_nodes()
                                         )

        graph = dgl.node_subgraph(graph, perm)
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm



#  get large graphs
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batch_graphs = dgl.batch(graphs)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return batch_graphs, batch_labels

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = [data[i] for i in train_indices]
    test_set = [data[i] for i in test_indices]
    return train_set, test_set



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
        

@torch.no_grad()
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
