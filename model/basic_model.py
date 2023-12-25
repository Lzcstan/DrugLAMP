import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

from collections.abc import Mapping
from configs import get_model_defaults
from model.PGCA import GuidedCrossAttention
from model.cross_modality import CrossModality
from model.self_supervised_learning import SSL
from model.PMMA import MultiHeadLinearAttention, PairedMultimodelAttention

CONFIGS = {
    'LAMP': get_model_defaults,
}

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels.float())
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)
    
class DrugLAMPBase(nn.Module):
    def __init__(self, n_drug_feature, n_prot_feature, n_hidden=128, **cfg):
        super(DrugLAMPBase, self).__init__()
        drug_padding = cfg["DRUG"]["PADDING"]
        drug_in_feats = cfg["DRUG"]["NODE_IN_FEATS"]
        self.site_len = cfg['PROTEIN']['SITE_LEN']
        self.seq_len_q = cfg['PROTEIN']['SEQ_LEN']
        protein_padding = cfg["PROTEIN"]["PADDING"]
        protein_kernel_size = cfg["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = cfg["DECODER"]["IN_DIM"]
        mlp_binary = cfg["DECODER"]["BINARY"]
        mlp_out_dim = cfg["DECODER"]["OUT_DIM"]
        mlp_hidden_dim = cfg["DECODER"]["HIDDEN_DIM"]
        drug_embedding = n_hidden
        drug_hidden_feats = [n_hidden] * 3
        protein_emb_dim = n_hidden
        protein_num_filters = [n_hidden] * 3

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinCNN(protein_emb_dim, protein_num_filters, 
                                            protein_kernel_size, protein_padding)

        # SSL
        self.ssl_model = SSL(
            prot_extractor=self.protein_extractor,
            n_prot_feature=n_prot_feature,
            drug_ssl_type='simsiam',
            n_hidden=n_hidden
        )

        # CrossModality
        self.cm_model = CrossModality(
            use_cm=True,
            hidden_size=n_hidden,
            max_margin=cfg["RS"]["MAX_MARGIN"],
            n_re=cfg["RS"]["RESET_EPOCH"]
        )

        # LAMP config
        model_cfg = CONFIGS['LAMP'](n_hidden)

        # Drug branch
        self.lin_d1 = nn.Linear(n_drug_feature + 1, 2 * n_hidden)
        self.act_d = nn.GELU()
        self.d_norm = nn.LayerNorm(2 * n_hidden)
        self.lin_d2 = nn.Linear(2 * n_hidden, n_hidden)

        # Prot branch
        self.p_adaptor_wo_skip_connect = FeedForwardLayer(n_prot_feature + 1, n_hidden)
        self.lin_p1 = nn.Linear(n_prot_feature + 1, 2 * n_hidden)
        self.act_p = nn.GELU()
        self.p_norm = nn.LayerNorm(2 * n_hidden)
        self.lin_p2 = nn.Linear(2 * n_hidden, n_hidden)

        self.v_gca = GuidedCrossAttention(embed_dim=n_hidden, num_heads=1)
        self.v_mhla = MultiHeadLinearAttention(d_model=n_hidden * 2, d_diff=n_hidden * 8, nhead=8, dropout=model_cfg.mlha_dropout, activation='gelu')
        self.v_gca_norm = nn.LayerNorm(n_hidden * 2)
        self.x_gca = GuidedCrossAttention(embed_dim=n_hidden, num_heads=1)
        self.x_mhla = MultiHeadLinearAttention(d_model=n_hidden * 2, d_diff=n_hidden * 8, nhead=8, dropout=model_cfg.mlha_dropout, activation='gelu')
        self.x_gca_norm = nn.LayerNorm(n_hidden * 2)

        self.pmma = PairedMultimodelAttention(config=model_cfg, vis=False)
        self.mlp_classifier = MLP(mlp_in_dim * 2, mlp_hidden_dim * 2, mlp_out_dim * 2, binary=mlp_binary)

    def get_cross_attn_mat(self, modality='v'):
        if modality == 'v':
            self.A_v_gca = self.A_v_gca.cpu()
            return self.A_v_gca
        else:
            self.A_x_gca = self.A_x_gca.cpu()
            return self.A_x_gca
        
    def get_inter_attn_mat(self):
        return self.attn, self.guide_attn

    def forward(self, vd, vp, xd, xp, mode="train"):
        pass
        
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats) # Tid: Node feats variable
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats
    
class ProteinCNN(nn.Module): # Tid: add fill bit
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26 + 1, embedding_dim - 1, padding_idx=0) # Tid: add fill bit, fit mlm
        else:
            self.embedding = nn.Embedding(26 + 1, embedding_dim - 1) # Tid: add fill bit, fit mlm
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding='same')
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding='same')
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding='same')
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v, fill_mask):
        v = self.embedding(v.long())
        v = torch.cat((v, fill_mask.unsqueeze(-1)), dim=-1)
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v))) # -2
        v = self.bn2(F.relu(self.conv2(v))) # -5
        v = self.bn3(F.relu(self.conv3(v))) # -8
        v = v.view(v.size(0), v.size(2), -1)
        return v
    
class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_h)
        self.lin2 = nn.Linear(d_h, d_in)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_h)

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.norm(x)
        x = self.lin2(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(self.act1(self.fc1(x)))
        x = self.bn2(self.act2(self.fc2(x)))
        x = self.bn3(self.act3(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class GCN(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['both', 'both']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is not performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is not applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph for all layers. By default, will not
        allow zero in degree nodes.
    """

    def __init__(
        self,
        in_feats,
        hidden_feats=None,
        gnn_norm=None,
        activation=None,
        residual=None,
        batchnorm=None,
        dropout=None,
        allow_zero_in_degree=None,
    ):
        super(GCN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = ["both" for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0.0 for _ in range(n_layers)]
        lengths = [
            len(hidden_feats),
            len(gnn_norm),
            len(activation),
            len(residual),
            len(batchnorm),
            len(dropout),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_feats, gnn_norm, "
            "activation, residual, batchnorm and dropout to "
            "be the same, got {}".format(lengths)
        )

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GCNLayer(
                    in_feats,
                    hidden_feats[i],
                    gnn_norm[i],
                    activation[i],
                    residual[i],
                    batchnorm[i],
                    dropout[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] in initialization.
        """
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats
    
class GCNLayer(nn.Module):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    gnn_norm : str
        The message passing normalizer, which can be `'right'`, `'both'` or `'none'`. The
        `'right'` normalizer divides the aggregated messages by each node's in-degree.
        The `'both'` normalizer corresponds to the symmetric adjacency normalization in
        the original GCN paper. The `'none'` normalizer simply sums the messages.
        Default to be 'none'.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph. Defaults to False.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        gnn_norm="both",
        activation=None,
        residual=True,
        batchnorm=True,
        dropout=0.0,
        allow_zero_in_degree=False,
    ):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(
            in_feats=in_feats,
            out_feats=out_feats,
            norm=gnn_norm,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        """
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats
    
class GraphConv(nn.Module):
    r"""Graph convolutional layer from `Semi-Supervised Classification with Graph Convolutional
    Networks <https://arxiv.org/abs/1609.02907>`__

    Mathematically it is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ji}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`),
    and :math:`\sigma` is an activation function.

    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{e_{ji}}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    This is NOT equivalent to the weighted graph convolutional network formulation in the paper.

    To customize the normalization term :math:`c_{ji}`, one can first set ``norm='none'`` for
    the model, and send the pre-normalized :math:`e_{ji}` to the forward computation. We provide
    :class:`~dgl.nn.pytorch.EdgeWeightNorm` to normalize scalar edge weight following the GCN paper.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer.  Can be one of the following values:

        * ``right``, to divide the aggregated messages by each node's in-degrees,
          which is equivalent to averaging the received messages.

        * ``none``, where no normalization is applied.

        * ``both`` (default), where the messages are scaled with :math:`1/c_{ji}` above, equivalent
          to symmetric normalization.

        * ``left``, to divide the messages sent out from each node by its out-degrees,
          equivalent to random walk normalization.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise Exception('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise Exception('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_u('h', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
    
def expand_as_pair(input_, g=None):
    """Return a pair of same element if the input is not a pair.

    If the graph is a block, obtain the feature of destination nodes from the source nodes.

    Parameters
    ----------
    input_ : Tensor, dict[str, Tensor], or their pairs
        The input features
    g : DGLGraph or None
        The graph.

        If None, skip checking if the graph is a block.

    Returns
    -------
    tuple[Tensor, Tensor] or tuple[dict[str, Tensor], dict[str, Tensor]]
        The features for input and output nodes
    """
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()
            }
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_