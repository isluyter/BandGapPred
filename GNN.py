import torch
import torch.nn as nn
class GNNLayer(nn.Module):
    def __init__(self, ndim, edim):
        super().__init__()
        self.ndim = ndim
        self.edim = edim
        self.msgMLP = nn.Sequential(
            nn.Linear(ndim*2+edim, 2*ndim),
            nn.LeakyReLU(),
            nn.Linear(2*ndim, ndim),
            nn.LeakyReLU(),
            nn.Linear(ndim, ndim)
        )
        self.updMLP = nn.Sequential(
            nn.Linear(ndim, ndim),
            nn.LeakyReLU(),
            nn.Linear(ndim, ndim)
        )
    #nodes.shape = (num_atoms, ndim)
    #edges.shape = (num_edges, edim)
    #edge_idx.shape = (2, num_edges)
    def forward(self, nodes, edges, edge_idx):
        num_atoms = nodes.shape[0]
        unprc_msgs = torch.cat([nodes[edge_idx[0]], nodes[edge_idx[1]], edges], dim = 1) #shape = (num_edges, ndim*2+edim)
        msgs = self.msgMLP.forward(unprc_msgs) #shape = (num_edges, ndim)

        deg_count = torch.bincount(edge_idx[1], minlength=num_atoms).clamp(min=1) 

        tgt_idx = edge_idx[1].unsqueeze(1).repeat(1, self.ndim)                    
        out = torch.zeros_like(nodes)
        out = out.scatter_add(dim=0, src=msgs, index=tgt_idx)                       

        out = out / deg_count.unsqueeze(1)  
        
        #edge_count_idx = edge_idx.flatten()
        #deg_count = torch.bincount(edge_count_idx, minlength=num_atoms)

        #edge_idx = edge_idx.unsqueeze(2).repeat(1,1, self.ndim)
        #out= torch.zeros_like(nodes)
        #out= out.scatter_add(dim=0, src=msgs, index=edge_idx[0])
        #out= out.scatter_add(dim=0, src=msgs, index=edge_idx[1])

        #deg_count = deg_count.unsqueeze(1).repeat(1, self.ndim)
        #out/= deg_count

        nodes = nodes + out

        nodes = self.updMLP.forward(nodes) #shape = (num_atoms, ndim)
        return nodes
        
#node --> 13 features upsamplimg to ndim
#edge --> 2 features upsampling to edim
#edge indexes --> shows connections
#global --> 7 (onehot encoded)



class GNN(nn.Module):
    def __init__(self, nodefeats, edgefeats, ndim, edim, globfeats, num_layers):
        super().__init__()
        self.nodefeats = nodefeats
        self.edgefeats = edgefeats
        self.ndim = ndim
        self.edim = edim
        self.globfeats = globfeats
        self.num_layers = num_layers
        self.layers = nn.ModuleList([GNNLayer(ndim,edim) for _ in range(num_layers)])

        self.finalMLP = nn.Sequential(
            nn.Linear(ndim+globfeats, ndim),
            nn.LeakyReLU(),
            nn.Linear(ndim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        self.nodeproj = nn.Parameter(torch.randn(nodefeats, ndim))
        self.edgeproj = nn.Parameter(torch.randn(edgefeats, edim))

    def forward(self, atoms, bonds, edge_idx, globfeats):
        nodes = atoms @ self.nodeproj
        edges = bonds @ self.edgeproj
        for i in range(self.num_layers):
            nodes = self.layers[i].forward(nodes, edges, edge_idx)
        num_atoms = nodes.shape[0]
        nodes = nodes.sum(dim=0)
        nodes = nodes/num_atoms
        nodes = torch.cat([nodes, globfeats])
        out = self.finalMLP(nodes)
        return out


