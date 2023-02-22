import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd

from torch_scatter import scatter_add

class RWLayer(nn.Module):
    def __init__(self, input_dim, n_graphs, n_nodes, hidden_dim, lamda):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_graphs = n_graphs
        self.lamda = lamda
        self.w_adj = nn.Parameter(torch.FloatTensor(n_graphs, (n_nodes*(n_nodes-1))//2))
        self.w_x = nn.Parameter(torch.FloatTensor(n_graphs, n_nodes, hidden_dim))
        self.b = torch.nn.Parameter(torch.tensor(0.))
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.w_adj.data.uniform_(-1, 1)
        self.w_x.data.uniform_(-1, 1)
        
    def forward(self, z, adj, x, px):
        w_adj = torch.zeros(self.n_graphs, self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        w_adj[:,idx[0],idx[1]] = self.relu(self.tanh(self.w_adj))
        w_adj = w_adj + torch.transpose(w_adj, 1, 2)
        x = self.fc(x)
        S = self.sigmoid(torch.einsum("ab,cdb->acd", (x, self.w_x))+self.b)
        Z = z.view(x.size(0), self.n_graphs, self.n_nodes)
        Z = torch.mul(Z, S)
        x = torch.einsum("abc,bdc->abd", (Z, w_adj))
        x = torch.reshape(x, (x.size(0), -1))
        x = torch.mm(adj, x)
        x = torch.mul(x.view(-1), S.view(-1))
        z = px + self.lamda*x
        return z


def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res


class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, n_graphs, n_nodes, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.n_graphs = n_graphs
        self.n_nodes = n_nodes
        self.kwargs = kwargs
        
    def forward(self, adj, x, px):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, adj, x, px), 
                torch.zeros(self.n_graphs*self.n_nodes*x.size(0), device=x.device), **self.kwargs)
        z = self.f(z, adj, x, px)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, adj, x, px)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z


class GRWNN(nn.Module):
    def __init__(self, input_dim, n_graphs, n_nodes, hidden_dim, lamda, n_classes, dropout):
        super(GRWNN, self).__init__()
        self.n_nodes = n_nodes
        self.n_graphs = n_graphs
        f = RWLayer(input_dim, n_graphs, n_nodes, hidden_dim, lamda)
        self.deq = DEQFixedPoint(f, forward_iteration, n_graphs, n_nodes, tol=1e-4, max_iter=50)
        self.ln = nn.LayerNorm(n_graphs)
        self.fc1 = nn.Linear(n_graphs, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        edge_index, x = data.edge_index, data.x
        v = torch.ones(edge_index.size(1), device=x.device)
        adj = torch.sparse.FloatTensor(edge_index, v, torch.Size([x.size(0),x.size(0)])).to(x.device)

        qx = torch.ones(x.size(0), self.n_graphs, self.n_nodes, device=x.device)
        px = qx

        z = self.deq(adj, x, px.view(-1))
        Z = z.view(adj.size(1), self.n_graphs, self.n_nodes)
        Z = torch.mul(Z, qx)
        out = torch.sum(Z, dim=2)
        out = scatter_add(out, data.batch, dim=0)
        out = self.ln(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
