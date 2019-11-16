import torch

__all__ = ['ResidualFeedForwardNet']

class ResidualFeedFowardBlock(torch.nn.Module):
    '''Block of two feed-forward layer with a reisdual connection:
      
            f(W1^T x + b1)         f(W2^T h1 + b2 )         h2 + x 
        x ------------------> h1 --------------------> h2 ----------> y
        |                                              ^
        |               Residual connection            | 
        +----------------------------------------------+
        
    '''
    
    def __init__(self, dim_in, width, activation_fn=torch.nn.Tanh):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim_in, width)
        self.layer2 = torch.nn.Linear(width, dim_in)
        self.activation_fn = activation_fn()
    
    def forward(self, x):
        h1 = self.activation_fn(self.layer1(x))
        h2 = self.activation_fn(self.layer2(h1))
        return h2 + x
    

class ResidualFeedForwardNet(torch.nn.Module):
    
    def __init__(self, dim_in, nblocks=1, block_width=10):
        super().__init__()
        self._dim_in = dim_in
        self.blocks = torch.nn.Sequential(*[
            ResidualFeedFowardBlock(dim_in, block_width)
            for i in range(nblocks)
        ])
    
    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        # input and output dimension are the same in our residual network.
        return self._dim_in
    
    def forward(self, X):
        return self.blocks(X)