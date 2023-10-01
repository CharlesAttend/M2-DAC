import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))

# Test du gradient de Linear (sur le même modèle que MSE)
n = 10  # Number of exemple
d = 5  # Input dimension
p = 1  # Output dimension
X = torch.randn(n, d, requires_grad=True, dtype=torch.float64)
W = torch.randn(d, p, requires_grad=True, dtype=torch.float64)
b = torch.randn(p, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(linear, (X, W, b))
