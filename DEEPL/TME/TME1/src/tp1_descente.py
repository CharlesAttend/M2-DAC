import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ctx_MSE = Context()
    ctx_linear = Context()
    ## Calcul du forward (loss)
    yhat = Linear.forward(ctx_linear, x, w, b)
    loss = MSE.forward(ctx_MSE, yhat, y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar("Loss/train", loss.mean(), n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss.mean()}")

    ##  Calcul du backward (grad_w, grad_b)
    d_yhat, d_y = MSE.backward(
        ctx_MSE, torch.zeros(50, 3) + 0.067
    )  # WTF pourquoi ça marche
    # d_yhat, d_y = MSE.backward(ctx_MSE, loss.mean())  # WTF pourquoi ça marche
    # d_yhat, d_y = MSE.backward(ctx_MSE, loss)
    _, grad_w, grad_b = Linear.backward(ctx_linear, d_yhat)

    ##  Mise à jour des paramètres du modèle
    with torch.no_grad():
        w = w - epsilon * grad_w
        b = b - epsilon * grad_b
