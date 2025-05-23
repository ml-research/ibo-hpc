import torch.optim as optim
import torch.distributions as dists

def train_mlp(mlp, x, y, gpu, epochs=10000):

    optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    for epoch in range(epochs):
        mlp.train()
        optimizer.zero_grad()
        
        # full data forward pass: Get predicted k and theta from the model
        pred = mlp(x.to(gpu))
        prediction = pred
        k_pred, theta_pred = prediction[:, 0], prediction[:, 1]
        
        # Compute the negative log-likelihood loss for Gamma distribution
        gamma_dist = dists.Gamma(k_pred, theta_pred)
        log_likelihood = gamma_dist.log_prob(y.to(gpu))
        loss = -log_likelihood.mean()  # Minimize negative log-likelihood

        # Backpropagation and optimization
        loss.backward()

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}/{epochs} \t Loss: {loss.item()}")

        optimizer.step()

    return mlp