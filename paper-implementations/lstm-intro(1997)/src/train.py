import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 1. Zero Gradients
            optimiser.zero_grad()

            # 2. Forward Pass
            predictions, _ = model(x_batch)

            # 3. Calculate Loss
            loss = criterion(predictions, y_batch)

            # 4. Backward Pass
            loss.backward()

            # 5. Gradient Clipping
            ## To prevent gradients from exploding,
            ## clip the global norm at 1.0
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 6. Optimiser step
            optimiser.step()

            # 7. Add to the total loss
            total_loss += loss.item()

        if epoch % 20 == 0:
            # !Implement Accuracy
            print(f"Epoch {epoch+1:4d} | Loss {total_loss/len(dataloader)} | Acc --")