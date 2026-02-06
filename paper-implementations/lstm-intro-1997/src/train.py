import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

def train_model(model, dataloader, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    model.train()
    total_steps = epochs * len(dataloader)
    loop = tqdm(total=total_steps)
    for epoch in range(epochs):
        total_loss = 0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 1. Zero Gradients
            optimiser.zero_grad()

            # 2. Forward Pass
            predictions, _ = model(x_batch)

            # 3. Calculate Loss
            ## reshape predictions, and y_batch for CrossEntropyLoss
            ## predictions: (batch_size, seq_len, vocab_size):- (N, vocab_size)
            ## y_batch:     (batch_size, seq_len):- (N)
            predictions = predictions.reshape(-1, predictions.size(-1))
            y_batch = y_batch.reshape(-1) 

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

            # update progress bar description
            loop.update(1)
            loop.set_postfix(epoch=epoch+1, loss=loss.item())

        if (epoch+1) % 20 == 0:
            # !Implement Accuracy
            print(f"Epoch {epoch+1:4d} | Loss {total_loss/len(dataloader):.4f} | Acc --")

    loop.close()