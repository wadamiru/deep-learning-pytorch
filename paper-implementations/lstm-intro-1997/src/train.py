import torch
import torch.nn as nn

from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimiser, criterion, early_stopper, device, epochs=100):
    print("device:", device)
    model.to(device)

    total_steps = epochs * len(train_loader)
    loop = tqdm(total=total_steps)

    for epoch in range(epochs):
        # ----- Training -----#
        model.train()
        train_loss = 0

        for x_batch, y_batch in train_loader:
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

            # 7. Add to the total train loss
            train_loss += loss.item()

            # update training progress bar description
            loop.update(1)
            loop.set_postfix(epoch=epoch+1, loss=loss.item())

        # Calculate average train loss
        train_loss = train_loss/len(train_loader)

        # ----- Validation -----#
        ## Repeat the same general process without gradient calculations
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions, _ = model(x_batch)
                predictions = predictions.reshape(-1, predictions.size(-1))
                y_batch = y_batch.reshape(-1)

                # Calculate validation loss
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
            
            # Calculate average validation loss
            val_loss = val_loss/len(val_loader)

        if (epoch+1) % 20 == 0:
            # !Implement Accuracy
            print(f"Epoch {epoch+1:4d} | Train Loss {train_loss:.4f} | Acc --")

        # Early stopping
        if early_stopper.step(val_loss):
            print("Early Stopping Triggered")
            break

    loop.close()