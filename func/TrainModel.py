import torch

def TrainingModel(model,
                  X_train,
                  y_train,
                  X_val,
                  y_val,
                  criterion,
                  optimizer,
                  epochs,
                  verbose=10):

    # Create the loop
    for epoch in range(epochs):
        # Set model to train mode
        model.train()

        # Forward
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass and Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % verbose == 0:
            print(f'Epochs: {epoch} | Loss: {loss.item():.4f}')


    # Evaluate the model
    model.eval()
    with torch.inference_mode():
        ValPred = model(X_val)
        ValLoss = criterion(ValPred, y_val)
        print(f'Validation Loss: {ValLoss:.4f}')



def ConvertToTensor(X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test):

    # Dependent Variable
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Target
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor
