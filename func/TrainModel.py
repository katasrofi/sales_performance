import torch

def TrainingModel(model,
                  X_train,
                  y_train,
                  X_val,
                  y_val,
                  metrics,
                  criterion,
                  optimizer,
                  epochs):
    metric = metrics

    # Create the loop
    for epoch in range(epochs):
        # Set model to train mode
        model.train()

        # Forward
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        # mse = MeanSquadError()
        metric.update(y_pred, y_train)

        # Backward pass and Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = metric.compute()
            print(f'Epochs: {epoch} | MSE: {acc:.4f}| Loss: {loss.item():.4f}')


    # Evaluate the model
    model.eval()
    with torch.inference_mode():
        ValPred = model(X_val)
        ValLoss = criterion(ValPred, y_val)
        # mse = MeanSquadError()
        ValMse = metric.compute()
        print(f'MSE Loss: {ValMse:.4f}')
        print(f'Validation Loss: {ValLoss:.4f}')

    metric.reset()


def ConvertToTensor(X_train,
                    X_test,
                    y_train,
                    y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
