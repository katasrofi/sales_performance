from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, root_mean_squared_error

def PipeModel(models,
              X_train,
              y_train,
              X_val,
              y_val):
    # Loop the list
    trained_pipelines = {}
    for model in models:
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_train)
        y_val_pred = pipe.predict(X_val)
        # Training loss
        train_mse = mean_squared_error(y_train, y_pred)
        train_rmse = root_mean_squared_error(y_train, y_pred)

        # Validation loss
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)

        # Print{
        print(f'{model.__class__.__name__} Training MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f}')
        print(f'{model.__class__.__name__} Validation MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f}')

        # Return the pipeline
        trained_pipelines[model.__class__.__name__] = pipe

    return trained_pipelines
