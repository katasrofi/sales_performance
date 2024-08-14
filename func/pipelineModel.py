from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def PipeModel(models,
              X_train,
              y_train,
              X_val,
              y_val):
    # Loop the list
    for model in models:
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_train, y_train)
        score = pipe.score(X_val, y_val)
        print(f'{model.__class__.__name__} score: {score*100:.4f}%')
