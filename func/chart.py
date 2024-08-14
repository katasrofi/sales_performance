import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import seaborn as sns
import os
import numpy as np
import warnings

def PieBarChart(data: pd.DataFrame,
             DateParameter,
             NumericParameter,
             TitleSize=20,
             PieTitle=None,
             BarTitle=None,
             path=None,
             name_file=None,
             save=False,
             figsizes=(14,6),
             PieSubplot=(1, 2, 1),
             BarSubplot=(1, 2, 2),
             explode=None,
             colors=['#00BFFF',  # Blue Light
                     '#32CD32',  # Green Light
                     '#FFA500']  # Orange Light
             ):
                 warnings.filterwarnings('ignore')
                 # Convert the data into list
                 df = data.groupby(DateParameter)[NumericParameter].sum().reset_index()
                 df_date = df[DateParameter].tolist()
                 df_numeric = df[NumericParameter].tolist()

                 # Configure the Area Chart
                 plt.figure(figsize=figsizes)
                 plt.subplot(*PieSubplot)

                 # Configure explode
                 if explode is None:
                     explode = [0] * len(df_numeric)
                 elif len(explode) != len(df_numeric):
                     if len(explode) > len(df_numeric):
                         explode = explode[:len(df_numeric)]
                     else:
                         explode.extend([0] * (len(df_numeric) - len(explode)))

                 # Create the chart
                 plt.pie(df_numeric,
                         labels=df_date,
                         autopct='%2.1f%%',
                         explode=explode,
                         colors=colors)

                 # Configure the labels
                 if PieTitle:
                     plt.title(PieTitle, fontsize=TitleSize)
                 else:
                     plt.title(NumericParameter, fontsize=TitleSize)

                 # Save the Chart
                 if save and path and name_file:
                     save_path = os.path.join(path, name_file + 'Pie.png')
                     plt.savefig(save_path)

                 # Configure BarChart
                 plt.subplot(*BarSubplot)
                 ax = sns.barplot(data=data,
                                  x=DateParameter,
                                  y=NumericParameter,
                                  palette=colors)

                 for i in ax.containers:
                     ax.bar_label(i)

                 # Configure the labels
                 if BarTitle:
                     plt.title(BarTitle, fontsize=TitleSize)
                 else:
                     plt.title(NumericParameter, fontsize=TitleSize)

                 if save and path and name_file:
                     save_path = os.path.join(path, name_file + 'Bar.png')
                     plt.savefig(save_path)

                 plt.show()



def LineChart(data: pd.DataFrame,
              DateParameter,
              NumericParameter,
              LineTitle,
              index=None,
              xlabel=None,
              ylabel=None,
              fontsize=20,
              labelsize=16,
              figsize=(16, 6),
              save=False,
              path=None,
              name_file=None):
    if index:
        data.reset_index(drop=True, inplace=True)
        data[index] = pd.to_datetime(data[index])
        data.set_index(index, inplace=True)

    data.resample(DateParameter)[NumericParameter].sum().plot(kind='line', figsize=figsize, legend=None)

    if LineTitle:
        plt.title(LineTitle + ' 2010 - 2012', fontsize=fontsize)
    else:
        plt.title(NumericParameter)

    if xlabel and ylabel:
        plt.xlabel(xlabel, fontsize=labelsize)
        plt.ylabel(ylabel, fontsize=labelsize)

    if save and path and name_file:
        save_path = os.path.join(path, name_file + 'Line.png')
        plt.savefig(save_path)

    plt.show()



def MlChart(model,
            target,
            prediction,
            title,
            X_train=None,
            y_train=None,
            ComparisonPlot=False,
            ResidualPlot=False,
            DistributionPlot=False,
            LearningCurve=False):

    from sklearn.pipeline import make_pipeline
    # Data Prediction
    y_pred = model.predict(prediction)

    # Comparison between Actual value and Prediction
    if ComparisonPlot:
        plt.figure(figsize=(16, 6))
        plt.scatter(target, y_pred)
        plt.title(f'{title.__class__.__name__} Actual data Vs Prediction', fontsize=20)
        plt.xlabel('Actual data', fontsize=16)
        plt.ylabel('Prediction', fontsize=16)
        plt.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=2)
        plt.show()

    # Residual plot
    if ResidualPlot:
        residu = target - y_pred
        plt.figure(figsize=(16, 6))
        plt.scatter(y_pred, residu)
        plt.title('Residual plot', fontsize=20)
        plt.xlabel('Prediction', fontsize=16)
        plt.ylabel('Residual', fontsize=16)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()

    # Residual Distribution
    if DistributionPlot:
        plt.figure(figsize=(16, 6))
        sns.histplot(residu, kde=True, legend=None)
        plt.title('Residual Distributions', fontsize=20)
        plt.xlabel('Residual', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.show()

    # Learning Curve
    if LearningCurve and X_train is not None and y_train is not None:
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring='r2',
            train_sizes=np.linspace(0.1, 1.0, 10)
            )

        train_scores_mean = train_scores.mean(axis=1)
        val_scores_mean = val_scores.mean(axis=1)

        plt.figure(figsize=(16, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation score')
        plt.title(f'{title.__class__.__name__} Learning Curve', fontsize=20)
        plt.xlabel('Training Size', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        plt.legend(loc='best')
        plt.show()

