import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import seaborn as sns
import os
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
