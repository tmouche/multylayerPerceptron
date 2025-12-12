
import pandas
import numpy as np
import plotly.graph_objects as go
import plotly.express as px 
import plotly.figure_factory as ff


def pair_plot(path_to_dataset:str):
    df = pandas.read_csv(path_to_dataset, header=None)
    df = df.iloc[:, 1:]

    print(df)
    # exit(1)

    # Création du pairplot
    fig = ff.create_scatterplotmatrix(
        df,
        index=1,
        diag="histogram",
        height=1536,
        width=2048,
        title="Relation between Features",
        colormap_type="cat"
    )

    # Mise à jour des traces pour afficher des histogrammes sur la diagonale

    fig.write_html("plots/pairplot.html", auto_open=True)


if __name__ == "__main__":
    pair_plot("data.csv")