import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 
df = pd.read_csv('medical_examination.csv')

# 2.
bmi = df['weight'] / (df['height'] / 100)**2
df['overweight'] = (bmi > 25).astype(int)

# 3.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Draw Categorical Plot
def draw_cat_plot():
    """
    Crea un gráfico de barras categórico que muestra el recuento de características
    agrupadas por estado cardiovascular.
    """
    # 5.
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=[
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'
        ])

    # 6.
    df_cat = df_cat.groupby(['cardio', 'variable',
                             'value']).size().reset_index(name='total')

    # 7. 
    g = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar')

    # 8. 
    fig = g.fig

    # 9.
    fig.savefig('catplot.png')
    return fig


# 10. 
def draw_heat_map():
    """
    Crea un mapa de calor de correlación para visualizar la relación
    entre las diferentes variables del conjunto de datos.
    """
    # 11. 
    df_heat = df[(df['ap_lo'] <= df['ap_hi'])
                 & (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12. 
    corr = df_heat.corr()

    # 13.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. 
    fig, ax = plt.subplots(figsize=(11, 9))

    # 15. 
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        linewidths=.5,
        square=True,
        cbar_kws={"shrink": .5},
        ax=ax)

    # 16.
    fig.savefig('heatmap.png')
    return fig
