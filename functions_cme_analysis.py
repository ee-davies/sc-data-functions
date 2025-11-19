import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import integrate

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

"""
FUNCTIONS TO ANALYSE DIFFERENT PARTS OF ICMES
"""

"""
GENERAL
"""

def plotly_mag(df, save_fig=False):
    pio.renderers.default = 'browser'
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for column, color in zip(['bt', 'bx', 'by', 'bz'], ['black', 'red', 'green', 'blue']):
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df[column],
                name=column.upper(),
                line_color=color
            ),
            row=1, col=1
        )
    fig.show()
    if save_fig == True:
        now = datetime.now().strftime('%Y%m%d%H%M')
        fig.write_html(f'plotly_mag_{now}.html') 


"""
SHOCK
"""

"""
SHEATH
"""

"""
MAGNETIC EJECTA
"""

def get_DiP(df, mo_start, mo_end):
    mask = (df.time >= mo_start) & (df.time <= mo_end)
    ME = df[mask]
    times = [(time - np.min(ME['time'])).total_seconds() for time in ME['time']]
    B_int = integrate.cumtrapz(ME['bt'], times)
    #loop to find halfway point
    i = 0
    while B_int[i] < B_int[-1] / 2:
        i += 1    
    DiP = times[i]/times[-1]
    return DiP


def get_gexp_power(r1,r2,b1,b2):
    power = (np.log(b2)-np.log(b1))/(np.log(r2)-np.log(r1))
    return power