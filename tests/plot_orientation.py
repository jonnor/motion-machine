
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

columns = ['time', 'acc_x', 'acc_y', 'acc_z',
           'orientation_x', 'orientation_y', 'orientation_z',
           'pitch', 'roll']

def load_imu_npy(path):
    data = np.load(path)
    return pd.DataFrame(data, columns=columns)

def plot_imu(df):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Accelerometer', 'Orientation', 'Pitch & Roll'),
                        vertical_spacing=0.08)

    for col in ['acc_x', 'acc_y', 'acc_z']:
        fig.add_trace(go.Scatter(x=df['time'], y=df[col], name=col), row=1, col=1)

    for col in ['orientation_x', 'orientation_y', 'orientation_z']:
        fig.add_trace(go.Scatter(x=df['time'], y=df[col], name=col), row=2, col=1)
    fig.update_yaxes(range=[-1.0, 1.0], row=2, col=1)

    for col in ['pitch', 'roll']:
        fig.add_trace(go.Scatter(x=df['time'], y=df[col], name=col), row=3, col=1)
    fig.update_yaxes(range=[-180, 180], row=3, col=1)

    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_layout(height=800, title='IMU Data')
    return fig


def main():
    df = load_imu_npy('test_gravity_estimator_rotations.npy')
    fig = plot_imu(df)
    fig.show()

if __name__ == '__main__':
    main()
