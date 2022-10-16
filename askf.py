import pandas as pd
import numpy as np

SKF_LIFE_CHART = pd.DataFrame(
    np.array([[0.005, 0.1, 0.105, 0.115, 0.14, 0.16, 0.185, 0.205, 0.24, 0.26, 0.31, 0.33, 0.365, 0.38],
              [0.01, 0.1, 0.11, 0.12, 0.15, 0.19, 0.215, 0.25, 0.3, 0.35, 0.42, 0.48, 0.53, 0.57],
              [0.02, 0.1, 0.113, 0.13, 0.17, 0.22, 0.27, 0.32, 0.42, 0.48, 0.65, 0.75, 0.85, 0.95],
              [0.05, 0.1, 0.118, 0.14, 0.21, 0.3, 0.41, 0.52, 0.75, 0.9, 1.4, 1.7, 2.2, 2.5],
              [0.1, 0.1, 0.123, 0.155, 0.26, 0.42, 0.63, 0.85, 1.35, 1.9, 3.2, 4.5, 6, 7],
              [0.2, 0.1, 0.13, 0.18, 0.34, 0.62, 1.05, 1.6, 3, 4.7, 10, 16, 25, 30],
              [0.5, 0.1, 0.143, 0.22, 0.58, 1.4, 3, 5.5, 15, 35, 90, 150, 250, 450],
              [1, 0.1, 0.155, 0.27, 1, 3.2, 9.5, 23, 90, 150, 150, 150, 150, 150],
              [2, 0.1, 0.175, 0.37, 2, 11, 48, 150, 150, 150, 150, 150, 150, 150],
              [5, 0.1, 0.215, 0.6, 8, 95, 500, 500, 500, 500, 500, 500, 500, 500]
              ]),
    columns=["ncPu/P", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "1.0", "1.5", "2.0", "3.0", "4.0"])

def determine_askf(kappa, ncpup):
    """
    Calculates the SKF life modification factor a_SKF for a given kappa value and a given η_c * P_u/P for a radial ball
    bearing. See diagram 9 on page 96 of the SKF Rolling bearing catalogue for reference.
    :param kappa: The viscosity ratio - a measure of the lubrication condition of the bearing
    :param ncpup: The value of η_c * P_u/P
    :return: a_skf
    """

    ncpups = [SKF_LIFE_CHART[SKF_LIFE_CHART['ncPu/P'].gt(ncpup)].index[0] - 1,
              SKF_LIFE_CHART[SKF_LIFE_CHART['ncPu/P'].gt(ncpup)].index[0]]

    kappas = [np.argmax(pd.to_numeric(list(SKF_LIFE_CHART.columns)[1::]) > kappa),
              np.argmax(pd.to_numeric(list(SKF_LIFE_CHART.columns)[1::]) > kappa) + 1]

    inputs = [[(pd.to_numeric(list(SKF_LIFE_CHART.columns)[1::]))[kappas[0]-1],
               (pd.to_numeric(list(SKF_LIFE_CHART.columns)[1::]))[kappas[1]-1]],
              [SKF_LIFE_CHART.iloc[ncpups[0], 0], SKF_LIFE_CHART.iloc[ncpups[1], 0]]]

    vals = [[SKF_LIFE_CHART.iloc[ncpups[0], kappas[0]], SKF_LIFE_CHART.iloc[ncpups[0], kappas[1]]],
            [SKF_LIFE_CHART.iloc[ncpups[1], kappas[0]], SKF_LIFE_CHART.iloc[ncpups[1], kappas[1]]]]

    # Linearly interpolate by kappa
    vals = [linterp(inputs[0][0], inputs[0][1], vals[0][0], vals[0][1], kappa),
            linterp(inputs[0][0], inputs[0][1], vals[1][0], vals[1][1], kappa) ]

    vals = linterp(inputs[1][0], inputs[1][1], vals[0], vals[1], ncpup)
    print(f"Final value for a_skf: {vals}")


def calculate_ncpup(fatigue_limit, effective_load, contamination_factor = 0.7):
    """
    Calculates n_c * p_u / p
    :param fatigue_limit: P_u - must have same units as the effective load
    :param effective_load: P - must have same units as the fatigue limit
    :param contamination_factor: Defaults to 0.7 based on normal cleanliness for sealed bearings that are greased for life
    :return: n_c * p_u / p
    """
    return fatigue_limit*contamination_factor/effective_load

def linterp(x1, x2, y1, y2, x3):
    """
    Linearly interpolates to determine y3 for a range of values defined by [x1,y1], [x2, y2], [x3, y3 = ?]
    return: y3
    """
    m = (y2 - y1) / (x2 - x1)
    return (x3 - x1) * m + y1


