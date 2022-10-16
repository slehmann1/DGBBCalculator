# Author: Sam Lehmann
# Date: 2022-10-16
# Description: Deep groove ball bearing calculator - Determines frontiers under which a bearing may operate.
# These frontiers are determined based on static factor of safety and the factor of safety for operating lifetime.
# Impacts of lubrication not considered. Only examines deep groove ball bearings and is based on a combination of
# ISO 281, ISO 76, and the methodology outlined in the SKF rolling bearings catalogue. A large range of assumptions
# apply; these should be rigorously understood along with theory behind bearing life prior to usage.

import math
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_POINTS = 1000  # Number of datapoints determined per bearing

STATIC_FOS = 1
LIFE_FOS = 1
LIFE_ADJ_FACTOR = 0.64  # Converts from 90% life to 95% life: Note 0.64 = 95

# Calculation factors for deep groove ball bearings. Taken from table 9 of SKF rolling bearing catalogue (page 257).
DGBB_CALC_FACTORS = pd.DataFrame(
    np.array([[0.172, 0.19, 0.56, 2.3],
              [0.345, 0.22, 0.56, 1.99],
              [0.689, 0.26, 0.56, 1.71],
              [1.03, 0.28, 0.56, 1.55],
              [1.38, 0.3, 0.56, 1.45],
              [2.07, 0.34, 0.56, 1.31],
              [3.45, 0.38, 0.56, 1.15],
              [5.17, 0.42, 0.56, 1.04],
              [6.89, 0.44, 0.56, 1]
              ]), columns=["f_0F_a/C_0", "e", "X", "Y"])

# Limits rated loads to P <= INDETERMINATE_LOAD_LIMITATION *C
# For indeterminate loads under long term operating conditions, SKF recommends limiting P to 0.1/C, based on fretting
# of the bearing seat
INDETERMINATE_LOAD_LIMITATION = 0.1

# Features of a bearing
bearing_param_tuple = namedtuple("bearing_tuple", "dynamic_rating static_rating static_radial_load_factor "
                                                  "static_axial_load_factor life_eqn_exponent name "
                                                  "calculation_factor")
# Features of a load case applied to a bearing
load_param_tuple = namedtuple("load_case_tuple", "f_a f_r name")

# Features of a shaft/bearing setup
# x_2, x_1: measured in m, see FBD for reference
# bearing preload: N
# effective speed: RPM
# L10H: number of hours at the corresponding life adjustment factor
shaft_param_tuple = namedtuple("shaft_tuple", "x_2 x_1 effective_speed l10h name")


def get_bearing_loads(load_case, name, shaft_case, axial_preload):
    """
    Determines the loads applied to both bearings
    :param load_case: A pandas dataframe with columns defined as "f_r_shaft", "f_a_shaft", "m_shaft", "duty_cycle". All
    units are [N]
    :param name:The name of the load case
    :param shaft_case:A named tuple defining the shaft parameters
    :return: An array [load_case_2, load_case_1] where both elements are load_param_tuple
    """

    # Determine weighted averages for loads based on duty cycles
    f_a_shaft = sum(load_case.f_a_shaft * load_case.duty_cycle)
    f_r_shaft = sum(load_case.f_r_shaft * load_case.duty_cycle)
    m_shaft = sum((load_case.m_shaft) * load_case.duty_cycle)

    # Define that bearing 2 takes the axial load (bearing 2 is locating)
    f_a_2 = f_a_shaft + axial_preload
    f_a_1 = axial_preload

    # Determine radial loads. Reference FBD.
    f_r_2 = (shaft_case.x_1 * f_r_shaft + m_shaft) / (shaft_case.x_2 - shaft_case.x_1)
    f_r_1 = -f_r_shaft - f_r_2

    load_case_2 = load_param_tuple(abs(f_a_2), abs(f_r_2), name=f"{name} (Bearing 2)")
    load_case_1 = load_param_tuple(abs(f_a_1), abs(f_r_1), name=f"{name} (Bearing 1)")

    return [load_case_2, load_case_1]


def determine_bearing_rated_curve(bearing_parameters, shaft_parameters, max_axial=3000):
    """
    Creates a boundary curve of [radial loads, axial loads] for a bearing under the given load case
    :param bearing_parameters: The bearing parameters
    :param shaft_parameters: The shaft parameters
    :param max_axial: The max axial load to be examined [N]
    :return: An array of [radial_loads,axial_loads] in [N]
    """

    l10 = get_l10(shaft_parameters.l10h, shaft_parameters.effective_speed)
    axial_data = np.arange(0, max_axial, (max_axial / DATA_POINTS))
    radial_data = np.zeros(len(axial_data))

    for i in range(0, len(axial_data)):
        radial_data[i] = min(get_static_radial_capacity(bearing_parameters, axial_data[i]),
                             get_dynamic_radial_capacity(bearing_parameters, axial_data[i], l10))
    return [radial_data, axial_data]


def get_dynamic_radial_capacity(bearing_params, f_a, l10):
    """
    Determines radial capacity based on equivalent dynamic load for a given axial load based on the basic life
    equation and specified life requirements
    :param l10: Basic life rating units are [10^6 rotations]
    :param bearing_params: A named bearing tuple
    :param f_a: The axial force applied (N)
    :return: The radial force (N)
    """
    # f_r: radial load
    # f_a: axial load
    # p: equivalent dynamic load
    # x: dynamic radial load factor for bearing
    # y: dynamic axial load factor for bearing

    # Determine equivalent dynamic load based on the basic life equation
    max_p = math.e ** (math.log(bearing_params.dynamic_rating) - math.log(l10) / bearing_params.life_eqn_exponent)

    if max_p / INDETERMINATE_LOAD_LIMITATION > bearing_params.static_rating:
        max_p = INDETERMINATE_LOAD_LIMITATION * bearing_params.static_rating

    # determine x and y
    [_, x, y] = get_calculation_factors(bearing_params.calculation_factor, f_a, bearing_params.static_rating)

    f_r = (max_p - y * f_a) / x

    return max(f_r, 0)


def get_static_radial_capacity(bearing_params, f_a, static_fos=STATIC_FOS):
    """
    Determines radial capacity based on static load rating for a given axial load based on the STATIC_FOS
    :param bearing_params: A named bearing tuple
    :param f_a: The axial force applied (N)
    :param static_fos: The static factor of safety to be applied
    :return: The radial force (N)
    """
    # f_r: radial load
    # f_a: axial load
    # p_0: equivalent static load
    # x_0: radial load factor for bearing
    # y_0: axial load factor for bearing

    # Set equivalent static load based on the bearing rating and the static fos
    max_p_0 = bearing_params.static_rating / static_fos

    f_r = (max_p_0 - bearing_params.static_axial_load_factor * f_a) / bearing_params.static_radial_load_factor

    if f_r > max_p_0:
        f_r = max_p_0

    return max(f_r, 0)


def get_l10(l10h, effective_speed):
    """
    Determines the L10 for a given effective speed and operating lifetime at the reliability set by the LIFE_ADJ_FACTOR
    :param l10h: The operating lifetime in hours at the reliability set by the LIFE_ADJ_FACTOR
    :param effective_speed: The effective speed of the bearing [rpm]
    :return: Basic life rating (L10) units are [10^6 rotations]
    """
    # Hours- How long the bearing lasts at 90% reliability
    l10h_90 = l10h * LIFE_FOS / LIFE_ADJ_FACTOR
    l10 = l10h_90 * 60 * effective_speed / 10 ** 6

    return l10


def get_calculation_factors(f_0, F_a, C_O, bearing_type="DGBB", clearance="Normal", arrangement="single"):
    """
    Returns e, X, and Y calculation factors for a deep groove ball bearing
    :param f_0: Calculation factor
    :param F_a: Axial load (kN)
    :param C_O: Basic static load rating (kN)
    :param arrangement: Bearing arrangement. Only single supported.
    :param clearance: Bearing clearance. Only normal clearance supported.
    :param bearing_type: Bearing type. Only deep groove ball bearings supported.
    :return: [e, X, Y] = [ limit for the load ratio, radial load calculation factor, axial load calculation factor]
    """

    # Exception handling
    if clearance != "Normal":
        raise Exception("Clearances other than normal not currently supported")
    if bearing_type != "DGBB":
        raise Exception("Only deep groove ball bearings supported")
    if arrangement != "single":
        raise Exception("Only single bearings supported")

    x3 = f_0 * F_a / C_O
    index = 0

    # Return first entry
    if x3 <= min(DGBB_CALC_FACTORS.loc[:, "f_0F_a/C_0"]):
        return [DGBB_CALC_FACTORS.loc[0, "e"], DGBB_CALC_FACTORS.loc[0, "X"], DGBB_CALC_FACTORS.loc[0, "Y"]]

    # Return last entry
    if x3 >= max(DGBB_CALC_FACTORS.loc[:, "f_0F_a/C_0"]):
        return [DGBB_CALC_FACTORS.loc[:, "e"].iloc[-1], DGBB_CALC_FACTORS.loc[:, "X"].iloc[-1],
                DGBB_CALC_FACTORS.loc[:, "Y"].iloc[-1]]

    # Not first or last, linear interpolation required
    for i in range(0, len(DGBB_CALC_FACTORS.loc[:, "f_0F_a/C_0"])):
        if DGBB_CALC_FACTORS.loc[i, "f_0F_a/C_0"] > x3:
            index = i - 1
            break

    x1 = DGBB_CALC_FACTORS.loc[index, "f_0F_a/C_0"]
    x2 = DGBB_CALC_FACTORS.loc[index + 1, "f_0F_a/C_0"]

    e = lin_interp(x1, x2, x3, DGBB_CALC_FACTORS.loc[index, "e"], DGBB_CALC_FACTORS.loc[index + 1, "e"])
    X = lin_interp(x1, x2, x3, DGBB_CALC_FACTORS.loc[index, "X"], DGBB_CALC_FACTORS.loc[index + 1, "X"])
    Y = lin_interp(x1, x2, x3, DGBB_CALC_FACTORS.loc[index, "Y"], DGBB_CALC_FACTORS.loc[index + 1, "Y"])

    return [e, X, Y]


def plot_rated_curve(bearing_frontiers, load_cases, title):
    """
    Plots a bearing frontier with specific load cases overlaid
    :param bearing_frontiers: A list of bearing frontiers: x,y data defining the bearing SOAC
    :param load_cases: An array of load_param_tuple to be overlaid on the bearing frontier.
    :param title: The graph title
    :return: None
    """

    # Plot bearing frontiers
    for bearing_frontier in bearing_frontiers:
        plt.plot(bearing_frontier.radial_data, bearing_frontier.axial_data, label=bearing_frontier.bearing_params.name)

    # Plot load cases
    for load_case in load_cases:
        for bearing_case in load_case:
            plt.scatter(bearing_case.f_r, bearing_case.f_a, label=bearing_case.name)

    # Graph formatting
    plt.title(title)
    plt.xlabel("Bearing Radial Load (N)")
    plt.ylabel("Bearing Axial Load (N)")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    plt.show()


def lin_interp(x1, x2, x3, y1, y2):
    """
    For points x1,x2,y1,y2 linearly interpolates a value of y for a value of x: x3 between x1 and x2
    """

    return y1 + (y2 - y1) * (x3 - x1) / (x2 - x1)


class BearingFrontier:
    """
    Container of axial_data [N], radial_data [N], and bearing_params
    """

    def __init__(self, bearing_params, axial_data=None, radial_data=None):
        """

        :param bearing_params: A bearing_param_tuple
        :param axial_data: An array of floats
        :param radial_data: An array of floats
        """
        self.bearing_params = bearing_params
        self.axial_data = axial_data
        self.radial_data = radial_data


# Currently setup to illustrate a sample usecase for this tool
def sample_use_case():
    # Define Bearing Parameters of Interest
    params_61806 = bearing_param_tuple(dynamic_rating=4490, static_rating=2900, static_radial_load_factor=0.6,
                                       static_axial_load_factor=0.5, life_eqn_exponent=3, name="61806 Bearing",
                                       calculation_factor=14)
    params_61809 = bearing_param_tuple(dynamic_rating=6630, static_rating=6100, static_radial_load_factor=0.6,
                                       static_axial_load_factor=0.5, life_eqn_exponent=3, name="61809 Bearing",
                                       calculation_factor=17)
    params_61909 = bearing_param_tuple(dynamic_rating=14000, static_rating=10800, static_radial_load_factor=0.6,
                                       static_axial_load_factor=0.5, life_eqn_exponent=3, name="61909 Bearing",
                                       calculation_factor=16)
    params_61814 = bearing_param_tuple(dynamic_rating=12400, static_rating=13200, static_radial_load_factor=0.6,
                                       static_axial_load_factor=0.5, life_eqn_exponent=3, name="61814 Bearing",
                                       calculation_factor=17)
    params_61914 = bearing_param_tuple(dynamic_rating=23800, static_rating=18300, static_radial_load_factor=0.6,
                                       static_axial_load_factor=0.5, life_eqn_exponent=3, name="61914 Bearing",
                                       calculation_factor=14)

    # Define shaft setup of interest
    shaft_setup = shaft_param_tuple(x_2=0.0175, x_1=0.153, effective_speed=250, l10h=20000,
                                    name="Sample Shaft Case")

    print(f"Static FOS: {STATIC_FOS}, Life FOS: {LIFE_FOS}, With Life Adjustment Factor Of {LIFE_ADJ_FACTOR}, "
          f"Target Hour Lifetime: {shaft_setup.l10h}, Effective Speed {shaft_setup.effective_speed}")

    # Generate bearing frontiers for bearings of interest
    skf_61806 = BearingFrontier(params_61806)
    [skf_61806.radial_data, skf_61806.axial_data] = determine_bearing_rated_curve(params_61806, shaft_setup,
                                                                                  max_axial=2500)
    skf_61809 = BearingFrontier(params_61809)
    [skf_61809.radial_data, skf_61809.axial_data] = determine_bearing_rated_curve(params_61809, shaft_setup,
                                                                                  max_axial=2500)
    skf_61909 = BearingFrontier(params_61909)
    [skf_61909.radial_data, skf_61909.axial_data] = determine_bearing_rated_curve(params_61909, shaft_setup,
                                                                                  max_axial=2500)
    skf_61814 = BearingFrontier(params_61814)
    [skf_61814.radial_data, skf_61814.axial_data] = determine_bearing_rated_curve(params_61814, shaft_setup,
                                                                                  max_axial=2500)
    skf_61914 = BearingFrontier(params_61914)
    [skf_61914.radial_data, skf_61914.axial_data] = determine_bearing_rated_curve(params_61914, shaft_setup,
                                                                                  max_axial=2500)

    # Define loadcases of interest
    load_case_1 = pd.DataFrame(
        np.array([[448.15, 42.22, 56.46, 1.0]]), columns=["f_r_shaft", "f_a_shaft", "m_shaft", "duty_cycle"])
    load_case_2 = pd.DataFrame(
        np.array([[341.62, 43.04, 32.19, 1.0]]), columns=["f_r_shaft", "f_a_shaft", "m_shaft", "duty_cycle"])
    load_case_3 = pd.DataFrame(
        np.array([[261.86, 32.99, 24.67, 1]
                  ]), columns=["f_r_shaft", "f_a_shaft", "m_shaft", "duty_cycle"])
    load_case_4 = pd.DataFrame(
        np.array([[163.32, 24.05, 15.39, 1],
                  ]), columns=["f_r_shaft", "f_a_shaft", "m_shaft", "duty_cycle"])
    load_case_5 = pd.DataFrame(
        np.array([[118.75, 17.78, 11.19, 0.5],
                  [220.5, 30.8, 11.5, 0.3],
                  [430.25, 50.7, 50.19, 0.2]
                  ]), columns=["f_r_shaft", "f_a_shaft", "m_shaft", "duty_cycle"])

    # Compute loads applied to the bearings
    bearing_loads = [
        get_bearing_loads(load_case_1, "Load Case as per Req 4.7A", shaft_setup, axial_preload=350),
        get_bearing_loads(load_case_2, "Load Case as per Req 4.7B", shaft_setup, axial_preload=350),
        get_bearing_loads(load_case_3, "Load Case as per Req 4.8A", shaft_setup, axial_preload=225),
        get_bearing_loads(load_case_4, "Load Case as per Req 4.8B", shaft_setup, axial_preload=225),
        get_bearing_loads(load_case_5, "Load Case as per Req 5.2", shaft_setup, axial_preload=150)]

    # Generate the plot
    plot_rated_curve([skf_61806, skf_61809, skf_61909, skf_61814, skf_61914], bearing_loads,
                     "Sample Bearing And Shaft Load Case at 250RPM")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sample_use_case()
