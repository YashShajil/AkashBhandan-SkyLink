"""MOCK ORACLE - NOT COURSE DATA.
Stand-in for the TA's high-fidelity rotor code so the M3 pipeline can be built and tested
before real results arrive. It has the features a real FM map tends to have: FM rises with disk
loading, a small-rotor (low-Reynolds) penalty, a sharp drop at very high blade loading (stall),
and measurement noise. Replace by putting the TA's replies in oracle/response_<k>.csv.
"""
import numpy as np

SIGMA = 0.004


def fm_true(D, T):
    DL = T / (np.pi * D ** 2 / 4)                        # disk loading, N/m^2
    fm = 0.80 - 0.16 * np.exp(-DL / 70.0) - 0.035 * (0.6 / D) ** 0.7
    stall = 0.09 * 0.5 * (1 + np.tanh((DL - 230.0) / 12.0))
    return fm - stall


def query(D, T, seed=0):
    rng = np.random.default_rng(seed + int(1e6 * np.sum(D)) % 99991)
    return fm_true(np.asarray(D), np.asarray(T)) + rng.normal(0, SIGMA, np.shape(D))
