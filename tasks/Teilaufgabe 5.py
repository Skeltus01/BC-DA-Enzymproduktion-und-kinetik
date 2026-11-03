# Michaelis-Menten Kinetik eines Enzyms
# Nun soll die enzymatische Charakterisierung erfolgen. Dazu wird das Enzym mit einer variierenden Menge Substrats
# versetzt und über den Zeitraum von 10 Minuten die Bildung eines Produkts colorimetrisch verfolgt.
# Somit kann die jeweilig gebildete Menge Produkts quantifiziert werden, was in dem Tab ‚MnM‘ abgelegt ist.
# Für jede Substratkonzentration wurden drei biologisch unabhängige Replikate durchgeführt.

import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy
import scipy.stats as stats
import scipy.optimize as optimize
import math
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.multicomp as smm
import statsmodels.formula.api as smf
from matplotlib.ticker import MultipleLocator

df = pd.read_csv(
    "data/MnM.csv",
    sep=";",
    header=None,
    dtype=str,
    encoding="cp1252"
)

konz = [2, 4, 8, 16, 24, 32, 39, 47, 56, 66, 77, 90, 110]

def get(c, rep):
    i, z = konz.index(c), 5 + 10 * (rep - 1)
    out = df.iloc[z:z+6, 3*i:3*i+2].apply(lambda s: s.str.replace(",", ".").astype(float))
    out.columns = ["Time (min)", "Product (nM)"]
    return out.reset_index(drop=True)

""" Auslesen der Daten:

>>> Konzentration: 4 µM
>>> Replikat: 2
>>> get(4, 2)
"""
print(get(4, 2))


# Bearbeiten Sie nun folgende Aufgaben und Fragen:



# 5.i
# Bestimmen Sie für jede Substratkonzentration und jedes Replikat die Anfangsgeschwindigkeit v0
# der Reaktion! Auch wenn es sich hierbei um einen linearen Fit handelt, kann dennoch auch die nicht-lineare
# Regressionsanalyse eingesetzt werden.



# 5.ii
# Tragen Sie nun die soeben bestimmten Anfangsgeschwindigkeiten v0 gegen die Substratkonzentration [S]
# auf! Werten Sie die Daten gemäß der Michaelis-Menten Kinetik aus und bestimmen somit die besten Werte
# für vmax und KM sowie die zugehörigen Standardfehler! Bewerten Sie die Güte der Kurvenanpassung!




# 5.iii
# Transformieren Sie die Daten nun in die doppelt-reziproke Form nach Lineweaver-Burk!
# Tragen Sie die Daten auf und führen einen Fit von 1 / v0 gegen 1 / [S] auf.
# Unterscheiden sich die erhaltenen Werte für die Parameter vmax und KM von den zuvor bestimmten?



# 5.iv
# Falls die vorherige Frage bejaht wurde, diskutieren Sie, woran das liegen und wie dem zu begegnen sein
# könnte. Was würden Sie einem neuen Studenten als Vorgehen empfehlen, wenn sie/er ähnliche Experimente auswertet?



