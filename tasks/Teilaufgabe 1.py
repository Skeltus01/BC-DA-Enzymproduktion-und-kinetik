# Heterologe Produktion eines Enzyms in verschiedenen Nährmedien
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
import math
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Einlesen der Daten aus der CSV (aus XLSX konvertiert)
df = pd.read_csv(
    "data/media.csv",
    sep=";",
    header=3,
    decimal=","
)

lb_values = df["LB"].dropna()
tb_values = df["TB"].dropna()

# 1.i
# Stellen Sie die Ausbeuten für LB und TB in geeigneter Weise grafisch dar!
# Bestimmen Sie für beide Datensätze Mittelwert, Varianz, Median und die Quartile Q1 und Q3!

# Ausbeuten
###

# Werte der Datensätze
lb_mean = lb_values.mean()
lb_var = lb_values.var()
lb_median = lb_values.median()
lb_q1 = lb_values.quantile(0.25)
lb_q3 = lb_values.quantile(0.75)
print(
    "Mittelwert LB:", lb_mean,
    "Varianz LB:", lb_var,
    "Median LB:", lb_median,
    "Quartile LB: (Q1/Q3)", lb_q1, lb_q3
)

tb_mean = tb_values.mean()
tb_var = tb_values.var()
tb_median = tb_values.median()
tb_q1 = tb_values.quantile(0.25)
tb_q3 = tb_values.quantile(0.75)
print(
    "Mittelwert TB:", tb_mean,
    "Varianz TB:", tb_var,
    "Median TB:", tb_median,
    "Quartile TB: (Q1/Q3)", tb_q1, tb_q3
)


# 1.ii
# Bestimmen Sie für beide Medien 95%-Konfidenzintervalle, innerhalb deren sich der wirkliche Mittelwert der Ausbeute befindet!
alpha = 0.05                    #α
confidence = 1 - alpha          #1-α  -Konfidenzniveau

###### LB ######
# Stichprobengröße und Standardfehler (SEM)
lb_n = len(lb_values)           #N          -Stichprobengröße
lb_df = lb_n - 1                #ν (N-1)    -Freiheitsgrade
lb_s = lb_values.std(ddof=1)    #s          -Standardabweichung
lb_sem = lb_s / np.sqrt(lb_n)   #s_x̄        -Standardfehler des Mittelwertes (SEM)

# Quantile und Grenzen des Konfidenzintervalls
lb_q = stats.t.ppf(1-alpha/2, df=lb_df)

g0 = lb_mean - lb_q * lb_sem
g1 = lb_mean + lb_q * lb_sem

print(f"Quantil t_(v,1-α/2): {lb_q:.4f}")
print(f"95%-Konfidenzintervall (LB): [{g0:.4f}, {g1:.4f}];")


###### TB ######
# Stichprobengröße und Standardfehler (SEM)
tb_n = len(tb_values)           #N          -Stichprobengröße
tb_df = tb_n - 1                #ν (N-1)    -Freiheitsgrade
tb_s = tb_values.std(ddof=1)    #s          -Standardabweichung
tb_sem = tb_s / np.sqrt(tb_n)   #s_x̄        -Standardfehler des Mittelwertes (SEM)

# Quantile und Grenzen des Konfidenzintervalls
tb_q = stats.t.ppf(1-alpha/2, df=tb_df)

g0 = tb_mean - tb_q * tb_sem
g1 = tb_mean + tb_q * tb_sem

print(f"Quantil t_(v,1-α/2) (TB): {tb_q:.4f}")
print(f"95%-Konfidenzintervall (TB): [{g0:.4f}, {g1:.4f}];")



def Konfidenzintervall(x:pd.Series):
    x_mean = x.mean()
    x_var = x.var()
    x_median = x.median()
    x_q1 = x.quantile(0.25)
    x_q3 = x.quantile(0.75)
    print(
        f"Mittelwert {str(x.name)}:", x_mean,
        f"Varianz {str(x.name)}:", x_var,
        f"Median {str(x.name)}:", x_median,
        f"Quartile {str(x.name)}: (Q1/Q3)", x_q1, x_q3
    )

    alpha = 0.05
    confidence = 1 - alpha

    x_n = len(x)
    x_df = x_n - 1
    x_s = x.std(ddof=1)
    x_sem = x_s / np.sqrt(x_n)

    # Quantile und Grenzen des Konfidenzintervalls
    x_q = stats.t.ppf(1 - alpha / 2, df=x_df)

    g0 = x_mean - x_q * x_sem
    g1 = x_mean + x_q * x_sem

    print(f"Quantil t_(v,1-α/2): {x_q:.4f}")
    print(f"95%-Konfidenzintervall (LB): [{g0:.4f}, {g1:.4f}];")
    return x

Konfidenzintervall(lb_values)

# 1.iii
# In der Literatur ist für die Produktion des gleichen Enzymes in LB-Medium eine Ausbeute von 3 mg pro Liter Expressionskultur beschrieben.
# Testen Sie auf einem Konfidenzniveau von α = 0.05 die einseitige Hypothese, dass die nun erzielte Ausbeute bei Verwendung von LB-Medium größer sei!
# Führen Sie im Anschluss den gleichen Test für TB-Medium durch!
# Welche Voraussetzungen müssen für die Tests gegeben sein und wie können Sie das ggf. prüfen?





# 1.iv
# Vergleichen Sie nun die für LB und TB erzielten Ausbeuten! Testen Sie auf
# Konfidenzniveau α = 0.01 die zweiseitige Hypothese, dass sich die Ausbeuten
# unterscheiden! Auch hier sollte vorab eine Betrachtung der Voraussetzungen erfolgen.