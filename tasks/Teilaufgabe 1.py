# Heterologe Produktion eines Enzyms in verschiedenen Nährmedien
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
import math
import pandas as pd


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
confidence = 0.95
alpha = 1 - confidence


# Stichprobengröße und Standardfehler (SEM)
lb_n = len(lb_values)-1
lb_sem = stats.sem(lb_values)  # oder lb_values.std(ddof=1) / np.sqrt(n_lb)

ci_lb = stats.t.interval(
    confidence=confidence,
    df=lb_n,
    loc=lb_mean,
    scale=lb_sem)
conf1 = stats.norm.ppf(alpha/2)
ci_lb2 = lb_mean + np.array([-1, 1]) * conf1 * lb_sem
#conf2 = stats.t.ppf(alpha/2, df=v)
print("95%-Konfidenzintervall (LB): ", ci_lb)
print("95%-Konfidenzintervall (LB): ", ci_lb2)

alpha = 0.95
v = len(tb_values)-1
conf1 = stats.norm.ppf(alpha/2)
conf2 = stats.t.ppf(alpha/2, df=v)
#print(conf1, conf2)


###
lb_v = lb_values - 1
lb_s = lb_values.std(ddof=1) / np.sqrt(lb_n)
t_crit = stats.t.ppf(1 - alpha / 2, df=lb_v)
marg_err = t_crit * lb_s

ci_lower = lb_mean - marg_err
ci_upper = lb_mean + marg_err

print(type(ci_lower))
print(type(ci_upper))
print(ci_lower)
print(ci_upper)

print(f"Mittelwert LB: {lb_mean:.4f}")
#print(f"Kritischer t-Wert (df={lb_v}): {t_crit:.4f}")
print(f"95%-Konfidenzintervall: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 1.iii
# In der Literatur ist für die Produktion des gleichen Enzymes in LB-Medium eine Ausbeute von 3 mg pro Liter Expressionskultur beschrieben.
# Testen Sie auf einem Konfidenzniveau von α = 0.05 die einseitige Hypothese, dass die nun erzielte Ausbeute bei Verwendung von LB-Medium größer sei!
# Führen Sie im Anschluss den gleichen Test für TB-Medium durch!
# Welche Voraussetzungen müssen für die Tests gegeben sein und wie können Sie das ggf. prüfen?





# 1.iv
# Vergleichen Sie nun die für LB und TB erzielten Ausbeuten! Testen Sie auf
# Konfidenzniveau α = 0.01 die zweiseitige Hypothese, dass sich die Ausbeuten
# unterscheiden! Auch hier sollte vorab eine Betrachtung der Voraussetzungen erfolgen.