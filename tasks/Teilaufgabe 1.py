# Heterologe Produktion eines Enzyms in verschiedenen Nährmedien
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
import math
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.multicomp as smm
import statsmodels.formula.api as smf
from matplotlib.ticker import MultipleLocator

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

#BOXPLOT
fig, ax = plt.subplots(figsize=(5, 4))
#Whiskers werden über den modifizierten Boxplot berechnet (1.5 * IQR), aber Extremwerte begrenzen die Whiskers
ax.boxplot([lb_values, tb_values], tick_labels=['LB', 'TB'])
for i, x in enumerate([lb_values, tb_values], start=1):
    ax.scatter(np.random.normal(i, 0.04, len(x)), x, s=15, alpha=0.6) #Stripplot
ax.set_ylabel('Ausbeute (mg pro L Kultur)')
ax.set_title('Enzymausbeute nach 24 h')
ax.set_ylim(bottom=0, auto=True)
#fig.savefig('plots/a1_boxplot.svg') ##nur zur Neugenerierung der Boxplots auskommentieren
plt.show()

#HISTOGRAMM
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
for ax, (name, x) in zip(axes, [('LB', lb_values), ('TB', tb_values)]):
    m = int(round(1 + 3.32 * np.log10(len(x)))) #Sturges (VL02, F8)
    print(m)
    ax.hist(x, bins=m, edgecolor='black')
    ax.set_title(name)
    ax.set_xlabel('Ausbeute (mg/L)')
    ax.xaxis.set_major_locator(MultipleLocator(1))
axes[0].set_ylabel('absolute Häufigkeit')
ax.set_xlim(left=0, auto=True)
fig.tight_layout()
#fig.savefig('plots/a1_histogramme.svg') ##nur zur Neugenerierung der Histogramme auskommentieren
plt.show()


# Bestimmen Sie für beide Datensätze Mittelwert, Varianz, Median und die Quartile Q1 und Q3!
###### LB ######
lb_mean = lb_values.mean()
lb_var = lb_values.var()
lb_median = lb_values.median()

lb_q1_lin = lb_values.quantile(0.25) #linear; der Boxplot verwendet lineare Interpolation
lb_q3_lin = lb_values.quantile(0.75) #linear

lb_values_sort = lb_values.sort_values()            #np.percentile sortiert an sich auch selbst, hier nur zur Veranschaulichung
lb_q1, lb_q3 = np.percentile(lb_values_sort, [25, 75], method='midpoint')

print(f"Mittelwert LB: {lb_mean:.4f}")
print(f"Varianz LB: {lb_var:.4f}")
print(f"Median LB: {lb_median:.4f}")
print(f"Quartile LB (linear): (Q1/Q3): {lb_q1_lin:.4f}/{lb_q3_lin:.4f}")
print(f"Quartile LB: (Q1/Q3): {lb_q1:.4f}/{lb_q3:.4f}")
print()

###### TB ######
tb_mean = tb_values.mean()
tb_var = tb_values.var()
tb_median = tb_values.median()

tb_q1_lin = tb_values.quantile(0.25) #linear; der Boxplot verwendet lineare Interpolation
tb_q3_lin = tb_values.quantile(0.75) #linear

tb_values_sort = tb_values.sort_values()            #np.percentile sortiert an sich auch selbst, hier nur zur Veranschaulichung
tb_q1, tb_q3 = np.percentile(tb_values_sort, [25, 75], method='midpoint')

print(f"Mittelwert TB: {tb_mean:.4f}")
print(f"Varianz TB: {tb_var:.4f}")
print(f"Median TB: {tb_median:.4f}")
print(f"Quartile TB (linear): (Q1/Q3): {tb_q1_lin:.4f}/{tb_q3_lin:.4f}")
print(f"Quartile TB: (Q1/Q3): {tb_q1:.4f}/{tb_q3:.4f}")
print()


# 1.ii
# Bestimmen Sie für beide Medien 95%-Konfidenzintervalle, innerhalb derer sich der wirkliche Mittelwert der Ausbeute befindet!
alpha = 0.05                    #α
confidence = 1 - alpha          #1-α  -Konfidenzniveau

###### LB ######
# Stichprobengröße und Standardfehler (SEM)
lb_n = len(lb_values)           #N          -Stichprobengröße
lb_dof = lb_n - 1                #ν (N-1)    -Freiheitsgrade
lb_s = lb_values.std(ddof=1)    #s          -Standardabweichung (N-1)
lb_sem = lb_s / np.sqrt(lb_n)   #s_x̄        -Standardfehler des Mittelwertes (SEM)

# Quantile und Grenzen des Konfidenzintervalls
lb_q = stats.t.ppf(1-alpha/2, df=lb_dof)

g0 = lb_mean - lb_q * lb_sem
g1 = lb_mean + lb_q * lb_sem

print(f"Quantil t_(v,1-α/2): {lb_q:.4f}")
print(f"95%-Konfidenzintervall (LB): [{g0:.4f}, {g1:.4f}];")


###### TB ######
# Stichprobengröße und Standardfehler (SEM)
tb_n = len(tb_values)           #N          -Stichprobengröße
tb_dof = tb_n - 1                #ν (N-1)    -Freiheitsgrade
tb_s = tb_values.std(ddof=1)    #s          -Standardabweichung
tb_sem = tb_s / np.sqrt(tb_n)   #s_x̄        -Standardfehler des Mittelwertes (SEM)

# Quantile und Grenzen des Konfidenzintervalls
tb_q = stats.t.ppf(1-alpha/2, df=tb_dof)

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

# 1.iii
# In der Literatur ist für die Produktion des gleichen Enzymes in LB-Medium eine Ausbeute von 3 mg pro Liter Expressionskultur beschrieben.
# Testen Sie auf einem Konfidenzniveau von α = 0.05 die einseitige Hypothese, dass die nun erzielte Ausbeute bei Verwendung von LB-Medium größer sei!
# Führen Sie im Anschluss den gleichen Test für TB-Medium durch!
# Welche Voraussetzungen müssen für die Tests gegeben sein und wie können Sie das ggf. prüfen?

# H_1: µ > µ_0 = 3 mg/L
# H_0: µ <= 3 mg/L
µ_0 = 3

###### LB ######
# Shapiro-Wilk-Test
# H_0: Die Daten sind normalverteilt.
# H_1: Die Daten sind nicht normalverteilt.
lb_shapiro = stats.shapiro(lb_values)
print()
print(f"Shapiro-Wilk: W = {lb_shapiro.statistic:.4f}, p = {lb_shapiro.pvalue:.4f}")
if (lb_shapiro.pvalue < alpha):
    print("Shapiro: H_0 wird abgelehnt, die Daten sind nicht normalverteilt.")
else:
    print("Shapiro: H_0 kann nicht abgelehnt werden, es gibt keine Evidenz gegen die Normalverteilung.")

#Berechnung der Teststatistik t
#kompletter Test:
lb_ttest = stats.ttest_1samp(lb_values, µ_0, alternative='greater')
#lb_p_value = lb_ttest.pvalue

lb_t = (lb_mean - µ_0) * (np.sqrt(lb_n)/lb_s)
#Notiz: lb_df = lb_n-1
lb_p_value = -stats.t.cdf(lb_t, lb_dof)+1

#Händisch
print(f"Teststatistik t (LB): {lb_t:.4f}")
print(f"Freiheitsgrade (LB): {lb_dof:.4f}")
print(f"p-Wert (LB): {lb_p_value:.4f}")


#wenn p<α (0.05), dann Ablehnung von H_0
if(lb_p_value < alpha):
    print("Die Ausbeute bei Verwendung von LB-Medium ist signifikant größer als 3 mg/L. "
          "H_1 wird angenommen, H_0 wird abgelehnt")
else:
    print("Die Ausbeute bei Verwendung von LB-Medium ist nicht signifikant größer als 3 mg/L. "
          "H_1 kann nicht angenommen werden, H_0 kann nicht abgelehnt werden. Daraus folgt nicht, dass H_0 gilt.")


###### TB ######
# Shapiro-Wilk-Test
# H_0: Die Daten sind normalverteilt.
# H_1: Die Daten sind nicht normalverteilt.
tb_shapiro = stats.shapiro(tb_values)
print()
print(f"Shapiro-Wilk: W = {tb_shapiro.statistic:.4f}, p = {tb_shapiro.pvalue:.4f}")
if (tb_shapiro.pvalue < alpha):
    print("Shapiro: H_0 wird abgelehnt, die Daten sind nicht normalverteilt.")
else:
    print("Shapiro: H_0 kann nicht abgelehnt werden, es gibt keine Evidenz gegen die Normalverteilung.")

#Berechnung der Teststatistik t
#kompletter Test:
tb_ttest = stats.ttest_1samp(tb_values, µ_0, alternative='greater')
#tb_p_value = tb_ttest.pvalue

#Q-Q-Plot --> Graphische Überprüfung der Normalverteilung
#VL08:37
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for ax, (name, x) in zip(axes, [('LB', lb_values), ('TB', tb_values)]):
    stats.probplot(x, plot=ax)
    ax.set_title(f'Q-Q-Plot {name}')
fig.tight_layout();
#fig.savefig('plots/a1_qqplots.svg') ##nur zur Neugenerierung der Q-Q-Plots auskommentieren
plt.show()


tb_t = (tb_mean - µ_0) * (np.sqrt(tb_n)/tb_s)
#Notiz: tb_df = tb_n-1
tb_p_value = -stats.t.cdf(tb_t, tb_dof)+1

#Händisch
print(f"Teststatistik t (TB): {tb_t:.4f}")
print(f"Freiheitsgrade (TB): {tb_dof:.4f}")
print(f"p-Wert (TB): {tb_p_value:.4f}")

#wenn p<α (0.05), dann Ablehnung von H_0
if(tb_p_value < alpha):
    print("Die Ausbeute bei Verwendung von TB-Medium ist signifikant größer als 3 mg/L. "
          "H_1 wird angenommen, H_0 wird abgelehnt")
else:
    print("Die Ausbeute bei Verwendung von TB-Medium ist nicht signifikant größer als 3 mg/L. "
          "H_1 kann nicht angenommen werden, H_0 kann nicht abgelehnt werden. Daraus folgt nicht, dass H_0 gilt.")

# 1.iv
# Vergleichen Sie nun die für LB und TB erzielten Ausbeuten! Testen Sie auf
# Konfidenzniveau α = 0.01 die zweiseitige Hypothese, dass sich die Ausbeuten
# unterscheiden! Auch hier sollte vorab eine Betrachtung der Voraussetzungen erfolgen.

alpha = 0.01
confidence = 1-alpha
#Gibt es einen signifikanten Unterschied zwischen LB und TB? Man will sich sehr (höheres Konfidenzniveau) sicher sein, bevor man sich auf ein Medium festlegt
#Man lässt höchstens 1% Risiko für einen falsch positiven Befund zu (Fehler 1. Art)

#H_1: µ_LB != µ_TB
#H_0: µ_LB = µ_TB

#Unabhängige Stichproben? Ja! Kein gepaarter t-Test, kein Wilcoxon-Vorzeichentest
#Normalverteilung? Ja!
#Varianz-Homogenität

#H1: σ^2_LB != σ^2_TB
#H0: σ^2_LB = σ^2_TB
levene = stats.levene(lb_values, tb_values, center='mean')

if levene.pvalue < 0.05: #ein strenges σ hier (Voraussetzungstest) sogar ungünstig (also 0.05 üblich). Da wird eher eine verletzte Voraussetzung (Fehler 2. Art) übersehen und genau diesen Fehler will man hier vermeiden.
    print("Hier machen wir theoretisch einen Welch-Test")
else:
    print("Hier machen wir theoretisch einen Student-Test")

welch = stats.ttest_ind(lb_values, tb_values, equal_var=False, alternative='two-sided')
#student = stats.ttest_ind(lb_values, tb_values, equal_var=True, alternative='two-sided')
print(f"welch_p={welch.pvalue}, welch_t={welch.statistic}")

if(welch.pvalue < alpha):
    print("Wir verwerfen H_0: TB ist signifikant besser als LB")
else:
    print("Wir können H_0 nicht verwerfen: kein signifikanter Vorteil von TB gegenüber LB nachgewiesen")