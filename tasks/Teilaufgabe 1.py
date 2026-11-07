### Heterologe Produktion eines Enzyms in verschiedenen Nährmedien

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Sehr häufig werden in der biochemischen Forschung Proteine und Enzyme durch heterologe Genexpression hergestellt.
# Für beste Ausbeute sind zahlreiche Parameter zu variieren, darunter z. B. die Wahl des Nahrmediums (dieses Teilprojekt)
# sowie des Expressionsstamms und der Dauer der Expression (siehe Teilprojekt #2).
# Die Daten in dem Tab 'media' wurden erhoben in mehreren Expressionsversuchen bei Verwendung von 'Lysogeny Broth' (LB)
# bzw. 'Terrific Broth' (TB) Medium. Da das produzierte Enzym einen im sichtbaren Spektrum absorbierenden Cofactor trägt,
# kann die bei der Expression erzielte Menge/Ausbeute an Enzym (in Einheiten von Milligramm) pro Liter Expressionskultur
# sehr leicht abgeschätzt werden. (Aufgrund der Art dieser Bestimmung kann es rechnerisch auch zu negativen Werten
# für die Ausbeute kommen - nicht ablenken lassen dadurch.) Die einzelnen Einträge in der Tabelle beziehen sich auf
# Wiederholungsmessungen.

"""
Die einzelnen Tabellenblätter der Excel-Tabelle wurden separat als CSV-Dateien exportiert. Die Daten lassen sich so in 
PyCharm besser einsehen, der Import ist nahezu identisch wie mit Excel-Dateien. 
"""

# CSV-Datei einlesen:
df = pd.read_csv(
    "data/media.csv",
    sep=";",
    header=3,
    decimal=","
)

# Spalten für LB- und TB-Medium extrahieren und fehlende Werte (NaN) entfernen
lb_values = df["LB"].dropna()
tb_values = df["TB"].dropna()

# Bearbeiten Sie nun folgende Aufgaben und Fragen:

########################################################################################################################

# region Aufgabe 1.i
# Stellen Sie die Ausbeuten für LB und TB in geeigneter Weise grafisch dar!

# BOXPLOT
"""
Der Boxplot visualisiert Median, Quartile (Q1/Q3) und den IQR (Interquartilsabstand = Q3 - Q1).
Die Whiskers erstrecken sich maximal bis 1.5 · IQR vom Boxrand. Werte darüber werden als
einzelne Punkte (Ausreißer/Extremwerte) dargestellt.
Der Stripplot (gestreute Punkte) zeigt zusätzlich alle Einzelmessungen und macht die
Stichprobengröße und die Datenverteilung sichtbar.
"""
fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot([lb_values, tb_values], tick_labels=['LB', 'TB'])
for i, x in enumerate([lb_values, tb_values], start=1):
    ax.scatter(np.random.normal(i, 0.04, len(x)), x, s=15, alpha=0.6) # Stripplot
ax.set_ylabel('Ausbeute (mg pro L Kultur)')
ax.set_title('Enzymausbeute nach 24 h')
ax.set_ylim(bottom=0, auto=True)
fig.savefig('plots/a1_boxplot.svg') # nur zur Neugenerierung der Boxplots auskommentieren
plt.show()

# HISTOGRAMM
"""
Das Histogramm zeigt die Häufigkeitsverteilung der Ausbeuten.
Die Anzahl der Klassen (Bins) wird nach der Regel von Sturges bestimmt:
  k = round(1 + 3.32 · log10(N))
"""
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
for ax, (name, x) in zip(axes, [('LB', lb_values), ('TB', tb_values)]):
    m = int(round(1 + 3.32 * np.log10(len(x))))  # Sturges-Regel für Anzahl der Klassen
    ax.hist(x, bins=m, edgecolor='black')
    ax.set_title(name)
    ax.set_xlabel('Ausbeute (mg/L)')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(left=0, auto=True)
axes[0].set_ylabel('absolute Häufigkeit')
fig.tight_layout()
fig.savefig('plots/a1_histogramme.svg')  # nur zur Neugenerierung der Histogramme auskommentieren
plt.show()


# Bestimmen Sie für beide Datensätze Mittelwert, Varianz, Median und die Quartile Q1 und Q3!

def kennzahlen(values:pd.Series):
    """
    Berechnet und gibt die wichtigsten deskriptiven Kennzahlen einer Stichprobe aus:
    - Mittelwert (arithmetisches Mittel)
    - Varianz (stichprobenbezogen, ddof=1 → Division durch N-1)
    - Median (mittlerer Wert, robust gegenüber Ausreißern)
    - Quartile Q1 (25 %) und Q3 (75 %) — zwei Berechnungsmethoden:
        a) lineare Interpolation (pandas.quantile, Standard)
        b) Midpoint-Methode (np.percentile, entspricht dem, was matplotlibs boxplot intern nutzt)
    """
    mean = values.mean()
    var = values.var(ddof=1)
    median = values.median()

    q1_lin, q3_lin = values.quantile([0.25, 0.75])      # lineare Interpolation (pandas-Standard)

    values_sort = values.sort_values()  # np.percentile sortiert intern selbst, hier nur zur Veranschaulichung
    q1, q3 = np.percentile(values_sort, [25, 75], method='midpoint')  # Midpoint: Mittelwert der beiden mittleren Werte an der Quantil-Stelle

    print(f"Mittelwert {str(values.name)}: {mean:.4f}")
    print(f"Varianz {str(values.name)}: {var:.4f}")
    print(f"Median {str(values.name)}: {median:.4f}")
    print(f"Quartile {str(values.name)} Q1/Q3 (linear): {q1_lin:.4f}/{q3_lin:.4f}")
    print(f"Quartile {str(values.name)} Q1/Q3 (midpoint): {q1:.4f}/{q3:.4f}")
    print()


###### LB ######
kennzahlen(lb_values)

###### TB ######
kennzahlen(tb_values)

# endregion ############################################################################################################

# region Aufgabe 1.ii
# Bestimmen Sie für beide Medien 95%-Konfidenzintervalle, innerhalb derer sich der wirkliche Mittelwert der Ausbeute befindet!
alpha = 0.05                    #α    - Signifikanzniveau
confidence = 1 - alpha          #1-α  - Konfidenzniveau

def konfidenzintervalle(values:pd.Series):
    """
    Berechnet das (1-α)-Konfidenzintervall für den Populationsmittelwert µ.
    
    Da die Populationsvarianz unbekannt ist und durch die Stichprobenvarianz
    geschätzt wird, kommt die t-Verteilung (nicht die Normalverteilung) zum Einsatz.
    Das ist besonders bei kleinen Stichproben (N < 30) wichtig.
    
    Formel:  x̄ ± t_(v, 1-α/2) · s_x̄
    wobei s_x̄ = s / √N  (Standardfehler des Mittelwertes)
    und v = N - 1  (Freiheitsgrade)
    """
    # Stichprobengröße und Standardfehler (SEM)
    N = len(values)             # Stichprobengröße
    dof = N - 1                 # Freiheitsgrade (degrees of freedom)
    s = values.std(ddof=1)      # Stichproben-Standardabweichung (ddof=1 → Division durch N-1)
    sem = s / np.sqrt(N)        # Standardfehler des Mittelwertes (s_x̄ = s / √N)

    # t-Quantil für das zweiseitige Konfidenzintervall
    # ppf = percent point function
    q = stats.t.ppf(1 - alpha / 2, df=dof)

    # Untere (g0) und obere (g1) Grenze des Konfidenzintervalls
    g0 = values.mean() - q * sem
    g1 = values.mean() + q * sem

    print(f"Stichprobengröße ({str(values.name)}): {N}")
    print(f"Standardabweichung ({str(values.name)}): {s:.4f}")
    print(f"Standardfehler ({str(values.name)}): {sem:.4f}")
    print(f"Quantil t_(v,1-α/2): {q:.4f}")
    print(f"{int(confidence*100)}%-Konfidenzintervall ({str(values.name)}): [{g0:.4f}, {g1:.4f}]")
    print()
    return g0, g1

###### LB ######
konfidenzintervalle(lb_values)

###### TB ######
konfidenzintervalle(tb_values)

# endregion ############################################################################################################

# region Aufgabe 1.iii
# In der Literatur ist für die Produktion des gleichen Enzymes in LB-Medium eine Ausbeute von 3 mg pro Liter Expressionskultur beschrieben.
# Testen Sie auf einem Konfidenzniveau von α = 0.05 die einseitige Hypothese, dass die nun erzielte Ausbeute bei Verwendung von LB-Medium größer sei!
# Führen Sie im Anschluss den gleichen Test für TB-Medium durch!
# Welche Voraussetzungen müssen für die Tests gegeben sein und wie können Sie das ggf. prüfen?

# H_1: µ > µ_0 = 3 mg/L
# H_0: µ <= 3 mg/L
alpha = 0.05
confidence = 1 - alpha
µ_0 = 3

def shapiro_wilk(values:pd.Series):
    """
    Shapiro-Wilk-Test auf Normalverteilung.

    H_0: Die Daten stammen aus einer normalverteilten Population.
    H_1: Die Daten stammen NICHT aus einer normalverteilten Population.

    Ein kleiner p-Wert (< α) bedeutet: H_0 ablehnen → Normalverteilung nicht gegeben.
    """
    shapiro = stats.shapiro(values)
    print(f"Shapiro-Wilk ({str(values.name)}): W = {shapiro.statistic:.4f}, p = {shapiro.pvalue:.4f}")
    if shapiro.pvalue < alpha:
        print(f"Shapiro ({str(values.name)}): H_0 wird abgelehnt, die Daten sind nicht normalverteilt.")
    else:
        print(f"Shapiro ({str(values.name)}): H_0 kann nicht abgelehnt werden, es gibt keine Evidenz gegen die Normalverteilung.")
    print()
    return shapiro

def t_test(values:pd.Series):
    """
    Einseitiger ein-Stichproben-t-Test (rechtsseitig).
    
    H_0: µ ≤ µ_0  (Die mittlere Ausbeute ist kleiner oder gleich 3 mg/L)
    H_1: µ > µ_0  (Die mittlere Ausbeute ist größer als 3 mg/L)
    
    Die Teststatistik t wird berechnet als:
        t = (x̄ - µ_0) · √N / s
    
    Der p-Wert ist die Wahrscheinlichkeit, unter H_0 einen t-Wert zu beobachten,
    der mindestens so extrem (groß) ist wie der berechnete.
    
    Voraussetzungen:
    - Unabhängige Stichproben (jeder Messwert stammt von einer eigenen Kultur)
    - Normalverteilung der Grundgesamtheit (geprüft mit Shapiro-Wilk + Q-Q-Plot)
    """
    # scipy-Version des Tests (zur Kontrolle)
    t_test = stats.ttest_1samp(values, µ_0, alternative='greater')

    # Manuelles Berechnen zur Veranschaulichung der Formel
    N = len(values)
    dof = N - 1
    s = values.std(ddof=1)

    # t-Teststatistik: (x̄ - µ_0) · √N / s
    t = (values.mean() - µ_0) * (np.sqrt(N) / s)
    # p-Wert für einseitigen (rechtsseitigen) Test: P(T > t) = 1 - CDF(t)
    p_value = 1 - stats.t.cdf(t, dof)

    print(f"Teststatistik t ({str(values.name)}): {t:.4f}")
    print(f"Freiheitsgrade ({str(values.name)}): {dof}")
    print(f"p-Wert ({str(values.name)}): {p_value:.4f}")
    print(f"scipy p-Wert (Kontrolle): {t_test.pvalue:.4f}")

    # Entscheidung: wenn p < α, dann H_0 ablehnen zugunsten von H_1
    if p_value < alpha:
        print(f"Die Ausbeute bei Verwendung von {str(values.name)}-Medium ist signifikant größer als {µ_0} mg/L. "
              "H_0 wird abgelehnt, H_1 wird angenommen.")
    else:
        print(f"Die Ausbeute bei Verwendung von {str(values.name)}-Medium ist nicht signifikant größer als {µ_0} mg/L. "
              "H_0 kann nicht abgelehnt werden. (Das bedeutet NICHT, dass H_0 wahr ist!)")
    print()


###### LB ######
shapiro_wilk(lb_values)
t_test(lb_values)


###### TB ######
shapiro_wilk(tb_values)
t_test(tb_values)


"""
Q-Q-Plot (Quantile-Quantile-Plot) → Graphische Überprüfung der Normalverteilung
Im Q-Q-Plot werden die quantilen der Daten gegen die quantilen einer theoretischen
Normalverteilung aufgetragen. Liegen die Punkte annähernd auf der Diagonalen,
ist die Normalverteilungsannahme plausibel.
Systematische Abweichungen (z. B. S-förmige Kurve) deuten auf Schiefe oder
schwerere/leichtere Verteilungsschwerter hin.
"""
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for ax, (name, x) in zip(axes, [('LB', lb_values), ('TB', tb_values)]):
    stats.probplot(x, plot=ax)
    ax.set_title(f'Q-Q-Plot {name}')
fig.tight_layout()
fig.savefig('plots/a1_qqplots.svg')  # nur zur Neugenerierung der Q-Q-Plots auskommentieren
plt.show()

# endregion ############################################################################################################

# region Aufgabe 1.iv
# Vergleichen Sie nun die für LB und TB erzielten Ausbeuten! Testen Sie auf
# Konfidenzniveau α = 0.01 die zweiseitige Hypothese, dass sich die Ausbeuten
# unterscheiden! Auch hier sollte vorab eine Betrachtung der Voraussetzungen erfolgen.

alpha = 0.01
confidence = 1-alpha
#Gibt es einen signifikanten Unterschied zwischen LB und TB? Man will sich sehr (höheres Konfidenzniveau) sicher sein,
# bevor man sich auf ein Medium festlegt
#Man lässt höchstens 1% Risiko für einen falsch positiven Befund zu (Fehler 1. Art)

#H_1: µ_LB != µ_TB
#H_0: µ_LB = µ_TB

# Unabhängige Stichproben? Ja! Kein gepaarter t-Test, kein Wilcoxon-Vorzeichentest.
# Normalverteilung? Ja! (wurde in 1.iii mit Shapiro-Wilk geprüft)
# Varianz-Homogenität? → Levene-Test

# Levene-Test auf Varianzgleichheit
# H_0: σ²_LB = σ²_TB  (Varianzen sind gleich)
# H_1: σ²_LB ≠ σ²_TB  (Varianzen sind ungleich)
levene = stats.levene(lb_values, tb_values, center='mean')
print(f"Levene-Test: p = {levene.pvalue:.4f}")

# Für den Voraussetzungstest (Varianzgleichheit) wird üblicherweise α = 0.05 verwendet,
# auch wenn der Haupttest bei α = 0.01 liegt. Grund: Man will eine verletzte Voraussetzung
# (Fehler 2. Art beim Voraussetzungstest) möglichst nicht übersehen.
if levene.pvalue < 0.05:
    print("Varianzgleichheit wird verworfen → Welch-Test (keine Annahme gleicher Varianzen)")
    use_welch = True
else:
    print("Varianzgleichheit kann nicht verworfen werden → Student-t-Test (gleiche Varianzen angenommen)")
    use_welch = False

# Zweiseitiger zwei-Stichproben-t-Test
# Welch-Test (equal_var=False) ist immer die sicherere Wahl, da er keine Varianzgleichheit voraussetzt.
# Bei ähnlichen Varianzen liefert er nahezu das gleiche Ergebnis wie der Student-Test.
welch = stats.ttest_ind(lb_values, tb_values, equal_var=False, alternative='two-sided')
student = stats.ttest_ind(lb_values, tb_values, equal_var=True, alternative='two-sided')

print(f"Welch-Test:  t = {welch.statistic:.4f}, p = {welch.pvalue:.4f}")
print(f"Student-Test: t = {student.statistic:.4f}, p = {student.pvalue:.4f}")


# Entscheidung basierend auf dem gewählten Test
# Bei zweiseitigem Test können wir nur sagen, ob sich die Mittelwerte unterscheiden — NICHT, welches Medium besser ist.
# Die Richtung muss separat geprüft werden.
if use_welch:
    p_value = welch.pvalue
    test_name = "Welch"
else:
    p_value = student.pvalue
    test_name = "Student"

if p_value < alpha:
    print(f"\n{test_name}-Test: H_0 wird verworfen. TB und LB unterscheiden sich signifikant (p = {p_value:.4f} < α = {alpha}).")
else:
    print(f"\n{test_name}-Test: H_0 kann nicht verworfen werden. Kein signifikanter Unterschied zwischen LB und TB (p = {p_value:.4f} > α = {alpha}).")
print()

# endregion ############################################################################################################