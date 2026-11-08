# Möglicher Einfluss des Experimentators auf die erzielte Ausbeute
# Bei der Durchführung der Experimente in den Teilprojekten #1 und #2 kam der Verdacht auf,
# dass systematische Einflüsse der den Versuch durchführenden Person bestehen könnten.
# Zu diesem Zweck führten (wie im Tab ‚person‘ aufgelistet) neben dem ursprünglichen Experimentator Jim
# auch noch die Kolleginnen Jane und Jacky entsprechende Versuche durch. Da (bekanntlich) der Chef der Abteilung
# es häufig besser wissen will, führt er das gleiche Experiment ebenso durch (PI).
# Als Bedingungen für die Produktion des Enzyms wurden hier eine Dauer von 24 h, TB Medium
# und der Stamm E. coli BL21 eingesetzt.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv(
    "data/person.csv",
    sep=";",
    skiprows=2,
    header=0,
    dtype=str
)

person_df = df.melt(var_name='Person', value_name='Ausbeute').dropna()
person_df['Ausbeute'] = person_df['Ausbeute'].apply(lambda s: (pd.to_numeric(s.replace(",", "."), errors='coerce')))

jim = person_df[person_df['Person'] == 'Jim']['Ausbeute']
jane = person_df[person_df['Person'] == 'Jane']['Ausbeute']
pi = person_df[person_df['Person'] == 'PI']['Ausbeute']
jacky = person_df[person_df['Person'] == 'Jacky']['Ausbeute']


# Bearbeiten Sie nun folgende Aufgaben und Fragen:


# 3.i
# Stellen Sie die Daten wiederum in geeigneter Weise grafisch dar und bestimmen die gängigen statistischen
# Kenngrößen Mittelwert und Varianz für alle Personen!

# BOXPLOT
personen = ['Jim', 'Jane', 'PI', 'Jacky']
daten = [person_df[person_df["Person"] == p]['Ausbeute'].values for p in personen]
fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot(daten, tick_labels=personen)
for i, x in enumerate(daten, start=1):
    # Leichte horizontale Streuung (Jitter), damit überlappende Punkte sichtbar werden
    ax.scatter(np.random.normal(i, 0.05, len(x)), x, s=15, alpha=0.6)
ax.set_ylabel('Ausbeute (mg/L)')
ax.set_title('BL21, TB, 24 h')
plt.savefig('plots/a3_boxplot.svg', bbox_inches='tight')
plt.close()

# Berechnung der deskriptiven Kenngrößen: Mittelwert und Varianz pro Person
# .agg() wendet die Funktionen auf jede Gruppe an
kenn = person_df.groupby('Person')['Ausbeute'].agg(['mean', 'var'])
print('Mittelwert und Varianz pro Person:')
print(kenn)



# 3.ii
# Führen Sie eine einfaktorielle Varianzanalyse nach Modell II durch und beantworten auf einem
# Konfidenzniveau von α = 0.05 die Frage, ob ein signifikanter Einfluss des zufälligen Einflussfaktors Experimentator
# auf die erzielten Ausbeuten des Enzyms besteht!

"""
HINTERGRUND: Modell I vs. Modell II bei der einfaktoriellem ANOVA

Modell I (fixed effects / feste Effekte):
  - Es gibt eine festgelegte Anzahl an Stufen, die gezielt ausgewählt wurden.
  - Die Frage ist: Unterscheiden sich GENAU diese Stufen voneinander?
  - Beispiel: Man vergleicht gezielt 4 bestimmte Enzymstämme.

Modell II (random effects / zufällige Effekte):
  - Die Stufen sind eine Zufallsauswahl aus einer größeren Grundgesamtheit möglicher Stufen.
  - Die Frage ist nicht, ob genau diese vier Personen sich unterscheiden.
  - Die Frage ist: Ist die Person überhaupt eine relevante Streuquelle?
  - Man will generalisieren: Würde man andere Personen aus dem Labor nehmen,
    gäbe es dann auch Unterschiede?

Hypothesen beim Modell II:
  H0: σ²_A = 0   (Die Varianz des Personeneffekts ist 0 → Person macht keinen Unterschied)
  H1: σ²_A > 0   (Die Varianz des Personeneffekts ist > 0 → Person ist eine Streuquelle)

Faktor: Person (Experimentator)
Stufen: Jim, Jane, PI, Jacky (Ausprägungen)
Werte: Ausbeute (mg/L)
"""

# VORAUSSETZUNGEN PRÜFEN
# 1. Normalverteilungstest (Shapiro-Wilk) pro Gruppe
#    H0: Die Daten sind normalverteilt.
#    p > 0.05 → H0 wird NICHT verworfen → Normalverteilung ist plausibel.
print('=== Shapiro-Wilk Normalverteilungstest ===')
for p, x in zip(personen, daten):
    sw_p = stats.shapiro(x).pvalue
    print(f'{p}: Shapiro-Wilk p = {sw_p:.3f}  -> {"Normalverteilung plausibel" if sw_p > 0.05 else "KEINE Normalverteilung"}')

# 2. Varianzhomogenitätstest (Levene)
#    H0: Die Varianzen sind gleich (homogen).
#    p > 0.05 → H0 wird NICHT verworfen → Varianzen sind homogen.
print('\n=== Levene Varianzhomogenitätstest ===')
lev_mean = stats.levene(*daten, center='mean')
print(f'Levene p-Wert (mean)   : {lev_mean.pvalue:.4f}  -> {"Varianzen homogen" if lev_mean.pvalue > 0.05 else "Varianzen HETEROGEN"}')


# ANOVA — F-Test
print('\n=== ANOVA F-Test ===')
anova = stats.f_oneway(*daten)
print(f'ANOVA F-Wert: {anova.statistic:.4f}, p-Wert: {anova.pvalue:.6f}')
if anova.pvalue < 0.05:
    print('Ergebnis: signifikanter Unterschied zwischen den Personen (alpha = 0.05)')
else:
    print('Ergebnis: KEIN signifikanter Unterschied zwischen den Personen (alpha = 0.05)')

# Freiheitsgrade
k = len(daten)  # Anzahl der Gruppen
n = len(daten[0])  # Probengröße pro Gruppe (hier gleich groß)
print(f'Freiheitsgrade: df_zwischen = {k - 1}, df_innerhalb = {k * (n - 1)}')


# MIXED-EFFECTS-MODELL (Random-Effects-Modell für Modell II)

# Es schätzt direkt die Varianzkomponenten:
#   - Group Var (σ²_A): Varianz zwischen den Personen (Personeneffekt)
#   - Scale (σ²_E): Residuale Varianz (innerhalb der Personen)
# Wenn σ²_A signifikant > 0 ist, gibt es einen Personeneffekt.
print('\n=== Mixed-Effects-Modell (Random Effects) ===')
mm = smf.mixedlm('Ausbeute ~ 1', data=person_df, groups=person_df['Person']).fit(reml=True)
print(mm.summary())
# 'Group Var' ≈ σ²_A (Varianz des Personeneffekts)
# 'Scale' ≈ σ²_E (residuale Varianz)
print('\nGeschätzte Effekte der einzelnen Personen (BLUPs):')
print(mm.random_effects)



# 3.iii
# Falls der Test bei der vorherigen Frage positiv ausfällt, diskutieren Sie, wie zu testen wäre,
# welche Stufe (d.h. welche Person) diesen signifikanten Einfluss ausübt!
# (Eine Lösung kann entweder auf Basis der vorhandenen Daten erfolgen [das ist anspruchsvoll und am besten mit Python adressierbar],
# oder Sie schlagen ein neues Experiment vor, mit dem eine Beantwortung möglich wäre.)

