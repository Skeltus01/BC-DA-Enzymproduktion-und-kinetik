# Anknüpfend an die vorherige Fragestellung wird nun für das gleiche Enzym (in TB Medium) getestet,
# welche Einflüsse die Faktoren verwendeter Expressionsstamm (E. coli BL21, LOBSTR, BL21 Rosetta, C41, BL21 Gold)
# und Expressionsdauer (bis zu 48 h) auf die Ausbeute haben. Daten hierzu finden sich im Tab ‚production‘.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.stats.multicomp as smm

df = pd.read_csv(
    "data/production.csv",
    sep=";",
    header=None,
    #Daten werden als String importiert, da es zuvor Probleme mit der Erkennung der Datentypen gab. Die notwendigen numerischen Daten werden später passend umgewandelt.
    dtype=str
)

namen = ['BL21', 'LOBSTR', 'BL21 Rosetta', 'C41', 'BL21 Gold']
zeiten = [6, 12, 18, 24, 48]

stamm = {}
teile = []
for name in namen:
    start = df.index[df[0] == name][0]
    block = df.iloc[start + 3: start + 17, 1:6]
    block = block.apply(lambda s: (pd.to_numeric(s.astype(str).str.replace(",", "."), errors='coerce')))
    block.columns = zeiten
    block = block.reset_index(drop=True)
    stamm[name] = block
    lang = block.melt(var_name='Zeit', value_name='Ausbeute')
    lang['Stamm'] = name
    teile.append(lang)

staemme_df = pd.concat(teile, ignore_index=True)

bl21 = stamm['BL21']
lobstr = stamm['LOBSTR']
bl21_rosetta = stamm['BL21 Rosetta']
c41 = stamm['C41']
bl21_gold = stamm['BL21 Gold']

# Bearbeiten Sie nun folgende Aufgaben und Fragen:

# 2.i
# Stellen Sie die Daten wiederum in geeigneter Weise grafisch dar und bestimmen die gängigen statistischen Kenngrößen Mittelwert
# und Varianz für alle Bedingungen!

# Plot: Mittelwert ± Standardabweichung über die Expressionsdauer für jeden Stamm
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for name, values in stamm.items():
    m = values.mean()
    s = values.std(ddof=1)
    ax.errorbar(m.index, m.values, yerr=s.values, marker='o', capsize=3, label=name)
ax.set_xlabel('Expressionsdauer (h)')
ax.set_ylabel('mittlere Ausbeute (mg/L)')
ax.set_xticks([6, 12, 18, 24, 48])
ax.set_ylim(0, 7)
ax.legend(title='Expressionsstamm', frameon=True, fontsize='small', title_fontsize='small')
fig.savefig('plots/a2_verlauf.svg', bbox_inches='tight')
plt.show()

# Boxplot: Verteilung der Ausbeute pro Stamm für jeden Zeitpunkt
# Jeder Boxplot zeigt Median, Quartile (25%-75%), Whisker (1.5*IQR) und Ausreißer
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
for ax, z in zip(axes, zeiten):
    # Für jeden Zeitpunkt z: sammle die Ausbeute-Werte aller Stämme als separate Gruppen
    daten = [staemme_df[(staemme_df.Zeit == z) & (staemme_df.Stamm == s)]['Ausbeute'].dropna() for s in namen]
    ax.boxplot(daten, tick_labels=namen)
    ax.set_title(f'{z} h'); ax.tick_params(axis='x', rotation=60)
axes[0].set_ylabel('Ausbeute (mg/L)')
fig.tight_layout()
fig.savefig('plots/a2_boxplots.svg')
plt.show()

# Mittelwerte & Varianzen für alle Kombinationen von Stamm und Zeitpunkt
# groupby gruppiert nach den Spalten 'Stamm' und 'Zeit', agg berechnet mean und var pro Gruppe
print('Mittelwerte & Varianzen')
print(staemme_df.groupby(['Stamm', 'Zeit'])['Ausbeute'].agg(['mean', 'var']))



# 2.ii
# Berechnen Sie für jeden Stamm und jeden Zeitpunkt 95%-Konfidenzintervalle für die mittlere Ausbeute!

"""
Das Konfidenzintervall (KI) gibt den Bereich an, in dem der wahre Populationsmittelwert mit
einer Wahrscheinlichkeit von (1-α) liegt. Bei kleinen Stichproben (n < 30) wird die t-Verteilung
statt der Normalverteilung verwendet, da diese den größeren Unsicherheitsbereich bei kleinen n berücksichtigt.
"""

alpha = 0.05                    # Signifikanzniveau α
confidence = 1 - alpha          # Konfidenzniveau 1-α = 0.95 (95%)

print('95%-Konfidenzintervalle für die mittlere Ausbeute:')

teile = []
for (stamm, zeit), values in staemme_df.groupby(['Stamm', 'Zeit'])['Ausbeute']:
    mean = values.mean()        # Stichprobenmittelwert x̄
    n = len(values)             # Stichprobengröße
    dof = n - 1                 # Freiheitsgrade ν = n-1 für die t-Verteilung
    s = values.std(ddof=1)      # Stichprobenstandardabweichung
    sem = s / np.sqrt(n)        # Standardfehler des Mittelwerts (SEM = s/√n)
    q = stats.t.ppf(1-alpha/2, df=dof)  # t-Quantil für 1-α/2 (zweiseitig)

    # Konfidenzintervall: [x̄ - t·SEM, x̄ + t·SEM]
    g0, g1 = stats.t.interval(1 - alpha, df=dof, loc=mean, scale=sem)
    print(f"{stamm}, {zeit}: Quantil t_(ν,1-α/2): {q:.4f}")
    print(f"{stamm}, {zeit}: {(1 - alpha) * 100:.0f}%-Konfidenzintervall: [{g0:.4f}, {g1:.4f}]")
    teile.append({'Stamm': stamm, 'Zeit': zeit, 'Mittelwert': mean, 'g0': g0, 'g1': g1})
ki_df = pd.DataFrame(teile).set_index(['Stamm', 'Zeit'])

# Plot: Mittelwert ± 95%-Konfidenzintervall über die Expressionsdauer
# Im Gegensatz zum Plot in 2.i (Mittelwert ± Standardabweichung) zeigt das KI die Unsicherheit
# des Mittelwerts selbst — schmalere Balken bei größeren Stichproben, da SEM = s/√n.
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for name in namen:
    k = ki_df.loc[name]
    ax.errorbar(k.index, k['Mittelwert'], yerr=[k['Mittelwert'] - k['g0'], k['g1'] - k['Mittelwert']], marker='o', capsize=3, label=name)
ax.set_xlabel('Expressionsdauer (h)')
ax.set_ylabel('mittlere Ausbeute ± 95%-KI (mg/L)')
ax.set_xticks(zeiten)
ax.set_ylim(0, 7)
ax.legend(title='Expressionsstamm', frameon=True, fontsize='small', title_fontsize='small')
fig.savefig('plots/a2_verlauf_KI.svg', bbox_inches='tight')
plt.show()

# 2.iii
# Führen Sie eine einfaktorielle Varianzanalyse für den Einflussfaktor Expressionsstamm durch und
# legen dabei die Messwerte beim Zeitpunkt 24 h zugrunde! Testen Sie auf einem Konfidenzniveau von α = 0.01,
# ob ein Einfluss dieses Faktors auf die Ausbeute an Enzym besteht! Falls dieser Test positiv ausfällt,
# führen Sie nun einen adäquaten post hoc-Test durch, um festzustellen, welche Stufe(n) des Faktors
# hier maßgeblich sind/ist!

"""
Die einfaktorielle ANOVA (One-Way ANOVA) prüft, ob sich die Mittelwerte mehrerer Gruppen
signifikant unterscheiden. Die Nullhypothese H0: Alle Gruppenmittelwerte sind gleich.
Alternativhypothese H1: Mindestens ein Gruppenmittelwert unterscheidet sich.

Voraussetzungen der ANOVA:
1. Feste Stufen (fixed effects, Modell I) — die gewählten Stämme sind bewusst und fest gewählt
2. Unabhängige, zufällige Fehler e_ij — Messungen sind unabhängig voneinander
3. Normalverteilung der Messwerte innerhalb jeder Gruppe — wird mit Shapiro-Wilk geprüft
4. Varianzhomogenität — die Varianzen der Gruppen sind gleich, wird mit Levene-Test geprüft
"""

df_24 = staemme_df[staemme_df['Zeit'] == 24]  # Filtert alle Daten auf den Zeitpunkt 24 h
staemme_24 = [df_24[df_24['Stamm'] == s]['Ausbeute'].values for s in namen]
# Ausgabe: Stammname, Stichprobengröße, Mittelwert, Varianz (ddof=1)
for s, g in zip(namen, staemme_24):
    print(s, len(g), g.mean().round(3), g.var(ddof=1).round(3))

### Shapiro-Wilk-Test: Prüft die Normalverteilungsannahme pro Gruppe
# H0: Die Daten sind normalverteilt. Bei p > α kann H0 nicht abgelehnt werden → Normalverteilung angenommen.
for name, value in zip(namen, staemme_24):
    shapiro = stats.shapiro(value)
    print(f"Shapiro-Wilk Test for {name}: W = {shapiro.statistic:.4f}, p = {shapiro.pvalue:.4f}")
    print()

# Residuen = Messwert - Gruppenmittelwert
# Die Residuen sollten normalverteilt sein, wenn die ANOVA-Voraussetzung erfüllt ist.
res = np.concatenate([value - value.mean() for value in staemme_24])
print(f'Shapiro Residuen: p = {stats.shapiro(res).pvalue:.4f}')
# Q-Q-Plot (Quantile-Quantile-Plot) der Residuen: Bei Normalverteilung liegen die Punkte auf der Diagonalen
stats.probplot(res, plot=plt)
plt.show()

### Levene-Test: Prüft die Varianzhomogenität
# H0: Die Varianzen aller Gruppen sind gleich. Bei p > α kann H0 nicht abgelehnt werden → Varianzhomogenität angenommen.
levene = stats.levene(*staemme_24, center='mean')
print(f"Levene: p = {levene.pvalue:.4f}")

### ANOVA — manuelle Berechnung der Varianzanalyse

"""
Die ANOVA zerlegt die Gesamtvarianz in:
- Varianz zwischen den Gruppen (Faktor) — wird durch den Einflussfaktor (Stamm) erklärt
- Varianz innerhalb der Gruppen (Fehler) — zufällige Streuung

Effektgröße eta² (η²): Anteil der Gesamtvarianz, der durch den Faktor erklärt wird.
η² = QS_Faktor / (QS_Faktor + QS_Fehler) = QS_Faktor / QS_Gesamt
"""

# Gesamtmittelwert — Mittelwert aller Messwerte bei 24 h
mean_ges = staemme_df.loc[staemme_df['Zeit'] == 24, 'Ausbeute'].mean()

# Quadratsumme Faktor (zwischen den Gruppen): Misst die Streuung der Gruppenmittelwerte um den Gesamtmittelwert
# QS_Faktor = Σ n_i · (x̄_i - x̄_ges)²
qs_faktor = sum(len(value) * (value.mean() - mean_ges)**2 for value in staemme_24)
print(f"QS Faktor: {qs_faktor:.4f}")

# Quadratsumme Fehler (innerhalb der Gruppen): Misst die Streuung innerhalb jeder Gruppe um den Gruppenmittelwert
# QS_Fehler = Σ Σ (x_ij - x̄_i)²
qs_fehler = sum(((value - value.mean())**2).sum() for value in staemme_24)
print(f"QS Fehler: {qs_fehler:.4f}")

# Berechnung der Freiheitsgrade
# k = Anzahl der Gruppen (Stämme), N = Gesamtstichprobengröße
k, N = len(namen), sum(len(s) for s in staemme_24)
dof_faktor = k - 1      # Freiheitsgrade zwischen Gruppen
dof_fehler = N - k      # Freiheitsgrade innerhalb Gruppen
print(f"Freiheitsgrade Faktor: {dof_faktor}, Freiheitsgrade Fehler: {dof_fehler}")

# Berechnung der mittleren Quadratsummen
# MQ = QS / df — normalisiert die Quadratsumme durch die Freiheitsgrade
mq_faktor = qs_faktor / dof_faktor
mq_fehler = qs_fehler / dof_fehler
print(f"MQ Faktor: {mq_faktor:.4f}, MQ Fehler: {mq_fehler:.4f}")

# F-Wert: Verhältnis der Varianz zwischen Gruppen zur Varianz innerhalb Gruppen
# Unter H0 sollte F ≈ 1 sein. Ein großes F deutet auf Unterschiede zwischen Gruppen hin.
f = mq_faktor / mq_fehler
print(f"f-Wert: {f:.4f}")


p = stats.f.sf(f, dof_faktor, dof_fehler) #hier sf statt cdf, da p-Wert sehr klein ist und cdf damit Probleme hat
print(f"p-Wert: {p:.4e}")

# Effektgröße η² (Eta-Quadrat): Anteil der Gesamtvarianz, der durch den Faktor erklärt wird
eta2 = qs_faktor / (qs_faktor + qs_fehler)
print(f"η²: {eta2:.4f}")

# Kontrolle: scipy.stats.f_oneway berechnet ANOVA automatisch — dient als Plausibilitätscheck
anova = stats.f_oneway(*staemme_24)
anova_p = anova.pvalue
anova_f = anova.statistic
print(f"ANOVA: f = {anova_f:.4f}, p = {anova_p:.4e}")

# Interpretation: Bei α = 0.01 und p < 0.01 wird H0 abgelehnt → es gibt signifikante
# Unterschiede zwischen mindestens zwei Stämmen. Dann ist ein Post-hoc-Test nötig.

### Tukey-HSD (Honestly Significant Difference) — Post-hoc-Test
# Wenn die ANOVA signifikant ist, zeigt der Tukey-HSD-Test, welche Paarvergleiche signifikant sind.
tukey = smm.pairwise_tukeyhsd(endog=df_24.loc[:, 'Ausbeute'], groups=df_24['Stamm'], alpha=0.01)
print(tukey)

print()
print('BL21 Gold, BL21 Rosetta und C41 weichen nicht signifikant voneinander ab')
print()
print('Zusammenfassend zeigt sich eine absteigende Tendenz der Ausbeute:')
print('BL21 (höchste) > BL21 Gold ≈ BL21 Rosetta ≈ C41 > LOBSTR (niedrigste).')
# Gold/Rosetta/C41 nicht signifikant voneinander abweichend (p-adj)
# BL21 > Gold/Rosetta/C41 > LOBSTR (meandiff)