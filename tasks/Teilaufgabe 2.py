# Anknüpfend an die vorherige Fragestellung wird nun für das gleiche Enzym (in TB Medium) getestet,
# welche Einflüsse die Faktoren verwendeter Expressionsstamm (E. coli BL21, LOBSTR, BL21 Rosetta, C41, BL21 Gold)
# und Expressionsdauer (bis zu 48 h) auf die Ausbeute haben. Daten hierzu finden sich im Tab ‚production‘.
from math import log

import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy
import scipy.stats as stats
import math
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.multicomp as smm
import statsmodels.formula.api as smf
from matplotlib.ticker import MultipleLocator

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

#Plot
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for stamm, daten in stamm.items():
    m = daten.mean()
    s = daten.std()
    ax.errorbar(m.index, m.values, yerr=s.values, marker='o', capsize=3, label=stamm)
ax.set_xlabel('Expressionsdauer (h)')
ax.set_ylabel('Ausbeute (mg/L)')
ax.set_xticks([6, 12, 18, 24, 48])
ax.set_ylim(0,7)
ax.legend(title='Expressionsstamm', frameon=True, fontsize='small', title_fontsize='small')
#fig.savefig('plots/a2_verlauf.svg', bbox_inches='tight')
plt.show()

#Boxplot
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
for ax, z in zip(axes, zeiten):
    daten = [staemme_df[(staemme_df.Zeit == z) & (staemme_df.Stamm == s)]['Ausbeute'].dropna() for s in namen]
    ax.boxplot(daten, tick_labels=namen)
    ax.set_title(f'{z} h'); ax.tick_params(axis='x', rotation=60)
axes[0].set_ylabel('Ausbeute (mg/L)')
fig.tight_layout()
#fig.savefig('plots/a2_boxplots.svg')
plt.show()

print('Mittelwerte & Varianzen')
print(staemme_df.groupby(['Stamm', 'Zeit'])['Ausbeute'].agg(['mean', 'var']))



# 2.ii
# Berechnen Sie für jeden Stamm und jeden Zeitpunkt 95%-Konfidenzintervalle für die mittlere Ausbeute!
alpha = 0.05                    #α
confidence = 1 - alpha          #1-α  -Konfidenzniveau

print('95%-Konfidenzintervalle für die mittlere Ausbeute:')

teile = []
for (stamm, zeit), values in staemme_df.groupby(['Stamm', 'Zeit'])['Ausbeute']:
    mean = values.mean()
    n = len(values)
    dof = n - 1
    s = values.std(ddof=1)
    sem = s / np.sqrt(n)
    q = stats.t.ppf(1-alpha/2, df=dof)
    g0, g1 = stats.t.interval(1 - alpha, df=dof, loc=mean, scale=sem)
    print(f"{stamm}, {zeit}: Quantil t_(ν,1-α/2): {q:.4f}")
    print(f"{stamm}, {zeit}: {(1 - alpha) * 100:.0f}%-Konfidenzintervall: [{g0:.4f}, {g1:.4f}]")
    teile.append({'Stamm': stamm, 'Zeit': zeit, 'Mittelwert': mean, 'g0': g0, 'g1': g1})
ki_df = pd.DataFrame(teile).set_index(['Stamm', 'Zeit'])
teile = sorted(staemme_df['Zeit'].unique())

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
for name in namen:
    k = ki_df.loc[name]
    ax.errorbar(k.index, k['Mittelwert'], yerr=[k['Mittelwert'] - k['g0'], k['g1'] - k['Mittelwert']], marker='o', capsize=3, label=name)
ax.set_xlabel('Expressionsdauer (h)')
ax.set_ylabel('Ausbeute (mg/L), Mittel ± 95-%-KI')
ax.set_xticks(zeiten)
ax.set_ylim(0,7)
ax.legend(title='Expressionsstamm', frameon=True, fontsize='small', title_fontsize='small')
fig.savefig('plots/a2_verlauf_KI.svg', bbox_inches='tight')
plt.show()

# 2.iii
# Führen Sie eine einfaktorielle Varianzanalyse für den Einflussfaktor Expressionsstamm durch und
# legen dabei die Messwerte beim Zeitpunkt 24 h zugrunde! Testen Sie auf einem Konfidenzniveau von α = 0.01,
# ob ein Einfluss dieses Faktors auf die Ausbeute an Enzym besteht! Falls dieser Test positiv ausfällt,
# führen Sie nun einen adäquaten post hoc-Test durch, um festzustellen, welche Stufe(n) des Faktors
# hier maßgeblich sind/ist!

# Voraussetzung:
# 1. Feste Stufen (fest gewählte Stämme) -> Modell 1
# 2. unabhängige, zufällige Fehler e_ij
# 3. Normalverteilung der Messwerte
# 4. Varianzhomogenität

df_24 = staemme_df[staemme_df['Zeit'] == 24] #temporär, um alle Daten auf 24h zu reduzieren
staemme_24 = [df_24[df_24['Stamm'] == s]['Ausbeute'].values for s in namen]
for s, g in zip(namen, staemme_24):
    print(s, len(g), g.mean().round(3), g.var(ddof=1).round(3))

### Shapiro-Wilk
#print(staemme_df)
for name, value in zip(namen, staemme_24):
    shapiro = stats.shapiro(value)
    print(f"Shapiro-Wilk Test for {name}: W = {shapiro.statistic:.4f}, p = {shapiro.pvalue:.4f}")
    print()

# Residues = Messwert - Gruppenmittel
res = np.concatenate([value - value.mean() for value in staemme_24])
print(f'Shapiro Residuen: p = {stats.shapiro(res).pvalue:.4f}')
stats.probplot(res, plot=plt)
plt.show()

### Levene
levene = stats.levene(*staemme_24, center='mean')
print(f"Levene: p = {levene.pvalue:.4f}")

### ANOVA
anova = stats.f_oneway(*staemme_24)
anova_p = anova.pvalue
anova_f = anova.statistic
print(f"ANOVA: f = {anova_f:.4f}, p = {anova_p:.4f}")

mean_ges = staemme_df.loc[staemme_df['Zeit'] == 24, 'Ausbeute'].mean()
print(mean_ges)

qs_faktor = sum(len(value) * (value.mean() - mean_ges)**2 for value in staemme_24)
print(f"QS Faktor: {qs_faktor:.4f}")
qs_fehler = sum(((value - value.mean())**2).sum() for value in staemme_24)
print(f"QS Fehler: {qs_fehler:.4f}")

#Berechnung der Freiheitsgrade
frei_faktor = len(namen) - 1 #freiheitsgrade_faktor =
frei_fehler = len(namen) * (len(staemme_df['Zeit'].unique()) - 1) #freiheitsgrade_fehler0

#Berechnung der mittleren Quadratsummen:
mq_faktor = qs_faktor / frei_faktor
mq_fehler = qs_fehler / frei_fehler

#stats.pairwise_tukeyhsd(ki_df.loc[:, 'Mittelwert'], ki_df.index.get_level_values('Stamm'), alpha=0.01)


# 2.iv
# Führen Sie nun eine zweifaktorielle Varianzanalyse durch und beantworten auf einem Konfidenzniveau
# von α = 0.05, ob die Einflußfaktoren Expressionsstamm und Expressionsdauer alleine oder in Kombination
# signifikant die Ausbeute beeinflussen! Falls ja, führen Sie eine Analyse der einzelnen Faktorstufen durch!



