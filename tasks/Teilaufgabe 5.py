# Michaelis-Menten-Kinetik eines Enzyms

# Nun soll die enzymatische Charakterisierung erfolgen. Dazu wird das Enzym mit
# einer variierenden Menge Substrats versetzt und über den Zeitraum von 10 Minuten
# die Bildung eines Produkts colorimetrisch verfolgt. Somit kann die jeweilig
# gebildete Menge Produkts quantifiziert werden, was in dem Tab ‚MnM‘ abgelegt ist.
# Für jede Substratkonzentration wurden drei biologisch unabhängige Replikate
# durchgeführt.

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

df = pd.read_csv(
    "data/MnM.csv",
    sep=";",
    header=None,
    dtype=str,
    encoding="cp1252"
)

# Substratkonzentrationen in µM: 13 verschiedene Werte von 2 bis 110 µM
konz = [2, 4, 8, 16, 24, 32, 39, 47, 56, 66, 77, 90, 110]


def get(c, rep):
    i = konz.index(c)                     # Spalten-Index der Konzentration
    z = 5 + 10 * (rep - 1)               # Startzeile des Replikats
    out = df.iloc[z:z + 6, 3 * i:3 * i + 2].apply(
        lambda s: s.str.replace(",", ".").astype(float)
    )
    out.columns = ["Time (min)", "Product (nM)"]
    return out.reset_index(drop=True)


# Testausgabe: Konzentration 4 µM, Replikat 2
print(get(4, 2))


# Bearbeiten Sie nun folgende Aufgaben und Fragen:



# 5.i
# Bestimmen Sie für jede Substratkonzentration und jedes Replikat die Anfangsgeschwindigkeit v0
# der Reaktion! Auch wenn es sich hierbei um einen linearen Fit handelt, kann dennoch auch die nicht-lineare
# Regressionsanalyse eingesetzt werden.


"""
In der Anfangsphase einer enzymatischen Reaktion (bei niedriger Produkt-
konzentration) steigt die Produktmenge linear mit der Zeit an. Die Steigung
dieser Geraden ist die Anfangsgeschwindigkeit v0.

P(t) = P0 + v0 · t

Mit scipy.stats.linregress() wird eine lineare Regression durchgeführt,
die Steigung (slope = v0), Standardfehler (stderr), Achsenabschnitt
(intercept = P0) und den Bestimmtheitskoeffizienten (R²) liefert.

Hinweis: Auch eine nicht-lineare Regression wäre möglich, aber für eine
Gerade ist die lineare Regression der einfachste und robusteste Ansatz.
"""

ergebnisse = []
for S in konz:
    for rep in [1, 2, 3]:
        daten = get(S, rep)
        t = daten["Time (min)"].values       # Zeit in Minuten
        P = daten["Product (nM)"].values     # Produktkonzentration in nM
        fit = stats.linregress(t, P)         # Lineare Regression: P = P0 + v0·t
        ergebnisse.append({
            'S': S,
            'rep': rep,
            'v0': fit.slope,                 # Anfangsgeschwindigkeit (nM/min)
            'se_v0': fit.stderr,             # Standardfehler der Steigung
            'P0': fit.intercept,             # Achsenabschnitt (Produkt bei t=0)
            'se_P0': fit.intercept_stderr,   # Standardfehler des Achsenabschnitts
            'R2': fit.rvalue**2              # Bestimmtheitskoeffizient (Qualität des linearen Fits)
        })

v = pd.DataFrame(ergebnisse)
print(v.round(3))

G3, G, G2 = 'blue', 'orange', 'lime'

fig, ax = plt.subplots(figsize=(6.3, 4.55), dpi=220)
for S, farbe in [(110, G3), (32, G), (4, G2)]:
    steigungen = []
    for rep, marker in zip([1, 2, 3], ['o', 's', '^']):
        d = get(S, rep)
        t = d["Time (min)"]
        P = d["Product (nM)"]
        fit = stats.linregress(t, P)
        steigungen.append(fit.slope)
        ax.plot(t, P, marker, color=farbe, ms=5, mec='white', mew=0.5)
        tt = np.array([0, 10.5])                            # Gerade bis knapp über 10 min
        ax.plot(tt, fit.intercept + fit.slope * tt, '-', color=farbe, lw=1.2, alpha=0.8)
    # Beschriftung rechts neben den Geraden, auf Höhe der mittleren Steigung
    ax.text(10.7, np.mean(steigungen) * 10.5, f'{S} µM',
            color=farbe, va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0, 12.6)
ax.set_ylim(bottom=-5)
ax.set_xlabel('t (min)')
ax.set_ylabel('[P] (nM)')
ax.set_title('Produktbildung, 3 von 13 Substratkonzentrationen', loc='left')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
#fig.savefig('plots/a5_progress_beispiele.svg')
#plt.show()
plt.close(fig)


## 5.ii
# Tragen Sie nun die soeben bestimmten Anfangsgeschwindigkeiten v0 gegen die Substratkonzentration [S]
# auf! Werten Sie die Daten gemäß der Michaelis-Menten Kinetik aus und bestimmen somit die besten Werte
# für v_max und KM sowie die zugehörigen Standardfehler! Bewerten Sie die Güte der Kurvenanpassung!


def mm(S, v_max, KM):
    """Michaelis-Menten-Gleichung: v0 = v_max · S / (KM + S)"""
    return v_max * S / (KM + S)

# Mittelwerte und Standardfehler pro Substratkonzentration berechnen
v_mean = (v.groupby('S')
          .agg(v0=('v0', 'mean'),
               std=('v0', 'std'),
               n=('v0', 'count'))
          .reset_index())
v_mean['se_v0'] = v_mean['std'] / np.sqrt(v_mean['n'])  # Standardfehler des Mittels

# Startwerte für die nicht-lineare Optimierung
p0 = [v['v0'].max(), 30.0]

# Nicht-linearer Fit der Einzelwerte
param, covar = optimize.curve_fit(
    mm, v['S'], v['v0'],
    p0=p0,
# hier ggf. Gewichtung nicht sinnvoll, da nur wenige Punkte vorhanden sind. Hier trotzdem die Formel
    # sigma=v_mean['se_v0'], absolute_sigma=True
)
se = np.sqrt(np.diag(covar))    # Standardfehler der Parameter
v_max, KM = param
print(f'v_max = {v_max:.2f} ± {se[0]:.2f} nM/min, K_M = {KM:.1f} ± {se[1]:.1f} μM')


# 95%-Konfidenzintervalle
N_pkt = len(v)
nu = N_pkt - len(param)             # Freiheitsgrade
t_q = stats.t.ppf(0.975, df=nu)  # t-Quantil für 95% Konfidenz
print('95%-KI v_max:', v_max - t_q * se[0], v_max + t_q * se[0])
print('95%-KI K_M: ', KM - t_q * se[1], KM + t_q * se[1])

# Korrelation zwischen v_max und KM – hohe Korrelation bedeutet, dass die
# Parameter stark voneinander abhängen und unsicher bestimmt sind
print('Korrelation ρ² (v_max/K_M):', covar[0, 1] / (se[0] * se[1]))


# Güte der Kurvenanpassung
# Residuen = Differenz zwischen gemessenen und angepassten Werten
# Hinweis: Da nur 3 Replikate pro Konzentration vorliegen, ist die Varianzschätzung
# aus den Replikaten zu ungenau für eine sinnvolle Gewichtung.
# Daher wird χ² ungewichtet berechnet — aber auf ALLEN 39 Einzelwerten,
# um eine zuverlässige Schätzung der Restvarianz zu erhalten.

# Residuen aller 39 Einzelwerte und 13 Mittelwerte (für χ²-Berechnung)
v['v_fit'] = mm(v['S'], *param)
v['res'] = v['v0'] - v['v_fit']

v_mean['v_fit'] = mm(v_mean['S'], *param)
v_mean['res'] = v_mean['v0'] - v_mean['v_fit']


# Chi-Quadrat aus allen Einzelwerten
N_einzel = len(v)                     # 39 Einzelwerte
nu_einzel = N_einzel - len(param)     # Freiheitsgrade: 37
chi2 = (v['res']**2).sum()            # Summe der quadrierten Residuen (ungewichtet)
chi2_red = chi2 / nu_einzel           # Reduziertes Chi-Quadrat = Schätzung der Restvarianz
print(f'χ² = {chi2:.3f}')
print(f'red. χ² = {chi2_red:.3f}')
print(f'(berechnet aus {N_einzel} Einzelwerten, ν = {nu_einzel})')


# Chi-Quadrat aus allen Mittelwerten
N_mittel = len(v_mean)                     # 13 Mittelwerte
nu_mittel = N_mittel - len(param)     # Freiheitsgrade
chi2 = (v_mean['res']**2).sum()            # Summe der quadrierten Residuen (ungewichtet)
chi2_red = chi2 / nu_mittel           # Reduziertes Chi-Quadrat = Schätzung der Restvarianz
print(f'χ² = {chi2:.3f}')
print(f'red. χ² = {chi2_red:.3f}')
print(f'(berechnet aus {N_mittel} Mittelwerten, ν = {nu_mittel})')






# Shapiro-Wilk-Test auf Normalverteilung der Residuen
# p > 0.05 → kein Hinweis gegen
print('Shapiro Residuen (Mittelwerte):', stats.shapiro(v['res']))

# === Plot: Michaelis-Menten-Kurve mit Fit ===
S_plot = np.linspace(0, v['S'].max(), 300)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Datenpunkte mit Fehlerbalken (Standardfehler des Mittels)
#ax1.errorbar(
#    v['S'], v['v0'],
#    yerr=v['se_v0'],
#    fmt='o', ms=4, capsize=3,
#    label=r'$\overline{v_0}$ (± SEM)'
#)
ax1.scatter(v['S'], v['v0'], label=r'$v_0$')

# Angepasste Michaelis-Menten-Kurve
ax1.plot(S_plot, mm(S_plot, *param), 'r-', label='Michaelis-Menten-Fit')

# Hilfslinien für v_max und v_max/2
ax1.axhline(v_max, color='grey', lw=1, linestyle='--')
ax1.axhline(v_max / 2, color='grey', lw=1, linestyle='--')
ax1.vlines(KM, 0, v_max / 2, color='grey', lw=1, linestyle='--')

# Sekundäre Achsen für v_max, v_max/2 und KM
ax_r = ax1.secondary_yaxis('right')
ax_r.set_yticks([v_max / 2, v_max])
ax_r.set_yticklabels([rf'$\frac{{1}}{{2}} v_{{max}}$ = {v_max / 2:.1f}',
                       rf'$v_{{max}}$ = {v_max:.1f}'])
ax1_b = ax1.secondary_xaxis('bottom')
#plt.setp(ax_b.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
ax1_b.set_xticks([KM])

ticks = ax1_b.xaxis.get_major_ticks()
t = ticks[0].tick1line
t.set_markersize(6)
t.set_markeredgewidth(2)

ax1_b.set_xticklabels([rf'$K_M$ = {KM:.1f} μM'])
#ax_b.set_xlabel('[S] (μM)')

ax1.set_ylim(0, v_max + 1)
ax1.set_xlim(0, v['S'].max() + 1)
ax1.set_ylabel('v₀ (nM/min)')
ax1.legend()
ax1.set_title('Michaelis-Menten-Fit')

ax2.scatter(v['S'], v['res'], s=15)
ax2.axhline(0, color='grey')
ax2.set_xlabel('[S] (μM)')
ax2.set_ylabel('Residuen')

fig.tight_layout()
fig.savefig('plots/a5_michaelis_menten.svg')
plt.close()

# 5.iii
# Transformieren Sie die Daten nun in die doppelt-reziproke Form nach Lineweaver-Burk!
# Tragen Sie die Daten auf und führen einen Fit von 1 / v0 gegen 1 / [S] auf.
# Unterscheiden sich die erhaltenen Werte für die Parameter v_max und KM von den zuvor bestimmten?


# Reziproke Transformation der MITTELWERTE
x = 1 / v['S']       # 1/[S] in 1/µM
y = 1 / v['v0']      # 1/v0 in min/nM

# Lineare Regression auf die transformierten Daten
lb = stats.linregress(x, y)

# Parameter aus Steigung und Achsenabschnitt berechnen
# 1/v0 = (KM/v_max) · (1/[S]) + 1/v_max
# Steigung = KM/v_max, Achsenabschnitt = 1/v_max
v_max_lb = 1 / lb.intercept        # v_max = 1 / Achsenabschnitt
KM_lb = lb.slope / lb.intercept   # KM = Steigung / Achsenabschnitt

print(f'Lineweaver-Burk:')
print(f'  v_max = {v_max_lb:.2f} nM/min')
print(f'  K_M   = {KM_lb:.1f} μM')
print(f'  R²    = {lb.rvalue**2:.4f}')
print(f'\nVergleich mit nicht-linearem Fit (5.ii):')
print(f'  v_max: {v_max:.2f} vs {v_max_lb:.2f} nM/min (Abweichung: {abs(v_max - v_max_lb)/v_max*100:.1f}%)')
print(f'  K_M:   {KM:.1f} vs {KM_lb:.1f} μM (Abweichung: {abs(KM - KM_lb)/KM*100:.1f}%)')

# === Fehlerfortpflanzung für Lineweaver-Burk-Parameter ===

# Die Standardfehler der transformierten Werte:
# σ(1/v0) = σ(v0) / v0²  (Fehlerfortpflanzung für reziproke Funktion)
sy = v['se_v0'] / v['v0']**2


# === Plot: Lineweaver-Burk-Diagramm ===
fig, ax = plt.subplots(figsize=(9, 6))

# Lineare Regression als rote Linie
#xf = np.linspace(0, x.max(), 100)
xf = np.linspace(-(1/KM_lb), x.max(), 100)
ax.plot(xf, lb.intercept + lb.slope * xf, 'r-', label='Lineare Regression')

# Datenpunkte mit Fehlerbalken
ax.errorbar(
    x, y, yerr=sy,
    fmt='o', ms=3, capsize=2,
    label=r'$1/v_0$ (± $σ_{1/v_0}$)'
)

# Achsenbeschriftungen und Titel
ax.set_xlabel('1/[S] (1/μM)')
ax.set_ylabel('1/v₀ (min/nM)')


# Achsen so wählen, dass der Achsenabschnitt sichtbar ist
ax.set_ylim(0, 1.1 * ax.get_ylim()[1])
ax.set_xlim(-(1 / KM_lb), ax.get_xlim()[1])

# Achsenbeschriftungen K_M und v_max
ax.scatter(0, 1/v_max_lb, label=r'$1/v_{max}$', color='red')
ax_x = ax.secondary_xaxis('bottom')
ax_x.set_xticks([-(1/KM_lb)])
ax_x.set_xticklabels([r"$-\frac{1}{K_M}$"])

ax_y = ax.secondary_yaxis('left')
ax_y.set_yticks([1/v_max_lb])
tick_labels = ax_y.set_yticklabels([r"$\frac{1}{v_{max}}$"])
# Offset der Beschriftung
dy = 8
offset = mtransforms.ScaledTranslation(0, dy/72, fig.dpi_scale_trans)
for label in ax_y.get_yticklabels():
    label.set_transform(label.get_transform() + offset)


# Achsen und Rahmen anpassen
ax.axvline(0, color='grey', linestyle='--')
#ax.vlines(-1/KM_lb, 0, 1/v_max_lb, color='grey', linestyle='--')
ax.hlines(1/v_max_lb, -1/KM_lb, 0, color='grey', linestyle='--')

ax.legend()
ax.set_title('Lineweaver-Burk-Plot')
fig.tight_layout()
fig.savefig('plots/a5_lineweaver_burk.svg')
plt.close()


# 5.iv
# Falls die vorherige Frage bejaht wurde, diskutieren Sie, woran das liegen und wie dem zu begegnen sein
# könnte. Was würden Sie einem neuen Studenten als Vorgehen empfehlen, wenn sie/er ähnliche Experimente auswertet?



