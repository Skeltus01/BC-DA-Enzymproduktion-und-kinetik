# Michaelis-Menten-Kinetik eines Enzyms

# Nun soll die enzymatische Charakterisierung erfolgen. Dazu wird das Enzym mit
# einer variierenden Menge Substrats versetzt und über den Zeitraum von 10 Minuten
# die Bildung eines Produkts colorimetrisch verfolgt. Somit kann die jeweilig
# gebildete Menge Produkts quantifiziert werden, was in dem Tab ‚MnM‘ abgelegt ist.
# Für jede Substratkonzentration wurden drei biologisch unabhängige Replikate
# durchgeführt.

import matplotlib.pyplot as plt
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
    """Liest die Messdaten für eine gegebene Substratkonzentration c (µM) und
    ein Replikat rep (1, 2 oder 3) aus der CSV-Datei aus.

    Die CSV-Struktur:
      - Spalten: pro Konzentration 3 Spalten (Zeit, Produkt, ggf. leer).
        Die Zeit-Spalte ist bei 3·i, die Produkt-Spalte bei 3·i+1.
      - Zeilen: Replikat 1 beginnt bei Zeile 5, Replikat 2 bei Zeile 15,
        Replikat 3 bei Zeile 25. Jedes Replikat hat 6 Zeilen Messwerte.
    """
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

# DataFrame mit allen 39 Ergebnissen (13 Konzentrationen × 3 Replikate)
v = pd.DataFrame(ergebnisse)
print(v.round(3))

# Qualitätskontrolle – Prüfen Sie, ob alle R²-Werte nahe bei 1 liegen.
# Niedrige R²-Werte deuten darauf hin, dass die Anfangsphase nicht linear ist
# oder dass Messfehler vorliegen.
# print(v[v['R2'] < 0.99][['S', 'rep', 'R2']])


## 5.ii
# Tragen Sie nun die soeben bestimmten Anfangsgeschwindigkeiten v0 gegen die Substratkonzentration [S]
# auf! Werten Sie die Daten gemäß der Michaelis-Menten Kinetik aus und bestimmen somit die besten Werte
# für v_max und KM sowie die zugehörigen Standardfehler! Bewerten Sie die Güte der Kurvenanpassung!


#   Die Michaelis-Menten-Gleichung ist ein nicht-lineares Modell:
#       v0 = v_max · [S] / (KM + [S])
#
#   Mit scipy.optimize.curve_fit() wird eine nicht-lineare Least-Squares-
#   Anpassung durchgeführt. Die Startwerte p0 sind wichtig, da der Algorithmus
#   iterativ arbeitet und von guten Startwerten abhängt.
#
#   v_max-Startwert: Der maximale gemessene v0-Wert (kann v_max nicht überschreiten)
#   KM-Startwert: 30 µM (grobe Schätzung aus dem Diagramm)
#
#   BLATT5 – WICHTIGER FEHLER:
#   Der erste curve_fit() Aufruf verwendet ALLE 39 Einzelwerte (jedes Replikat
#   separat), aber die Fehlerbalken im Plot zeigen nur die Mittelwerte!
#   Das ist inkonsistent: Entweder fittet man auf Mittelwerte pro Konzentration
#   (13 Punkte) oder man verwendet alle Einzelwerte mit Gewichtung (sigma).
#
#   Die bessere Vorgehensweise:
#   - Mittelwerte pro Konzentration bilden (13 Punkte)
#   - Mit sigma = Standardfehler des Mittels gewichten
#   - So erhält man korrekte Standardfehler für v_max und KM
# =============================================================================

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

# Nicht-linearer Fit auf Mittelwerte mit Gewichtung durch Standardfehler
# sigma=v_mean['se_v0'] gewichtet jeden Punkt nach seiner Unsicherheit
param, covar = optimize.curve_fit(
    mm, v_mean['S'], v_mean['v0'],
    p0=p0,
# hier ggf. Gewichtung nicht sinnvoll, da nur wenige Punkte vorhanden sind. Hier trotzdem die Formel
    # sigma=v_mean['se_v0'], absolute_sigma=True
)
se = np.sqrt(np.diag(covar))
v_max, KM = param
print(f'v_max = {v_max:.2f} ± {se[0]:.2f} nM/min, K_M = {KM:.1f} ± {se[1]:.1f} μM')


# 95%-Konfidenzintervalle
N_pkt = len(v_mean) #todo: zuvor len(v)
nu = N_pkt - len(param)             # Freiheitsgrade
t_q = stats.t.ppf(0.975, df=nu)  # t-Quantil für 95% Konfidenz
print('95%-KI v_max:', v_max - t_q * se[0], v_max + t_q * se[0])
print('95%-KI K_M: ', KM - t_q * se[1], KM + t_q * se[1])

# Korrelation zwischen v_max und KM – hohe Korrelation bedeutet, dass die
# Parameter stark voneinander abhängen und unsicher bestimmt sind
print('Korrelation v_max/K_M:', covar[0, 1] / (se[0] * se[1])) #todo: ρ²?


# Güte der Kurvenanpassung #todo: prüfen!
# Residuen = Differenz zwischen gemessenen und angepassten Werten
v_mean['res'] = v_mean['v0'] - mm(v_mean['S'], *param) #todo: zuvor v['res'] statt v_mean['res']

# Reduziertes Chi-Quadrat: Maß für die Güte der Anpassung
# χ²_red ≈ 1 bedeutet eine gute Anpassung
# χ²_red >> 1 bedeutet systematische Abweichungen oder unterschätzte Fehler
# χ²_red << 1 bedeutet überbewertete Fehler oder overfitting
#chi2_red = (v['res']**2).sum() / nu #todo: zuvor ← vs korrigiert ↓ - was macht den Unterschied?
chi2_red = (v_mean['res']**2 / v_mean['se_v0']**2).sum() / nu
print('red. χ² =', round(chi2_red, 3))

# Shapiro-Wilk-Test auf Normalverteilung der Residuen
# p > 0.05 → Residuen sind normalverteilt (Annahme der Regression erfüllt)
print('Shapiro Residuen:', stats.shapiro(v_mean['res']))

# === Plot: Michaelis-Menten-Kurve mit Fit ===
S_plot = np.linspace(0, v_mean['S'].max(), 300) #todo: zuvor v['S'] statt v_mean['S']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Datenpunkte mit Fehlerbalken (Standardfehler des Mittels)
ax1.errorbar(
    v_mean['S'], v_mean['v0'],
    yerr=v_mean['se_v0'],
    fmt='o', ms=4, capsize=3,
    label=r'$\overline{v_0}$ (± SEM)'
)

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
ax1.set_xlim(0, v_mean['S'].max() + 1)
ax1.set_ylabel('v₀ (nM/min)')
ax1.legend()
ax1.set_title('Michaelis-Menten-Fit')

ax2.scatter(v_mean['S'], v_mean['res'], s=15)
ax2.axhline(0, color='grey')
ax2.set_xlabel('[S] (μM)')
ax2.set_ylabel('Residuen')

fig.tight_layout()
fig.savefig('plots/a5_michaelis_menten.svg')
plt.show()

# Vergleich mit dem ursprünglichen (fehlerhaften) Ansatz
pw, cw = optimize.curve_fit(mm, v['S'], v['v0'], p0=p0, sigma=v['se_v0'])
print(f'\nVergleich – gewichteter Fit auf Einzelwerte:')
print(f'v_max = {pw[0]:.2f} ± {np.sqrt(cw[0,0]):.2f} nM/min')
print(f'K_M   = {pw[1]:.1f} ± {np.sqrt(cw[1,1]):.1f} μM')



# 5.iii
# Transformieren Sie die Daten nun in die doppelt-reziproke Form nach Lineweaver-Burk!
# Tragen Sie die Daten auf und führen einen Fit von 1 / v0 gegen 1 / [S] auf.
# Unterscheiden sich die erhaltenen Werte für die Parameter v_max und KM von den zuvor bestimmten?
#
#   Die Lineweaver-Burk-Gleichung ist die doppelt-reziproke Form der
#   Michaelis-Menten-Gleichung:
#
#       1/v0 = (KM / v_max) · (1/[S]) + 1/v_max
#
#   Dies entspricht einer Geradengleichung y = a·x + b mit:
#       Steigung a = KM / v_max
#       Achsenabschnitt b = 1 / v_max
#
#   Daraus folgt:
#       v_max = 1 / b
#       KM   = a / b

# Reziproke Transformation der MITTELWERTE
x = 1 / v_mean['S']       # 1/[S] in 1/µM #todo: warum hier Mittelwerte und nicht Einzelwerte
y = 1 / v_mean['v0']      # 1/v0 in min/nM

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
sy = v_mean['se_v0'] / v_mean['v0']**2 #todo: was bedeutet sy?


# === Plot: Lineweaver-Burk-Diagramm ===
fig, ax = plt.subplots(figsize=(9, 6))

# Lineare Regression als rote Linie
xf = np.linspace(0, x.max(), 100)
ax.plot(xf, lb.intercept + lb.slope * xf, 'r-', label='Lineare Regression')

# Datenpunkte mit Fehlerbalken
ax.errorbar(
    x, y, yerr=sy,
    fmt='o', ms=3, capsize=2,
    label=r'$1/\overline{v_0}$ (± SEM)'
)

# Achsenbeschriftungen und Titel
ax.set_xlabel('1/[S] (1/μM)')
ax.set_ylabel('1/v₀ (min/nM)')

# Achsen so wählen, dass der Achsenabschnitt sichtbar ist
ax.set_ylim(0, 1.1 * ax.get_ylim()[1])
ax.set_xlim(-(1 / KM_lb), ax.get_xlim()[1])

ax.legend()
ax.set_title('Lineweaver-Burk-Plot')
fig.tight_layout()
fig.savefig('plots/a5_lineweaver_burk.svg')
plt.close()


# 5.iv
# Falls die vorherige Frage bejaht wurde, diskutieren Sie, woran das liegen und wie dem zu begegnen sein
# könnte. Was würden Sie einem neuen Studenten als Vorgehen empfehlen, wenn sie/er ähnliche Experimente auswertet?



