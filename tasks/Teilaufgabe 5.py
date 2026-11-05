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

ergebnisse = []
for S in konz:
    for rep in [1, 2, 3]:
        daten = get(S, rep)
        t = daten["Time (min)"].values
        P = daten["Product (nM)"].values
        fit = stats.linregress(t, P)
        ergebnisse.append({'S': S, 'rep': rep,
            'v0': fit.slope, 'se_v0': fit.stderr,
            'P0': fit.intercept, 'se_P0': fit.intercept_stderr,
            'R2': fit.rvalue**2})
v = pd.DataFrame(ergebnisse)
print(v.round(3))


# 5.ii
# Tragen Sie nun die soeben bestimmten Anfangsgeschwindigkeiten v0 gegen die Substratkonzentration [S]
# auf! Werten Sie die Daten gemäß der Michaelis-Menten Kinetik aus und bestimmen somit die besten Werte
# für vmax und KM sowie die zugehörigen Standardfehler! Bewerten Sie die Güte der Kurvenanpassung!

def mm(S, vmax, KM):
    return vmax * S / (KM + S)
p0 = [v['v0'].max(), 30.0]

param, covar = optimize.curve_fit(mm, v['S'], v['v0'], p0=p0)
se = np.sqrt(np.diag(covar))
vmax, KM = param
print(f'v_max = {vmax:.2f} ± {se[0]:.2f} nM/min, K_M = {KM:.1f} ± {se[1]:.1f} μM')

N_pkt = len(v); nu = N_pkt - len(param)
t_q = stats.t.ppf(0.975, df=nu)
print('95%-KI v_max:', vmax - t_q*se[0], vmax + t_q*se[0])
print('95%-KI K_M: ', KM - t_q*se[1], KM + t_q*se[1])
print('Korrelation v_max/K_M:', covar[0, 1] / (se[0] * se[1]))

v['res'] = v['v0'] - mm(v['S'], *param)
chi2_red = (v['res']**2).sum() / nu # VL11, F34
var_rep = v.groupby('S')['v0'].var(ddof=1).mean() # gepoolte Replikatvarianz
print('red. χ² =', round(chi2_red, 3), ' Replikatvarianz =', round(var_rep, 3))
print('Shapiro Residuen:', stats.shapiro(v['res']))

S = np.linspace(0, v['S'].max(), 300)
fig, ax = plt.subplots(figsize=(9, 6))
v_mean = v.groupby('S')['v0'].agg(['mean', 'std', 'count']).reset_index()
ax.errorbar(v_mean['S'], v_mean['mean'], yerr=(v_mean['std']/np.sqrt(v_mean['count'])), fmt='o', ms=4, capsize=3, label=r'$\overline{v_0}$ (± SF der Replikate)')
ax.plot(S, mm(S, *param), 'r-', label='Michaelis-Menten-Fit')
ax.axhline(vmax, color='grey', lw=1)
ax.axhline(vmax / 2, color='grey', lw=1)
ax.vlines(KM, 0, vmax / 2, color='grey', lw=1)
ax_r = ax.secondary_yaxis('right')
ax_r.set_yticks([vmax / 2, vmax])
ax_r.set_yticklabels([rf'$\frac{{1}}{{2}}, v_{{max}}$ = {vmax / 2:.1f}', rf'$v_{{max}}$ = {vmax:.1f}'])
ax_b = ax.secondary_xaxis('bottom')
#ax_b.tick_params(labelrotation=45)
plt.setp(ax_b.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
ax_b.set_xticks([KM])
ax_b.set_xticklabels([rf'$K_M$ = {KM:.1f} μM'])
ax_b.set_xlabel('[S] (μM)')
ax.set_ylim(0, vmax+1)
ax.set_xlim(0, v_mean['S'].max()+1)
ax.set_ylabel('v₀ (nM/min)')
ax.legend()
ax.set_title('Michaelis-Menten-Fit')
fig.tight_layout()
fig.savefig('plots/a5_michaelis_menten.svg')
plt.show()


pw, cw = optimize.curve_fit(mm, v['S'], v['v0'], p0=p0, sigma=v['se_v0'])
print(pw, np.sqrt(np.diag(cw)))


# 5.iii
# Transformieren Sie die Daten nun in die doppelt-reziproke Form nach Lineweaver-Burk!
# Tragen Sie die Daten auf und führen einen Fit von 1 / v0 gegen 1 / [S] auf.
# Unterscheiden sich die erhaltenen Werte für die Parameter vmax und KM von den zuvor bestimmten?
print(f'v_mean: {v_mean}')


v_mean = (v.groupby('S')
            .agg(v0=('v0', 'mean'),
                 std=('v0', 'std'),
                 n=('v0', 'count'),
                 se_fit=('se_v0', lambda s: np.sqrt((s**2).sum()) / len(s)),  # fortgepflanzter Fitfehler
                 P0=('P0', 'mean'),
                 R2=('R2', 'mean'))
            .reset_index())

v_mean['se_v0'] = v_mean['std'] / np.sqrt(v_mean['n'])

#x = 1 / v_mean['S']
x = 1 / v['S']
#y = 1 / v_mean['v0']
y = 1 / v['v0']
lb = stats.linregress(x, y)
vmax_lb = 1 / lb.intercept
KM_lb = lb.slope / lb.intercept
print(f'LB: v_max = {vmax_lb:.2f} nM/min, K_M = {KM_lb:.1f} μM, R² = {lb.rvalue**2:.3f}')


"""
def gerade(x, a, b):
    return a + b * x
"""

"""
(a, b), cov_ab = optimize.curve_fit(gerade, x, y)
J = np.array([[-1 / a**2, 0], # Ableitungen von v_max = 1/a nach a, b
[-b / a**2, 1 / a]]) # Ableitungen von K_M = b/a nach a, b
cov_par = J @ cov_ab @ J.T
se_vmax_lb, se_KM_lb = np.sqrt(np.diag(cov_par))
print(f'v_max = {1/a:.2f} ± {se_vmax_lb:.2f}, K_M = {b/a:.1f} ± {se_KM_lb:.1f}')
"""
sy = v_mean['se_v0'] / v_mean['v0']**2 # VL07, F27: σ(1/v0) = σ(v0)/v0²
fig, ax = plt.subplots(figsize=(9, 6))
xf = np.linspace(0, x.max(), 100)
ax.plot(xf, lb.intercept + lb.slope * xf, 'r-', label='Lineare Regression')
ax.errorbar(x, y, yerr=sy, fmt='o', ms=3, capsize=2, label=r'$\frac{1}{\overline{v_0}}$ (± SF der Replikate)')
ax.set_xlabel('1/[S] (1/μM)'); ax.set_ylabel('1/v₀ (min/nM)')
ax.set_ylim(0, 1.1 * ax.get_ylim()[1])
ax.set_xlim(-(1/KM_lb), ax.get_xlim()[1])
ax.legend()
ax.set_title('Lineweaver-Burk-Plot')
fig.tight_layout()
fig.savefig('plots/a5_lineweaver_burk.svg')
plt.show()


# 5.iv
# Falls die vorherige Frage bejaht wurde, diskutieren Sie, woran das liegen und wie dem zu begegnen sein
# könnte. Was würden Sie einem neuen Studenten als Vorgehen empfehlen, wenn sie/er ähnliche Experimente auswertet?



