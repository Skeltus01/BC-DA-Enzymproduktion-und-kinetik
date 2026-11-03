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

personen = ['Jim', 'Jane', 'PI', 'Jacky']
daten = [person_df[person_df["Person"] == p]['Ausbeute'].values for p in personen]
fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot(daten, tick_labels=personen)
for i, x in enumerate(daten, start=1):
    ax.scatter(np.random.normal(i, 0.05, len(x)), x, s=15, alpha=0.6)
ax.set_ylabel('Ausbeute (mg/L)')
ax.set_title('BL21, TB, 24 h')
fig.savefig('plots/a3_boxplot.svg')
plt.show()

kenn = person_df.groupby('Person').agg(['mean', 'var'])
print(kenn)



# 3.ii
# Führen Sie eine einfaktorielle Varianzanalyse nach Modell II durch und beantworten auf einem
# Konfidenzniveau von α = 0.05 die Frage, ob ein signifikanter Einfluss des zufälligen Einflussfaktors Experimentator
# auf die erzielten Ausbeuten des Enzyms besteht!



# 3.iii
# Falls der Test bei der vorherigen Frage positiv ausfällt, diskutieren Sie, wie zu testen wäre,
# welche Stufe (d.h. welche Person) diesen signifikanten Einfluss ausübt!
# (Eine Lösung kann entweder auf Basis der vorhandenen Daten erfolgen [das ist anspruchsvoll und am besten mit Python adressierbar],
# oder Sie schlagen ein neues Experiment vor, mit dem eine Beantwortung möglich wäre.)



