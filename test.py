import matplotlib.pyplot as plt

pct_bb = [76.6, 78.9, 80.6, 82.7, 85.5]
pct_bb = [62.0,
66.1,
69.2,
76.1,
74.6]
year = [2017, 2018, 2019, 2020, 2021]

fig, ax = plt.subplots()

ax.plot(year, pct_bb, 'o-')
plt.show()
