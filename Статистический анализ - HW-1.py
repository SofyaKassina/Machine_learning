import pandas as pd
from sklearn import datasets
from scipy.stats import f_oneway
from collections import Counter
import seaborn as sns

boston_data = datasets.load_boston()
df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
df_boston['target'] = pd.Series(boston_data.target)
df_boston.head()

data = df_boston.drop (columns = ['CHAS', 'ZN', 'AGE', 'RAD', 'B', 'PTRATIO', 'TAX',
                                  'RM', 'INDUS', 'LSTAT', 'target'])
dir(data)

data_1, data_2, data_3  = data['CRIM'], data['NOX'], data['DIS']
stat, p = f_oneway(data_1, data_2, data_3)
print("Анализ дисперсионного теста (ANOVA)= ", stat)
print("P_value = ", p)

def mean(x:float) -> float:
    return sum(x)/ len(x)

def mode(x: float) -> float:
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

def median_odd(x:float) -> float:
    "если len(x) явдяется нечетной "
    return sorted(x)[len(x) // 2]

def median_even(x:float) -> float:
    "если len(x) явдяется четной "
    sorted_x = sorted(x)
    hi_midpoint = len(x) // 2
    return (sorted_x[hi_midpoint - 1] + sorted_x[hi_midpoint]) / 2

def median(v: float) -> float:
    return median_even(v) if len(v) % 2 == 0 else median_odd(v)


print('Значение среднего арифметического для первого, второго, третьего признаков = ', mean(data_1), mean(data_2), mean(data_3))
print('Мода для первого, второго, третьего признаков = ', mode(data_1), mode(data_2), mode(data_3))
print('Медиана для первого, второго, третьего признаков = ', median(data_1), median(data_2), median(data_3))

print('Стандартное отклонение для первого, второго, третьего признаков = ',data_1.std(), data_2.std(),data_3.std())

hist = data.hist(bins=3)
ax = data.plot.kde()
sns.heatmap(data)
