# Statistical Tests For Checking Normality

"""# import required packages"""

# 1. Shapiro-Wilk Test
import numpy as np
from scipy import stats

data = np.random.normal(0, 1, 100)
stat, p = stats.shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
  ###########################################################

# 2. Kolmogorov-Smirnov Test
from scipy.stats import kstest

data = np.random.normal(0, 1, 100)
stat, p = kstest(data, 'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
##############################################################
# 3. Anderson-Darling Test
from scipy.stats import anderson

data = np.random.normal(0, 1, 100)
result = anderson(data)
print('Statistic: %.3f' % result.statistic)
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
######################################################################
#  4. Lilliefors Test
from statsmodels.stats.diagnostic import lilliefors

data = np.random.normal(0, 1, 100)
stat, p = lilliefors(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
########################################################################
# 5. D'Agostino's K-squared Test
from scipy.stats import normaltest

data = np.random.normal(0, 1, 100)
stat, p = normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
#########################################################################
# 6. Jarque-Bera Test
from scipy.stats import jarque_bera

data = np.random.normal(0, 1, 100)
stat, p = jarque_bera(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
#######################################################################
# 7. Cramer-von Mises Criterion
from statsmodels.stats.diagnostic import normal_ad

data = np.random.normal(0, 1, 100)
stat, p = normal_ad(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
####################################################################
#8. Chi-Square Goodness of Fit Test
from scipy.stats import chisquare

data = np.random.normal(0, 1, 100)
observed, bins = np.histogram(data, bins='auto')
expected = np.diff(bins) * len(data) / (bins[-1] - bins[0])
stat, p = chisquare(observed, expected)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
################################################################
# 9. Pearson's Chi-Square Test
from scipy.stats import chisquare

data = np.random.normal(0, 1, 100)
observed, bins = np.histogram(data, bins='auto')
expected = np.diff(bins) * len(data) / (bins[-1] - bins[0])
stat, p = chisquare(observed, expected)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
###############################################################
# 10. Z-test for Skewness and Kurtosis
import scipy.stats as stats

data = np.random.normal(0, 1, 100)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)
z_skewness = skewness / np.sqrt(6/len(data))
z_kurtosis = kurtosis / np.sqrt(24/len(data))

print(f"Z-score for skewness: {z_skewness:.3f}")
print(f"Z-score for kurtosis: {z_kurtosis:.3f}")

if abs(z_skewness) < 1.96 and abs(z_kurtosis) < 1.96:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

##############################################################


