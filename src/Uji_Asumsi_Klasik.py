import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV
data = pd.read_csv('data\\CancerPatientDataSets.csv', delimiter=';')

# Daftar variabel yang ingin dianalisis
variable_combinations = [
        # dependen              # independen
    ('Chronic_Lung_Disease', ['Air_Pollution','Dust_Allergy','Occupational_Hazards','Genetic_Risk','Smoking','Passive_Smoker','Chest_Pain','Coughing_of_Blood','Shortness_of_Breath','Dry_Cough']),

    # Tambahkan kombinasi variabel lainnya sesuai kebutuhan
]

for dependent_var, independent_vars in variable_combinations:
    print("=" * 150)
    print(f"\nAnalisis Regresi untuk {dependent_var} dan {independent_vars}:\n")

    # Variabel dependen dan independen
    y = data[dependent_var]
    X = data[independent_vars]

    # Tambahkan konstanta ke variabel independen
    X = sm.add_constant(X)

    # Fit model regresi
    model = sm.OLS(y, X).fit()

    # # Tampilkan ringkasan hasil regresi
    # print(model.summary())

    # Uji Normalitas Residual
    shapiro_test_stat, shapiro_p_value = stats.shapiro(model.resid)
    print(f"Shapiro-Wilk Test Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}")
    if shapiro_p_value < 0.05:
        print("Normalitas: Residual tidak terdistribusi normal")
    else:
        print("Normalitas: Residual terdistribusi normal")

    # Visualisasi Q-Q Plot
    sm.qqplot(model.resid, line='s')
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    # Uji Heteroskedastisitas
    bp_test_stat, bp_p_value, _, _ = het_breuschpagan(model.resid, X)
    print(f"Breusch-Pagan Test Statistic: {bp_test_stat}, p-value: {bp_p_value}")
    if bp_p_value < 0.05:
        print("Heteroskedastisitas: Ada indikasi heteroskedastisitas")
    else:
        print("Heteroskedastisitas: Tidak ada indikasi heteroskedastisitas")

    # Visualisasi Scatter Plot Residual vs Fitted
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, scatter_kws={'alpha': 0.5})
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # Uji Autokorelasi
    durbin_watson_stat = sm.stats.durbin_watson(model.resid)
    print(f"Durbin-Watson Statistic: {durbin_watson_stat}")

    if durbin_watson_stat < 1.5 or durbin_watson_stat > 2.5:
        print("Autokorelasi: Ada indikasi autokorelasi")
    else:
        print("Autokorelasi: Tidak ada indikasi autokorelasi\n")

    # Visualisasi Correlogram (ACF Plot)
    sm.graphics.tsa.plot_acf(model.resid, lags=40)
    plt.title('Autocorrelation Function (ACF) Plot of Residuals')
    plt.show()
