import pandas as pd
import numpy as np

# Mengganti 'nama_file.csv' dengan nama file CSV yang sesuai
file_path = 'data\\CancerPatientDataSets.csv'

# Membaca data dari file CSV
data = pd.read_csv(file_path, delimiter=';')

# Memilih 10 variabel independen yang numerik
independent_variables = ['Air_Pollution','Dust_Allergy','Occupational_Hazards','Genetic_Risk','Smoking','Passive_Smoker','Chest_Pain','Coughing_of_Blood','Shortness_of_Breath','Dry_Cough']

# Menambahkan kolom konstanta ke variabel independen
X = np.column_stack((np.ones(len(data)), data[independent_variables]))

# Variabel dependen yang ingin diprediksi
y = data['Chronic_Lung_Disease']  # Ganti dengan variabel dependen yang sesuai

# Menghitung matriks (X'X) inverse
XtX_inv = np.linalg.inv(np.dot(X.T, X))

# Menghitung matriks (X'Y)
XtY = np.dot(X.T, y)

# Menghitung koefisien regresi
beta = np.dot(XtX_inv, XtY)

# Menampilkan koefisien regresi
print("Koefisien Regresi:")
for i, var in enumerate(['Intercept'] + independent_variables):
    print(f"{var}: {beta[i]}")
