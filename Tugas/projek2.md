---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Prediksi Jumlah Pengunjung Coffee Shop

### Latar Belakang

Coffee shop telah menjadi bagian penting dalam gaya hidup masyarakat modern, Tidak hanya menjadi tempat untuk menikmati kopi, coffee shop juga berfungsi sebagai tempat bersosialisasi, bekerja, belajar, dan bahkan menghadiri berbagai acara. Dengan semakin banyaknya coffee shop yang bermunculan, persaingan antar bisnis semakin ketat. Maka diperlukan kemampuan untuk memprediksi jumlah pengunjung. Data yang digunakan dalam pengerjaan tugas ini berupa data time series penjualan coffe sales yang didapat dari website kaggle [Coffe Sales](https://www.kaggle.com/datasets/ihelon/coffee-sales/data). Dengan memprediksi jumlah pengunjung yang membeli di coffe sales 3 hari kedepan.

### Tujuan

- Untuk mengoptimalkan perencanaan operasional
- Untuk meningkatkan efisiensi biaya produksi dan tenaga kerja

### Rumusan masalah

Bagaimana prediksi jumlah pengunjung dalam kurun waktu 3 hari kedepan?

```{code-cell}
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
```

```{code-cell}
data = 'https://raw.githubusercontent.com/fadh54/Tugas_PSD/refs/heads/main/sales_coffe.csv'
df_baru = pd.read_csv(data, delimiter=';')
df_baru
```

Menampilkan data yang akan digunakan untuk melakukan prediksi

## Data Understanding

```{code-cell}
# Merubah kolom 'Date' dalam format datetime dengan dayfirst=True
df_baru['date'] = pd.to_datetime(df_baru['date'], dayfirst=True, errors='coerce')

# Mengatur kolom 'Date' sebagai indeks
df_baru.set_index('date', inplace=True)
```

```{code-cell}
df_baru.head()
```

```{code-cell}
df_baru.plot()
```

Menampilkan grafik data pengunjung dari tanggal 01/03/2024 sampai dengan tanggal 07/10/2024

```{code-cell}
df_baru.shape
```

```{code-cell}
df_baru.info()
```

```{code-cell}
df_baru.dtypes
```

```{code-cell}
df_baru.isnull().sum()
```

```{code-cell}
df_baru.describe()
```

## Data Pre-Processing

Data pre-pocessing adalah proses mengolah data mentah yang didapat menjadi data yang lebih bersih agar mudah untuk diolah

### Sliding Window

melakukan slidding window untuk mendapatkan data penjualan 3 hari sebelumnya yang akan digunakan untuk melakukan prediksi jumlah pengunjung

```{code-cell}
def sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)].flatten())
        y.append(data[i + window_size])
    return pd.DataFrame(X), pd.DataFrame(y)

# Menggunakan fungsi sliding window
window_size = 3
X, y = sliding_window(df_baru['jumlah_pengunjung'].values.reshape(-1, 1), window_size)

# Menampilkan hasil
X.columns = [f'lag_{i}' for i in range(window_size,0,-1)]
y.columns = ['target']
result = pd.concat([X, y], axis=1)

print(result.head())
```

### Normalisasi

data slidding window yang sudah didapatkan akan dinormalisasi

```{code-cell}
target = df_baru['jumlah_pengunjung']

# Scale data
scaler = MinMaxScaler()
df_baru['jumlah_pengunjung'] = scaler.fit_transform(df_baru[['jumlah_pengunjung']])

def normalisasi_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)].flatten())
        y.append(data[i + window_size])
    return pd.DataFrame(X), pd.DataFrame(y)

window_size = 3
X, y = normalisasi_sliding_window(df_baru['jumlah_pengunjung'].values.reshape(-1, 1), window_size)

# Menampilkan hasil
X.columns = [f'lag_{i}' for i in range(window_size,0,-1)]
y.columns = ['target']
result = pd.concat([X, y], axis=1)

print(result.head())
```

## Transformasi Data

membagi data menjadi data train dan data test, dengan presentase 80% data train dan 20% data test

```{code-cell}
# Membagi data menjadi data train dan test (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menampilkan jumlah data train dan test
print("Jumlah data train:", len(X_train))
print("Jumlah data test:", len(X_test))
```

## Modeling

### Evaluasi Model

#### 1. MSE (Mean Squared Error)

- _Definisi_: MSE adalah rata-rata dari kuadrat selisih antara nilai yang diprediksi dan nilai aktual.
- _Rumus_:
  $$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
  Di mana $y_i$ adalah nilai aktual, $\hat{y}_i$ adalah nilai prediksi, dan $n$ adalah jumlah data.

#### 2. RMSE (Root Mean Squared Error)

- _Definisi_: RMSE adalah akar kuadrat dari MSE, yang mengembalikan satuan ke skala yang sama dengan data asli.
- _Rumus_:
  $$
  \text{RMSE} = \sqrt{\text{MSE}}
  $$

#### 3. MAPE (Mean Absolute Percentage Error)

- _Definisi_: MAPE mengukur kesalahan dalam prediksi sebagai persentase dari nilai aktual.
- _Rumus_:
  $$
  \text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%
  $$

### Begging Regresor

Rumus algoritma decision tree:

- Menghitung Entropy

  untuk menghitung ketidakpastian dalam data.

  $$
  \text {H(S)} = - \sum_{i=1}^{k} {p_i} . {log_2(p_i)}
  $$

  ${H(S)}$: Entropi dari dataset $S$

  ${p_i}$: Proporsi sampel kelas ke-$i$ dalam dataset $S$.

  ${k}$: Jumlah kelas dalam dataset.

- Menghitung Gain

  $$
  \text {IG}{(S,A)} = {H(S)} - \sum_{v∈V}^{k} \frac{|S_v|}{|S|} . {H}{(S_v)}
  $$

  ${H(S)}$ : Entropi dataset sebelum split.

  ${S_v}$ : Subset data setelah split berdasarkan nilai ${v}$ dari fitur ${A}$.

  ${|S_v|}$ : Jumlah sampel dalam subset ${|S_v|}$.

  ${|S|}$ : Total sampel dalam dataset ${S}$.

```{code-cell}
model_begging = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=20, random_state=42)
model_begging.fit(X_train, y_train)
y_pred = model_begging.predict(X_test)
```

```{code-cell}
# memprediksi jumlah pengunjung 7 hari ke depan
def predict_future(model_begging, last_window, steps=1):
    future_predictions = []
    for _ in range(steps):
        # Ensure current_window has the same column order as X_train
        current_window = pd.DataFrame([last_window], columns=X_train.columns)
        prediction = model_begging.predict(current_window)
        future_predictions.append(prediction[0])
        # Update current window dengan prediksi terbaru
        last_window = np.roll(last_window, -1)
        last_window[-1] = prediction[0]
    return future_predictions

# # Ambil jendela terakhir dari data
last_window = X.values[-1]

# # Prediksi 3 hari ke depan
future_steps = 3
future_predictions = predict_future(model_begging, last_window, future_steps)

# # Tampilkan prediksi masa depan
# future_df = pd.date_range(start=df_baru['date'].iloc[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')
future_df = pd.date_range(start=df_baru.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')
print("jumlah pengunjung 3 hari ke depan ",future_predictions)

plt.figure(figsize=(10, 5))
plt.plot(df_baru.index, df_baru['jumlah_pengunjung'], label='Data Aktual')
plt.plot(future_df, future_predictions, label='Prediksi 3 Hari ke Depan', linestyle='--')
plt.title('Prediksi Jumlah Pengunjung 3 Hari ke Depan')
plt.xlabel('Tanggal')
plt.ylabel('Prediksi Jumlah Pengunjung')
plt.grid()
plt.legend()
plt.show()
```

```{code-cell}
# Menghitung MSE, MAE, RMSE, R² dan MAPE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"{'Bagging Regressor'} \n MSE: {mse}\n MAE: {mae}\n RMSE: {rmse}\n R²: {r2}\n MAPE: {mape}%")
```

### Linear Regression

Rumus algoritma Regresi Linear

Regresi linear dapat dinyatakan dengan rumus:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
$$

di mana:

- $y$ adalah variabel dependen (target).
- $\beta_0$ adalah intercept (konstanta).
- $\beta_1, \beta_2, \ldots, \beta_n$ adalah koefisien regresi untuk masing-masing variabel independen $x_1, x_2, \ldots, x_n$.
- $\epsilon$ adalah error atau residual.

```{code-cell}
# Model Linear Regression
model_linear = BaggingRegressor(estimator=LinearRegression(), n_estimators=20, random_state=42)
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
```

```{code-cell}
# memprediksi jumlah pengunjung 7 hari ke depan
def predict_future(model, last_window, steps=1):
    future_predictions = []
    for _ in range(steps):
        current_window = pd.DataFrame([last_window], columns=X_train.columns)
        prediction = model.predict(current_window)
        future_predictions.append(prediction[0])
        # Update current window dengan prediksi terbaru
        last_window = np.roll(last_window, -1)
        last_window[-1] = prediction[0]
    return future_predictions

# # Ambil jendela terakhir dari data
last_window = X.values[-1]

# # Prediksi 3 hari ke depan
future_steps = 3
future_predictions = predict_future(model_linear, last_window, future_steps)

# # Tampilkan prediksi masa depan
# future_df = pd.date_range(start=df_baru['date'].iloc[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')
future_df = pd.date_range(start=df_baru.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')
print("jumlah pengunjung 3 hari ke depan ",future_predictions)

plt.figure(figsize=(10, 5))
plt.plot(df_baru.index, df_baru['jumlah_pengunjung'], label='Data Aktual')
# plt.plot(X_test.index, y_pred, label='Prediksi Linear Regression (Test)', color='green', linestyle='--')
plt.plot(future_df, future_predictions, label='Prediksi 3 Hari ke Depan',color='green', linestyle='--')
plt.title('Prediksi Jumlah Pengunjung 3 Hari ke Depan dengan Linear Regresion')
plt.xlabel('Tanggal')
plt.ylabel('Prediksi Jumlah Pengunjung')
plt.grid()
plt.legend()
plt.show()
```

```{code-cell}
# Menghitung MSE, MAE, RMSE, R² dan MAPE
mse = mean_squared_error(y_test, y_pred_linear)
mae = mean_absolute_error(y_test, y_pred_linear)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_linear)
mape = mean_absolute_percentage_error(y_test, y_pred_linear)
print(f"{'Linear Regression'} \n MSE: {mse}\n MAE: {mae}\n RMSE: {rmse}\n R²: {r2}\n MAPE: {mape}%")
```

### Support Vector Machine

Rumus Algoritma SVM

$$
\hat{y}_i = \sum_{i=1}^{n} ({α}_i - {α}_i^*) K({x}_i, {x})+b
$$

$\hat{y}$ : Prediksi nilai target.

${n}$ : Jumlah sampel pelatihan.

${α}_i, {α}_i^*$ : Parameter Lagrange yang dioptimalkan selama pelatihan.

$K({x}_i, {x})$ : Kernel yang mengukur kesamaan antara data pelatihan ${(x_i)}$ dan data uji ${x}$ .

```{code-cell}
model_svm = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model_svm = BaggingRegressor(estimator=model_svm, n_estimators=20, random_state=42)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
```

```{code-cell}
def predict_future(model_svm, last_window, steps=1):
    future_predictions = []
    for _ in range(steps):
        current_window = pd.DataFrame([last_window], columns=X_train.columns)
        prediction = model_svm.predict(current_window)
        future_predictions.append(prediction[0])
        # Update current window dengan prediksi terbaru
        last_window = np.roll(last_window, -1)
        last_window[-1] = prediction[0]
    return future_predictions

# Ambil jendela terakhir dari data
last_window = X.values[-1]

# Prediksi 3 hari ke depan
future_steps = 3
future_predictions = predict_future(model_svm, last_window, future_steps)

# Membuat tanggal untuk prediksi masa depan
future_df = pd.date_range(start=df_baru.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')

# Menampilkan prediksi jumlah pengunjung
print("Prediksi jumlah pengunjung 3 hari ke depan:", future_predictions)

# Visualisasi prediksi
plt.figure(figsize=(10, 5))
plt.plot(df_baru.index, df_baru['jumlah_pengunjung'], label='Data Aktual', color='blue')
plt.plot(future_df, future_predictions, label='Prediksi 3 Hari ke Depan (SVM)', color='green', linestyle='--')
plt.title('Prediksi Jumlah Pengunjung 3 Hari ke Depan dengan SVM')
plt.xlabel('Tanggal')
plt.ylabel('Prediksi Jumlah Pengunjung')
plt.grid()
plt.legend()
plt.show()
```

```{code-cell}
# Menghitung MSE, MAE, RMSE, R² dan MAPE
mse = mean_squared_error(y_test, y_pred_svm)
mae = mean_absolute_error(y_test, y_pred_svm)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_svm)
mape = mean_absolute_percentage_error(y_test, y_pred_svm)
print(f"{'Support Vector Machine'} \n MSE: {mse}\n MAE: {mae}\n RMSE: {rmse}\n R²: {r2}\n MAPE: {mape}%")
```

Kesimpulan dari hasil prediksi menggunakan ke-3 model tersebut, didapatkan best model nya yaitu prediksi menggunakan Linear regression.
