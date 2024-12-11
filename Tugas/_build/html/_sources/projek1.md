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

# Analisis Perilaku Konsumen Berdasarkan Data Penjualan Retail

### Latar Belakang

i era digital saat ini, pemilik retail harus mampu memahami pola konsumsi dan preferensi pelanggan untuk menciptakan strategi pemasaran yang efektif, meningkatkan pengalaman berbelanja, dan membangun loyalitas pelanggan. Salah satu cara untuk memahami perilaku konsumen yaitu dengan menganalisis data penjualan retail. untuk mendapatkan pola pembelian pelanggan diperlukan analisis asosiasi. Analisis asosiasi atau association rule mining sendiri merupakan teknik data mining untuk menemukan aturan assosiatif antara suatu kombinasi item. Dengan begitu pemilik retail dapat dengan mudah mengatur penempatan barangnya atau merancang promosi pemasaran dengan memakai kupon diskon untuk kombinasi barang tertentu.

### Tujuan

Untuk meningkatkan penjualan toko berdasarkan perilaku konsumen pada data penjualan

### Rumusan Masalah

Bagaimana pola pembelian konsumen dari data penjualan untuk meningkatkan tingkat penjualan produk?

```{code-cell}
pip install mlxtend
pip install --upgrade mlxtend
pip install apyori
```

```{code-cell}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# pip install apyori
import datetime
```

```{code-cell}
df = pd.read_csv('https://raw.githubusercontent.com/fadh54/Tugas_PSD/refs/heads/main/Retail.csv', delimiter=';')
```

Sekumpulan data yang diambil dari github dan akan digunakan untuk proses data mining

```{code-cell}
df.head()
```

```{code-cell}
df.info()
```

```{code-cell}
df.isnull().sum()
```

## Data Pre-Processing

Data pre-pocessing adalah proses mengolah data mentah yang didapat menjadi data yang lebih bersih agar mudah untuk diolah

### Data Cleansing

Proses untuk membersihkan data dengan menghapus variabel CustomerID yang memiliki nilai kosong, dan menghilangkan Transaksi yang merupakan transaksi kredit (Di awali dengan hufuf C pada invoice number)

```{code-cell}
df.head()
```

```{code-cell}
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')
```

```{code-cell}
df['PRODUCT'] = df['PRODUCT'].str.strip()
df['PRODUCT_CATEGORY'] = df['PRODUCT_CATEGORY'].str.strip()

df.dropna(axis=0, subset=['CustomerID'], inplace=True)
```

Membersihkan ruang di deskripsi produk, kategori produk, dan menghapus baris yang tidak memiliki data CustomerID yang valid

```{code-cell}
df.isnull().sum()
```

mengecek data yang memiliki nilai kosong

```{code-cell}
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~(df['InvoiceNo'].str[0] == 'C')]
df
```

menghapus variabel inoviceNO yang diawali dengan huruf C pada invoice numbernya

### Data Transformation

Data transformation proses untuk mengubah format data, mengkonversi tipe data, melakukan perhitungan data, menyaring data yang tidak relevan

```{code-cell}
Keranjang = (df.groupby(['InvoiceNo', 'PRODUCT_CATEGORY'])['Quantity'].sum()\
                                      .unstack().reset_index().fillna(0)\
                                      .set_index('InvoiceNo'))
Keranjang
```

```{code-cell}
Keranjang.iloc[:,[0,1,2,3,4,5,6,7]].head()
```

```{code-cell}
def ubah_angka(x):
  if x<=0:
    return 0
  else:
    return 1
barang = Keranjang.applymap(ubah_angka)
barang.head()
```

Kemudian melakukan encoding, proses encoding adalah proses mengubah data ke dalam bentuk angka, agar sistem atau komputer dapat memahami informasi dari dataset, jika barang kurang dari sama dengan 0 maka keranjang tersebut bernilai 0 dan jika lebih dari 1 maka nilainya adalah 1, sehingga jika sebuah nota pembelian barang A sebanyak 10 buah maka hanya akan dihitung 1. Karena analisis yang di gunakan menyaratkan seperti itu.

## Data Minning

Data mining atau bisa juga disebut data exploration, proses mencari pola atau informasi menarik dalam data dengan menggunakan teknik atau metode tertentu.

```{code-cell}
frequent_itemsets = apriori(barang, min_support=0.1, use_colnames=True)
num_itemsets = len(frequent_itemsets)
frequent_itemsets
```

membuat variable dari beberapa barang yang sering terbeli atau ada pada transaksi dengan menggunakan perintah apriori. Dengan data yang berasal dari dataframe barang dengan minimum nilai support 0.1 / 10%.membuat variable dari beberapa barang yang sering terbeli atau ada pada transaksi dengan menggunakan perintah apriori. Dengan data yang berasal dari dataframe barang dengan minimum nilai support 0.1 / 10%.

```{code-cell}
rules = association_rules(frequent_itemsets, num_itemsets, metric="lift", min_threshold=1)
rules
```

```{code-cell}
result = rules[(rules['lift'] >= 1) &
               (rules['confidence'] >= 0.5)]

apr_result = result.sort_values(by='confidence', ascending=False)
apr_result
```

Melakukan filter untuk nilai lift ratio lebih dari sama dengan 1 dengan tingkat confidence minimal 0.8 (lebih dari sama dengan 80%)

## Interpretation

Dari result table yang telah di filter, dapat ditarik kesimpulan bahwa produk-produk yang dibeli secara bersamaan oleh customer terhadap rule asosiasi pada dataset dengan min_support 0.1 / 10%, min_threshold = 1, dan nilai lift sebesar lebih dari sama dengan 1 serta tingkat confidence minimal yang diperhitungkan sebesar 0.5 (50%) diantaranya adalah:
Sabun & Sampooh, Parfum, and Kosmetik.
Biskuit and Kosmetik.
Minuman, Susu, and Kosmetik.

```{code-cell}
apr_result['consequents'].value_counts()
```

When your book is built, the contents of any `{code-cell}` blocks will be
executed with your default Jupyter kernel, and their outputs will be displayed
in-line with the rest of your content.

```{seealso}
Jupyter Book uses [Jupytext](https://jupytext.readthedocs.io/en/latest/) to convert text-based files to notebooks, and can support [many other text-based notebook files](https://jupyterbook.org/file-types/jupytext.html).
```

## Create a notebook with MyST Markdown

MyST Markdown notebooks are defined by two things:

1. YAML metadata that is needed to understand if / how it should convert text files to notebooks (including information about the kernel needed).
   See the YAML at the top of this page for example.
2. The presence of `{code-cell}` directives, which will be executed with your book.

That's all that is needed to get started!

## Quickly add YAML metadata for MyST Notebooks

If you have a markdown file and you'd like to quickly add YAML metadata to it, so that Jupyter Book will treat it as a MyST Markdown Notebook, run the following command:

```
jupyter-book myst init path/to/markdownfile.md
```
