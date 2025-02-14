# Hisse Senetlerinin Sektörel Benzerlik Analizi ve Sınıflandırma Modeli

**Bu proje, Tübitak'ta Senior Researcher, Quantitative Analyst olarak görevde bulunan [Dr. İsmail Güzel](https://www.linkedin.com/in/ismail-g%C3%BCzel-phd-7b9935a2/?locale=tr_TR) mentorluğunda, [Milli Teknoloji Akademisi](https://www.milliteknolojiakademisi.gov.tr/) kapsamında aldığım "Veri Yoğun Uygulamalar" eğitimi sonrasında tamamlanan bir projedir.**

## Proje Amacı
Bu projede, farklı sektörlerdeki hisse senetlerinin zaman serisi davranışları analiz edilerek, bir hisse senedinin hangi sektöre daha çok benzediği tespit edilmiş ve bu bilgilerin yatırım stratejilerinde kullanılabilirliği incelenmiştir.

## Kullanılan Araçlar ve Teknolojiler
- **Programlama Dili**: Python
- **Veri Kaynakları**: yfinance, finansal API'lar
- **Kütüphaneler**:
  - Veri Toplama: `yfinance`
  - Veri Ön İşleme: `pandas`, `numpy`
  - Zaman Serisi Analizi: `tsfresh`
  - Makine Öğrenmesi: `scikit-learn`, `XGBoost`
  - Veri Görselleştirme: `Matplotlib`, `BeautifulSoup`

## Proje Adımları
1. **Veri Toplama**
   - `yfinance` API'leri kullanılarak 2005'ten itibaren aylık getirilerle hisse senedi verileri toplandı.
   - Web scraping ile sektör ve hisse senedi listeleri çekildi.
```python
def fetch_sectors_names():
  url = "https://stockanalysis.com/stocks/industry/sectors/"
  response = requests.get(url)

  if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    df=pd.read_html(StringIO(str(soup.find_all("table"))))[0]

  else:
    print(f"Error: Failed to fetch data from page {url}")
  return df

|--------------------------------------------|

def fetch_industry_names():
  url = "https://stockanalysis.com/stocks/industry/all/"
  response = requests.get(url)

  if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    df=pd.read_html(StringIO(str(soup.find_all("table"))))[0]

  else:
    print(f"Error: Failed to fetch data from page {url}")
  return df

|--------------------------------------------|

def fetch_data(sectors):
  url = f"https://stockanalysis.com/stocks/sector/{sectors}/"
  response = requests.get(url)

  if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    df=pd.read_html(StringIO(str(soup.find_all("table"))))[0]
    df.drop(columns='No.', inplace=True)

  else:
    print(f"Error: Failed to fetch data from page {url}")
  return df

|--------------------------------------------|

# Örnek Kullanım
fetch_data(sectors='energy').to_csv('../data/stock_sectors/energy.csv')
fetch_data(sectors='financials').to_csv('../data/stock_sectors/financials.csv')

```

2. **Veri Ön İşleme**
   - Eksik veriler `forward-fill`, `backward-fill`, `ortalama` ve `medyan` ile dolduruldu.
```python
# Günlük açılış değerleri, "Open" fiyatları
data_open = final_data['Open']
# Günlük verileri aylığa çevir
data_open_monthly = data_open.resample('M').first()

# Her şirket için eksik verilerin ortalamasını bul
missing_ratio = data_open_monthly.isna().mean()
filtered_data = data_open_monthly.loc[:, missing_ratio < 0.4] # %40'tan fazla eksik veri olan sütunları at

# Aylık açılış fiyatları benzerlik göstereceği için ffill ve bfill kullan
filtered_data = filtered_data.ffill().bfill()

```
| Veri Doldurmanın Önemi                     | Görsel |
|--------------------------------------------|--------|
| **Temizlenmemiş veri** | ![image](https://github.com/user-attachments/assets/1a8fd36e-121a-4256-a911-6087981155f2)|
| **Temizlenmiş ve Düzenlenmiş veri** |![image](https://github.com/user-attachments/assets/925fd0d6-e486-4d08-9c87-7f35c3f2826e)|

   - Zaman serisi durağan hale getirilmek için log dönüşümü ve fark alma (“differencing”) işlemleri uygulandı.
```python
# Halihazırdaki Open sütununu güncellemek yerine yeni sütun açmak daha mantıklı
long_formatted_data["Open_LOG"] = np.log1p(long_formatted_data["Open"])
# çünkü bu kısmı tekrar tekrar çalıştırırsak Open değerleri sürekli değişiyor olacak.
```
| Log1/y Almanın Önemi                     | Görsel |
|--------------------------------------------|--------|
| **Aykırı ve uç değerler** | ![image](https://github.com/user-attachments/assets/a4569c32-8b8a-4679-84b8-c9c3c33bf4d9)|
| **Düzenlenmiş Değerler** |![image](https://github.com/user-attachments/assets/e6a5c547-355e-47ee-ad7c-609e5c3c9f74)|
    - **Open Değerleri**: İlk histogramdaki x=0'daki çubuk muhtemelen veri kümesinde çok sayıda sıfır veya çok küçük değerler olduğunu gösterir.
      - Bu, 'Open' sütununda çok sayıda sıfır veya sıfıra yakın değer varsa meydana gelebilir. Ancak önemli olan, x'teki diğer sayıların çok büyük olmasıdır.
    - **Open_LOG Değerleri**: Log dönüşümü (log1p) değerleri daha eşit bir şekilde dağılmıştır ve aşırı uç değerlerin etkisini azaltılmıştır
  
   - Kategorik değişkenler `one-hot encoding` veya `label encoding` ile sayısal hale getirildi.


| Kategorik Değişkenler                     | Görsel |
|--------------------------------------------|--------|
| **Encoding Yapılmamış Veri** | ![image](https://github.com/user-attachments/assets/fcbcfcdd-52e1-4871-972f-3a8f848e4e3b)|
| **Encoding Yapılmış Veri** |![Ekran görüntüsü 2025-02-14 185538](https://github.com/user-attachments/assets/557b1026-3f8e-4792-9d05-0038ffae985d)|

3. **Öznitelik Çıkarma ve Seçme**
   - `tsfresh` kütüphanesi ile istatistiksel öznitelikler (ortalama, standart sapma, otokorelasyon vb.) çıkarıldı.
```python
import tsfresh
from tsfresh.feature_extraction import EfficientFCParameters
# Extract features using only the 'Open' column
data_extract_features = tsfresh.extract_features(
  data_open_filled,
  column_id='Ticker',
  column_sort='Date',
  column_value='Open_LOG', # Explicitly specify the value column
  default_fc_parameters=EfficientFCParameters()
)

Index(['Open_LOG__variance_larger_than_standard_deviation',
 'Open_LOG__abs_energy', 'Open_LOG__mean_abs_change',
 'Open_LOG__mean_change',
 'Open_LOG__mean_second_derivative_central',
 'Open_LOG__median',
 ...
 'Open_LOG__fourier_entropy__bins_5',
 'Open_LOG__permutation_entropy__dimension_3__tau_1',
 'Open_LOG__permutation_entropy__dimension_4__tau_1',
 'Open_LOG__permutation_entropy__dimension_5__tau_1',
 'Open_LOG__query_similarity_count__query_None__threshold_0.0',
 'Open_LOG__mean_n_absolute_max__number_of_maxima_7'],
 dtype='object', length=777)

```
   - L1 regularization (Lasso), Recursive Feature Elimination (RFE) ve Principal Component Analysis (PCA) ile öznitelik seçimi yapıldı. En iyi sonucu veren seçildi. (RFE)
```python
# Model ve Veri İşleme Araçları
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
# Sınıflandırma Modelleri
from sklearn.ensemble import RandomForestClassifier,
# Özellik Seçimi ve Boyut İndirgeme
from sklearn.feature_selection import RFE

# Veriyi yükle
extracted_labeled_data = pd.read_csv('../data/processed_data/extracted_labeled_features.xlsx')

# Train ve Test için sütunları seç
X = extracted_labeled_data.drop(columns=['Label'])
y = extracted_labeled_data['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### 1. Recursive Feature Elimination (RFE) ###
selector_rfe = RFE(RandomForestClassifier(random_state=10))
selector_rfe.fit(X_train, y_train)

# Transform datasets
X_train_rfe = selector_rfe.transform(X_train)
X_test_rfe = selector_rfe.transform(X_test)

# Seçilen feature'ları yazdır
selected_features_rfe = X_train.columns[selector_rfe.get_support()]
print(f"RFE Selected Features ({len(selected_features_rfe)}): {list(selected_features_rfe)}")

# RFE Selected Features (385)

# Bu feature'ları kullanarak yeni veri setleri oluştur (385 yerine en iyi 20 feature seçilmiştir!)
X_train_top20_rf = X_train[top_20_features_rf]
X_test_top20_rf = X_test[top_20_features_rf]

# Model oluştur ve eğit
rf_model_top20 = RandomForestClassifier(random_state=10)
rf_model_top20.fit(X_train_top20_rf, y_train)

# Test seti üzerinde tahmin yap
y_pred_top20_rf = rf_model_top20.predict(X_test_top20_rf)

# Doğruluk (accuracy) hesapla RFE
accuracy_top20_rf = accuracy_score(y_test, y_pred_top20_rf)
print(f"\nEn önemli 20 feature kullanılarak elde edilen doğruluk: {accuracy_top20_rf:.4f}")

# Çapraz doğrulama yap RFE
cv_scores_top20_rf = cross_val_score(rf_model_top20, X_train_top20_rf, y_train, cv=5, scoring='accuracy')
print(f"Çapraz doğrulama skorları: {cv_scores_top20_rf}")
print(f"Ortalama çapraz doğrulama skoru: {cv_scores_top20_rf.mean():.4f}")
```
![image](https://github.com/user-attachments/assets/750838e5-85b1-463d-936a-04fa6934da0a)

4. **Model Geliştirme**
   - `Random Forest`, `Gradient Boosting`, `XGBoost` ve `CatBoost` algoritmaları ile modeller eğitildi.
```python
# Gerekli kütüphaneleri içe aktaralım
from sklearn.linear_model import LogisticRegression from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Label encoding (eğer label'lar string ise)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Farklı modelleri tanımla
models = {
  # Temel modeller
  "Logistic Regression": LogisticRegression(max_iter=1000,random_state=0),
  "SVC": SVC(random_state=0, probability=True),
  "Random Forest": RandomForestClassifier(n_estimators=100,random_state=10),
  "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,random_state=10),

  # Gelişmiş modeller
  "XGBoost": XGBClassifier(n_estimators=100, random_state=10,use_label_encoder=False, eval_metric='logloss'),
  "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100, 50),max_iter=1000, random_state=0),

  # Ek modeller
  "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=10),
  "Ridge Classifier": RidgeClassifier(random_state=0),
  "KNeighbors": KNeighborsClassifier(n_neighbors=5),
  "Gaussian NB": GaussianNB()
}

import os
import joblib # Modeli yüklemek için

# Kayıt dizinini oluştur (eğer yoksa)
save_dir = "../trained_models/"
os.makedirs(save_dir, exist_ok=True)

# Sonuçları tutacak liste
results = []

# Her modeli eğit, test et ve accuracy score hesapla
for model_name, model in models.items():
  print(f"\n{model_name} eğitiliyor...")
  model.fit(X_train_rfe, y_train_encoded) # Modeli eğit
  y_pred = model.predict(X_test_rfe) # Test verisiyle tahmin yap

  # Model dosya adını oluştur ve kaydet
  model_filename = os.path.join(save_dir, f"{model_name.replace(' ','_')}.pkl")
  joblib.dump(model, model_filename) # Modeli kaydet
  print(f"{model_name} kaydedildi: {model_filename}")

  acc = accuracy_score(y_test_encoded, y_pred) # Accuracy hesapla
  print(f"{model_name} Accuracy: {acc:.4f}")

  target_names = [str(cls) for cls in label_encoder.classes_]
  # Classification report yazdır
  print(classification_report(y_test_encoded, y_pred,target_names=target_names))

  # Sonuçları sakla
  results.append((model_name, acc))

# En iyi modeli bul
best_model = max(results, key=lambda x: x[1])
print(f"\nEn iyi model: {best_model[0]} (Accuracy: {best_model[1]:.4f})")
```
| Logistic Regression | SVC | Random Forest | Gradient Boosting | XGBoost |
|----------|----------|----------|----------|----------|
| ![image](https://github.com/user-attachments/assets/d305c903-52f3-4fb2-b899-dc95408d7d75)| ![image](https://github.com/user-attachments/assets/33d32ccb-7a7d-471a-86bb-5b92f5568bb4)|![image](https://github.com/user-attachments/assets/378ad18d-70dd-45f0-9302-d3955d5e3095)|![image](https://github.com/user-attachments/assets/e1e24265-7ad4-430d-8a20-f49f3b10bf9e)|![image](https://github.com/user-attachments/assets/dca0f8d1-c7b8-4367-8f44-17e6df35ae86)|
| MLP Classifier | AdaBoost | Ridge Classifier | KNeighbors | Gaussian NB |
| ![image](https://github.com/user-attachments/assets/94acd541-7c22-466b-8a07-b843ca0153a9)| ![image](https://github.com/user-attachments/assets/539f6866-131f-41cd-83e5-29b6da3133d2)| ![image](https://github.com/user-attachments/assets/427a7c92-e183-4a79-a724-1d2d4c55cb89)|![image](https://github.com/user-attachments/assets/61c01d24-d504-40c2-b93a-e65a9e972796)|![image](https://github.com/user-attachments/assets/0970c2bb-e2d4-4c20-924a-04e5ab342548)|

## Model Performances

| Rank | Model                | Test Accuracy |
|------|----------------------|--------------|
| 1️⃣  | **Gradient Boosting** | **0.816667** |
| 2️⃣  | **XGBoost**           | **0.805556** |
| 3️⃣  | **Ridge Classifier**  | **0.794444** |
| 4️⃣  | Random Forest        | 0.755556     |
| 5️⃣  | AdaBoost            | 0.727778     |
| 6️⃣  | Logistic Regression | 0.711111     |
| 7️⃣  | Gaussian NB         | 0.600000     |
| 8️⃣  | SVC                 | 0.594444     |
| 9️⃣  | MLP Classifier      | 0.561111     |
| 🔟  | KNeighbors          | 0.483333     |



   - `Grid Search` ve `Bayesian Optimization` kullanılarak hiperparametre optimizasyonu yapıldı.
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Model
gb_model = GradientBoostingClassifier()
# Hiperparametre aralığı (Bayesian Search)
param_space = {
  'n_estimators': Integer(50, 500), # Ağaç sayısı
  'learning_rate': Real(0.01, 0.2, prior='log-uniform'), # Öğrenme oranı
  'max_depth': Integer(3, 10), # Ağaç derinliği
  'subsample': Real(0.5, 1.0), # Rastgele örnekleme oranı
}
# Bayesian Search
bayes_search = BayesSearchCV(
  gb_model,
  param_space,
  n_iter=32, # Kaç farklı kombinasyon denenecek
  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
  scoring='accuracy',
  n_jobs=-1,
  verbose=2
)

# Modeli eğit
bayes_search.fit(X_train_rfe, y_train_encoded)

# En iyi modeli kaydet
best_model = bayes_search.best_estimator_joblib.dump(best_model,"../trained_models/Optimized_Gradient_Boosting.pkl")
```
   - `Cross-validation` ile modelin genelleme performansı değerlendirildi.
```python
# En iyi modeli yükle
best_model = joblib.load("../trained_models/Optimized_Gradient_Boosting.pkl")

# Cross-Validation ile performansı ölç
cv_scores = cross_val_score(best_model, X_train_rfe, y_train_encoded,cv=5, scoring='accuracy')

print("Cross-validation skorları:", cv_scores)
print("Ortalama doğruluk:", cv_scores.mean())
```
   - En iyi sonuç verilen model seçildi

5. **Model Değerlendirme**
   - Model performansı `accuracy`, `F1-score`, `ROC-AUC` metrikleriyle değerlendirildi.

6. **Sektörel Benzerlik Analizi**
   - Hisse senetlerinin sektörel benzerlikleri analiz edildi.
     - Bunun için ilk önce seçilen veri, modele uygun hale getirildi
```python
# --- 1. Veri İşleme Fonksiyonları ---
def preprocess_open_values(df):
  """ Open değerlerini işleyip log dönüşümü uygular ve 2005'ten itibaren filtreler. """
  df_data = yf.download(df, start='2005-01-01')

  df_open_data = df_data['Open'] # Sadece open verisini al
  df_open_data = df_open_data.ffill().fillna(0) # İlk ffill, sonra bütün boş değerleri 0 ile doldur
  df_open_data = df_open_data.resample('M').first() # Günlük verileri aylığa çevir.

  df_open_data_long = df_open_data.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Open') # Long türüne çevir
  df_open_data_long['Open_LOG'] = np.log1p(df_open_data_long['Open']) # Open değerlerinin, Log1/y değerini al

  return df_open_data_long[['Date', 'Ticker', 'Open_LOG']]

# --- 2. Özellik Çıkarma Fonksiyonu ---
def extract_features(df):
  """ Open_LOG üzerinden özellik çıkarır ve seçili olanları döndürür. """
  features = tsfresh.extract_features(
    df,
    column_id='Ticker',
    column_sort='Date',
    column_value='Open_LOG',
    default_fc_parameters=EfficientFCParameters()
  )
  return features[selected_features_rfe]

# --- 3. Pipeline Tanımlama ---
data_pipeline = Pipeline([
  ('preprocessing', FunctionTransformer(preprocess_open_values)), # Open işlemleri
  ('feature_extraction', FunctionTransformer(extract_features)), # Tsfresh özellik çıkarımı
])

# --- 4. Model Yükleme ---
#mlp_model = joblib.load("../trained_models/Gradient_Boosting.pkl") # Eğittiğin MLP modelini yükle
optimized_gradient_boost_model = joblib.load("../trained_models/Optimized_Gradient_Boosting.pkl")

def predict_sector(df):
  """ İşlenmiş veriyi modele sokar ve sektöre olan olasılıklarını döndürür. """
  X_processed = data_pipeline.transform(df)

  probabilities = optimized_gradient_boost_model.predict_proba(X_processed)[0] # Olasılıkları al
  sector_mapping = {0: 'Financials', 1: 'Healthcare', 2: 'Technology'}

  # Yuvarlanmış olasılıkları bir sözlük olarak döndür
  sector_probabilities = {sector_mapping[i]: round(prob, 4) for i, prob in enumerate(probabilities)}
  return sector_probabilities

```
   - Örneğin, Real-Estate sektöründeki hisselerin hangi sektörlere benzediği incelendi.
   
| Veri Tahmini Türü                          | Görsel |
|--------------------------------------------|--------|
| **Verisetinde olmayan bir endüstriden, şirket veri benzerliği** | ![image](https://github.com/user-attachments/assets/f0fe0e5e-a182-4f93-9a7c-c3b2d4be62a5) |
| **Verisetinde olan bir endüstriden, model içinde olmayan bir şirketin veri benzerliği** | ![image](https://github.com/user-attachments/assets/c3d35f6a-9fdc-4ed6-bd7a-5f77a720425f) |

7. **Sonuçların Görselleştirilmesi ve Raporlama**
   - Zaman serisi grafikleri, sektörel benzerlik matrisleri ve model performans metrikleri görselleştirildi.
   - Detaylı bir proje raporu hazırlandı.

## Beklenen Çıktılar
- **Sektörel benzerlikleri gösteren bir sınıflandırma modeli**
- **Farklı sektörlerdeki hisse senetlerinin davranışlarını analiz eden bir rapor**
- **Yatırım stratejileri için sektörel benzerlik bilgisi**

## Kurulum ve Kullanım
Bu projeyi çalıştırmak için:
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install yfinance tsfresh scikit-learn scikit-optimize XGBoost matplotlib
   ```
2. Verileri toplayın ve ön işleyin.
3. Modeli eğitin ve değerlendirin.
4. Sonuçları görselleştirin ve raporlayın.

## Lisans
Bu proje, Milli Teknoloji Akademisi kapsamında gerçekleştirilmiş olup akademik ve eğitim amacıyla kullanılabilir.
