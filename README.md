# ANN_ILE_KREDI_KARTI_DOLANDIRICILIGI_TESPITI

## 📌 Problem Tanımı
Finans sektöründe dolandırıcılık işlemleri banka ve müşteriler için ciddi maddi kayıplara yol açmaktadır.  
Bu projede amaç, kredi kartı işlemleri üzerinden dolandırıcılık olup olmadığını tahmin eden Yapay Sinir Ağı (ANN) modeli geliştirmektir.

---

## 💼 İşe Etkisi
Bu model:

- Banka zararını azaltır  
- Müşteri mağduriyetini düşürür  
- Gerçek zamanlı dolandırıcılık sistemlerine entegre edilebilir  
- Model, 100 dolandırıcılık işleminin **92 tanesini doğru şekilde yakalayabilmektedir**

---

## 📊 Veri Seti
- Anonimleştirilmiş kredi kartı işlem verileri  
- Şiddetli dengesiz sınıf yapısı (**%0.17 fraud**)  
- Sayısal özelliklerden oluşmaktadır  

---

## ⚙️ Ön İşleme
- Eksik verilerin temizlenmesi  
- StandardScaler ile ölçekleme  
- SMOTE ile veri dengelenmesi  

---

## 🧠 Model Yapısı (ANN)
- Giriş katmanı  
- Gizli katmanlar (ReLU)  
- Dropout katmanları  
- Çıkış katmanı (Softmax)  

---

## 📈 Değerlendirme Metrikleri
- ROC-AUC  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 🏆 Sonuçlar

| Metrik            | Değer |
| ----------------- | ----- |
| ROC-AUC           | 0.962 |
| Recall (Fraud)    | 0.857 |
| Precision (Fraud) | 0.442 |
| F1-Score          | 0.583 |

---
**Veri seti büyük olduğundan GitHub’a eklenmemiştir. Credit Card Fraud Dataset kullanılmaktadır**
## ▶ Projeyi Çalıştırma

```bash
pip install -r requirements.txt
python src/train.py


