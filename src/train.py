# =====================================================
# CREDIT CARD FRAUD DETECTION WITH ANN – CLEAN PIPELINE
# =====================================================

# 0. Reproducibility
import os
import random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)


# 1. Libraries
import pandas as pd
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns            # Visualization 

from sklearn.model_selection import train_test_split  # ML & Preprocessing
from sklearn.preprocessing import StandardScaler      # ML & Preprocessing
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
) # Metrics
from imblearn.over_sampling import SMOTE  # Imbalance handling
from tensorflow.keras.models import Sequential        # Deep Learning (ANN)
from tensorflow.keras.layers import Dense, Dropout    # Deep Learning (ANN)
from tensorflow.keras.callbacks import EarlyStopping  # Deep Learning (ANN)
from tensorflow.keras.utils import to_categorical     # Deep Learning (ANN)


# 2. Read data
df = pd.read_csv(
    "creditcard.csv"
)


# 3. Quick dataset overview
print(df.describe()) # Tanımlayıcı istatistikler
print(df.columns)    # Sütun adları
print(df.info())     # Veri tipleri ve missing value kontrolü


# 4. Target distribution (imbalance analysis)
print(df["Class"].value_counts())
sns.countplot(x="Class", data=df)
plt.title("Target imbalance")
plt.grid(True, axis="y", alpha=0.3)
plt.show()
# Verimiz doğal olarak çok dengesiz SMOTE yöntemi ile train setimizideki dengeyi sağlayalım  


# 5. Feature–target split
features = df[
    [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
        "V28", "Amount"
    ]  # x bağımsız değişken tablomuz
]
target = df["Class"]  # y etiket değerlerimiz


# 6. Train / test split (prevent leakage)
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=SEED
)
# Veri setimizi train (eğitim) için %80 test %20 için ayırdık
# x_train: modelimizi eğiteceğimiz bağımsız değişken tablomuz
# y_train: x_train tablomuzdaki verilerimizin etiketleri
# x_test : modelimizin eğitimi bittikten sonra modelimizi test edeceğimiz bağımsız değişkenler tablomuz
# y_test : x_test tablomuzun etiket değerleri


# 7. SMOTE only on training set
smote = SMOTE(random_state=SEED)
x_train, y_train = smote.fit_resample(x_train, y_train) # train veri setimizi dengeledik.


# 8. Feature scaling (fit on training only)
scaler = StandardScaler() 
x_train = scaler.fit_transform(x_train) # x train bağımsız değişkenimizi (0,1) arasına normalize ettik
x_test = scaler.transform(x_test)       # x test bağımsız değişkenimizi (0,1) arasına normalize ettik


# 9. One-hot encoding for ANN
y_train = to_categorical(y_train, num_classes=2) # y train etiketlerimizi kategorik hale getirdik
y_test = to_categorical(y_test, num_classes=2)   # y test etiketlerimizi kategorik hale getirdik


# 10. ANN model
model = Sequential()

# Giriş katmanı: 64 nöron, ReLU aktivasyonu
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
# Dropout katmanı: %25 oranında
model.add(Dropout(0.25))

# ikinci katman: 32 nöron, ReLU aktivasyonu
model.add(Dense(32, activation="relu"))
# Dropout katmanı: %5
model.add(Dropout(0.10))

# üçüncü katman: 16 nöron, ReLU aktivasyonu
model.add(Dense(16, activation="relu"))

# çıkış katmanı: 2 nöron, sigmoid aktivasyonu
model.add(Dense(2, activation="softmax"))

# model özeti
model.summary()

model.compile(
    optimizer = "adam",           # Adaptif momentum (Adam)
    loss = "binary_crossentropy", # İkili çapraz entropy
    metrics = ["accuracy"]        # Başarı metriği
)


# 11. Train with EarlyStopping
early = EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss")
# modelimizin eğitilirken "val_loss" değeri 5 adımda iyileşmezse eğitimi durdurulacak
  
history = model.fit(
    x_train,
    y_train,
    validation_split=0.25,  # validation oranımız %20
    epochs=50,             # modelimiz 40 döngü yapacak
    batch_size=128,        # veriler modele 128 parça halinde verilecek
    callbacks=[early],     # modeli durdurma koşulu
    verbose=1              # eğitim çıktısı görme 
)


# 12. Evaluate on test
# test verisi üzerindeki model performansı
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0) 
print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.5f}")

# 13. Learning curves
# modelin eğitim sürecindeki performansına bakalım
plt.figure()
plt.plot(history.history["accuracy"], marker="o", label="Training accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation accuracy")
plt.title("ANN accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history["loss"], marker="o", label="Training loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation loss")
plt.title("ANN loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# 14. Test predictions
y_prob = model.predict(x_test, verbose=0)       # x_test için modelin tahmin ettiği olasılıklar
y_pred = np.argmax(y_prob, axis=1)              # Her satırdaki en yüksek olasılığa sahip sınıfın indeksi
y_true = y_test.argmax(axis=1)                  # One-hot encoded y_test'ten gerçek sınıf indeksleri
     
# 15. ROC–AUC
roc_auc = roc_auc_score(y_true, y_prob[:, 1])   # y_true ile pozitif sınıf olasılıkları kullanılarak ROC–AUC hesaplanır
print("ROC AUC:", round(roc_auc, 5))            # Hesaplanan ROC–AUC değeri 

# ROC eğrisi
fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("Yanlış Pozitif Oranı")
plt.ylabel("Doğru Pozitif Oranı")
plt.title("ROC Eğrisi")
plt.legend()
plt.grid(True)
plt.show()

type(y_test) 


# 16. Precision–Recall
precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall curve")
plt.grid(True)
plt.show()


# 17. Confusion matrix & classification report
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
