<<<<<<< HEAD
# 🎧 Speech Enhancement with U-Net (VoiceBank+DEMAND)

Bu proje, gürültülü ses kayıtlarını temizlemek için **U-Net** mimarisi kullanılarak geliştirilen bir derin öğrenme projesidir. Şu an geliştirme aşamasındadır.

---

## 🛠️ Mevcut Durum (WIP)

- [x] **Dataset Katmanı:** Verilerin yüklenmesi ve gruplanması (`dataset.py`).
- [x] **Dönüşüm Katmanı:** STFT/iSTFT işlemleri ve faz koruma (`transforms.py`).
- [x] **Mimari Tasarımı:** Dinamik padding destekli U-Net modeli (`model.py`).
- [ ] **Eğitim Döngüsü:** Loss fonksiyonu ve Optimizer ayarları (Yükleniyor...).
- [ ] **Değerlendirme:** PESQ ve STOI metrikleri (Planlanıyor).

---

## 📂 Dosya Yapısı

* **dataset.py:** VoiceBank+DEMAND veri setini PyTorch'a uygun şekilde yükler.
* **transforms.py:** Sesi spektrograma çevirir ve iSTFT ile tekrar sese dönüştürür.
* **model.py:** Maske tabanlı U-Net model mimarisini içerir (~31M Parametre).

---
=======
>>>>>>> bb44a14 (Dataset classina yorum satirlari eklendi)
