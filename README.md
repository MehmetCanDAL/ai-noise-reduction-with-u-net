AI Noise Reduction (U-Net)
Bu projede, derin öğrenme (U-Net) kullanarak gürültülü ses kayıtlarını temizleyen bir model geliştiriyorum.

Ne Yapıyor?
Gürültülü sesleri alıyor, arka plandaki sesleri (dip ses, cızırtı vb.) ayıklıyor ve daha net bir ses haline
getiriyor.

Model mimarisi olarak U-Net kullanıldı.

Ses işleme taraflarında Librosa ve NumPy kütüphanelerinden yararlanıldı.

Nasıl Çalıştırılır?
1 - Gerekli kütüphaneleri kurun:
pip install -r requirements.txt
2 -Modeli eğitmek veya test etmek için main.py (veya senin dosyanın adı neyse onu yaz) dosyasını çalıştırın.

Kullanılan Araçlar
Python
TensorFlow / Keras