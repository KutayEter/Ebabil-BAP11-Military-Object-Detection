  # BAP-11 Otonom Sürü Kamikaze İHA Angajmanı Optimizasyonu
Bu proje, Ankara Bilim Üniversitesi BAP-11 – Otonom Sürü Kamikaze İHA Angajmanı Optimizasyonu kapsamında geliştirilen görüntü işleme ve hedef tespit çalışmalarını içermektedir.
Çalışmanın amacı, otonom İHA sistemlerinin sahadaki askeri hedefleri algılayabilmesi, sınıflandırabilmesi ve video akışı boyunca takip edebilmesi için bir nesne tespit ve takip altyapısı geliştirmektir.
Proje Yaklaşımı
Çalışmanın ilk aşamasında Unreal Engine ortamında sentetik askeri hedef görüntüleri üretilmiştir. Tank, hava savunma sistemi, askeri personel ve askeri araç sınıfları; farklı kamera açıları, mesafeler, yükseklikler, çevre koşulları ve hava durumları altında görüntülenmiştir.
Sentetik veri üretiminde Domain Randomization yaklaşımı kullanılarak modelin tek bir sahneye veya kamera koşuluna bağımlı kalmaması hedeflenmiştir.
## 🎥 Sim-to-Real Video Demonstration

YOLOv8s CLEAN V4 + DeepSORT + Kalman Filter kullanılarak gerçek video üzerinde çoklu nesne tespiti ve takibi gerçekleştirilmiştir.

[![Sim-to-Real Demo](Sim-to-Real/demo_preview.jpg)](Sim-to-Real/video_3_deepsort.mp4)

**Tespit sınıfları:** Tank, Hava Savunma, Askeri Personel, Askeri Araç

▶️ **Görsele tıklayarak videoyu açabilirsiniz.**

Kullanılan sınıflar:
ID	Sınıf
0	Tank
1	Hava Savunma
2	Askeri Personel
3	Askeri Araç

İlk YOLOv8s modeli herhangi bir hazır ağırlık kullanılmadan, tamamen training from scratch yöntemiyle eğitilmiştir. Sentetik validation sonuçları oldukça yüksek olmasına rağmen gerçek görüntüler üzerinde yapılan testlerde performansın aynı seviyede olmadığı görülmüştür.
Bu durum, sentetik ortam ile gerçek dünya görüntüleri arasındaki Sim-to-Real Domain Gap problemini ortaya çıkarmıştır.
Sim-to-Real Çalışması
Sentetik ortamda öğrenilen özelliklerin gerçek görüntülere aktarılabilmesi için model, gerçek askeri görüntülerle kontrollü biçimde fine-tune edilmiştir.
Bu süreçte:
- Farklı veri kaynaklarından alınan sınıflar ortak sınıf yapısına dönüştürüldü.
- Hatalı veya gereksiz bounding box etiketleri temizlendi.
- Image-label eşleşmeleri kontrol edildi.
- Gerçek hava savunma, personel, tank ve zırhlı araç görüntüleri eğitim sürecine dahil edildi.
- Farklı fine-tune sürümleri aynı gerçek test seti üzerinde karşılaştırıldı.
Model geliştirme sürecinde veri kalitesinin doğrudan model davranışını etkilediği görüldü. Özellikle uyumsuz zırhlı araç verileri bir eğitim sürümünde performans kaybına neden olmuş, veri seti temizlenerek yeniden yapılan fine-tune sonrasında bu problem giderilmiştir.
Final Model
Final model olarak YOLOv8s CLEAN V4 kullanılmıştır.
Başlıca validation sonuçları:
Precision    : 0.865
Recall       : 0.849
mAP50        : 0.888
mAP50-95     : 0.617
Parameters   : ~11.1M
Model ayrıca 160 gerçek görüntüden oluşan sabit bir Sim-to-Real test seti üzerinde değerlendirilmiştir.
Sınıf	Gerçek Görüntü Sonucu
Tank	75.0%
Hava Savunma	95.0%
Askeri Personel	62.5%
Askeri Araç	65.0%
Genel	74.4%


Bu oran, ground-truth bounding box tabanlı mAP değeri değildir. Aynı gerçek görüntü havuzunda beklenen sınıfın tespit edilip edilmediğini ölçen class-presence tabanlı Sim-to-Real karşılaştırma sonucudur.

Video Takibi
Model video görüntüleri üzerinde test edildiğinde bazı hedeflerin birkaç kare boyunca tespit edilip ardından kısa süreli kaybolduğu gözlemlenmiştir.
Bu problemi azaltmak için DeepSORT + Kalman Filter yapısı eklenmiştir.
Sistem akışı:
Video
  ↓
YOLOv8 Nesne Tespiti
  ↓
Bounding Box
  ↓
DeepSORT
  ↓
Kalman Filter
  ↓
Track ID ile Sürekli Hedef Takibi
DeepSORT, algılanan hedeflere bir kimlik numarası atarken Kalman Filter, YOLO'nun hedefi birkaç kare boyunca kaçırdığı durumlarda hedefin konumunu tahmin ederek takibin devam etmesini sağlar.
Bu yapı sayesinde video üzerindeki bounding box sürekliliği artırılmış ve kısa süreli tespit kayıpları azaltılmıştır.
Sonuç
Proje kapsamında sentetik veri ile başlayan bir YOLOv8 nesne tespit modeli, gerçek görüntülerle yapılan hedefli fine-tuning sayesinde gerçek dünya koşullarına daha iyi uyarlanmıştır.
Çalışmanın temel çıktıları:
- Unreal Engine tabanlı sentetik veri üretimi
- Domain Randomization
- YOLOv8s training from scratch
- Sim-to-Real Domain Gap analizi
- Dört sınıflı askeri hedef tespiti
- Gerçek görüntüler üzerinde karşılaştırmalı değerlendirme
- DeepSORT + Kalman Filter ile video tabanlı hedef takibi
Bu çalışma, BAP-11 Otonom Sürü Kamikaze İHA Angajmanı Optimizasyonu projesinin görüntü işleme ve hedef algılama altyapısının geliştirilmesine yönelik yürütülmüştür.
