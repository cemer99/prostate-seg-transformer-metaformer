# Prostat Ultrason Görüntülerinde Segmentasyon: Transformer ve MetaFormer Mimarilerinin Karşılaştırmalı Analizi

CRTA-PRONOBIS prostat fantom ultrason veri seti üzerinde **ViT**, **Vanilla Transformer** ve **MetaFormer** ailesi (PoolFormer, CaFormer, ConvFormer, RandFormer, IdentityFormer, FNet) mimarilerini kapsayan **20 model konfigürasyonunun** segmentasyon performansını karşılaştıran bir derin öğrenme projesi. Ayrıca **7 optimizer** ve **8 aktivasyon fonksiyonu** için ablasyon deneyleri yapılmıştır.



---

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Öne Çıkan Sonuçlar](#öne-çıkan-sonuçlar)
- [Veri Seti](#veri-seti)
- [Model Mimarileri](#model-mimarileri)
- [Eğitim Ayarları](#eğitim-ayarları)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Deneysel Sonuçlar](#deneysel-sonuçlar)
- [Çıktı Görselleri](#çıktı-görselleri)
- [Sınırlılıklar ve Gelecek Çalışmalar](#sınırlılıklar-ve-gelecek-çalışmalar)
- [Proje Yapısı](#proje-yapısı)
- [Kaynaklar](#kaynaklar)

---

## Proje Özeti

Transrektal ultrason (TRUS) görüntülerinde prostat sınırlarının elle belirlenmesi zaman alıcıdır ve gözlemciler arası tutarsızlık içerir. Bu çalışma şu soruya cevap arar: **Başarıyı getiren şey öz-dikkat (self-attention) mekanizmasının kendisi mi, yoksa onu çevreleyen MetaFormer çatısı mı?**

Projenin katkıları:

- Prostat ultrason segmentasyonunda ViT, Vanilla Transformer ve 6 MetaFormer varyantını bir arada karşılaştıran kapsamlı bir benchmark.
- Token mixer stratejisinin segmentasyon kalitesine etkisini ayrıştıran ablasyon deneyleri.
- Optimizer ve aktivasyon fonksiyonu seçiminin nicel analizi.
- Parametre sayısı, eğitim süresi ve doğruluk dengesi üzerine pratik tasarım önerileri.

## Öne Çıkan Sonuçlar

| Hedef | Model | Dice | Parametre |
|---|---|---|---|
| En yüksek doğruluk | **ViT-Wide** | **0.9809** | 19.74M |
| Doğruluk/verimlilik dengesi | Vanilla Transformer | 0.9786 | 5.17M |
| Doğruluk/verimlilik dengesi | MetaAttn-GELU | 0.9775 | 5.19M |
| Kısıtlı donanım (mobil/gömülü) | ViT-Small | 0.9591 | 1.06M |

- Attention ve evrişim tabanlı token mixer'lar en iyi sonuçları verirken, havuzlama (pooling) tabanlı MetaPool modelleri daha düşük doğrulukta kalmıştır.
- Rastgele matris kullanan RandFormer'ın bile rekabetçi sonuç vermesi, MetaFormer çatısının tek başına güçlü olduğunu gösterir.
- ViT mimarilerinde Adam, AdamW ve NAdam birbirine çok yakın ve en iyi sonuçları vermiştir. SGD tüm mimarilerde yetersiz kalmıştır.
- RMSprop ve Adadelta küçük modellerde başarılı, büyük modelde (ViT-Large) çökmüştür.
- Aktivasyon fonksiyonunun etkisi sınırlıdır (MetaFormer-Pool üzerinde en iyi ve en kötü arasındaki fark yalnızca ~0.015 Dice).

## Veri Seti

[CRTA-PRONOBIS Prostate Phantom Ultrasound Dataset](https://www.kaggle.com/datasets/lukaiktar/crta-pronobis-prostate-phantom-ultrasound-dataset) (Kaggle). Standart TRUS protokolüyle elde edilmiş prostat fantom görüntüleri ve piksel düzeyinde ikili segmentasyon maskelerini içerir.

Ön işleme:

- Görüntüler gri tonlamaya çevrilir ve **128×128** boyutuna yeniden boyutlandırılır.
- Piksel değerleri **[0, 1]** aralığına normalize edilir.
- Maskeler **0.5 eşiğiyle** ikili hâle getirilir.
- Veri **%80 eğitim / %20 doğrulama** olarak bölünür (seed = 42).

Notebook, `Train/augmentedDataset` ve `Train/augmentedDatasetMasks` klasörlerini kullanır.

## Model Mimarileri

Tüm modeller ortak yapıyı paylaşır: **Conv2D patch embedding (patch=16) → token mixer / encoder → SegDecoder** (2 adet 3×3 konv. + 1×1 konv., bilinear yukarı örnekleme).

### ViT (öğrenilebilir konum gömmesi)

| Model | Dim | Depth | Heads | Patch |
|---|---|---|---|---|
| ViT-Base | 256 | 6 | 8 | 16 |
| ViT-Small | 128 | 4 | 4 | 16 |
| ViT-Deep | 256 | 8 | 8 | 16 |
| ViT-Wide | 512 | 6 | 8 | 16 |

### Vanilla Transformer

Orijinal "Attention Is All You Need" yapısına sadık: sinüzoidal (öğrenilmeyen) konum kodlaması, Post-LN. `Vanilla-6` (depth=6) ve `Vanilla-8` (depth=8), ikisi de dim=256, heads=8.

### MetaFormer ailesi

Her blok: `Norm → Token Mixer → (+ artık bağlantı) → Norm → Kanal MLP → (+ artık bağlantı)`. MLP genişletme oranı 4.0, aktivasyon GELU.

| Varyant | Token Mixer |
|---|---|
| PoolFormer | 3 tokenlık ortalama havuzlama (`AvgPool1d`) |
| CaFormer | Multi-head self-attention |
| ConvFormer | 7 çekirdekli depthwise evrişim |
| RandFormer | Sabit rastgele karıştırma matrisi |
| FNet | 2B FFT (Fourier) karıştırma |
| IdentityFormer | Mixer yok (kontrol grubu, sadece MLP) |

## Eğitim Ayarları

| Parametre | Değer |
|---|---|
| Çerçeve | PyTorch |
| Donanım | Google Colab – NVIDIA Tesla T4 |
| Girdi boyutu | 128 × 128 |
| Batch size | 4 |
| Epoch | 15 (ana karşılaştırma), 20 (optimizer ablasyonu) |
| Başlangıç öğrenme oranı | 1e-4 |
| LR zamanlayıcı | Cosine Annealing (`T_max = epochs`) |
| Kayıp fonksiyonu | `0.3 × BCE + 0.7 × Dice Loss` |
| Metrikler | Dice, IoU (Jaccard) |
| Rastgelelik tohumu | 42 |

Optimizer ayarları: AdamW `weight_decay=1e-2`, SGD `momentum=0.9` + Nesterov, RMSprop `alpha=0.99`, Adadelta `rho=0.9`; Adam, NAdam ve Adagrad varsayılan ayarlarla kullanılmıştır.

## Kurulum

### Seçenek 1: Google Colab (önerilen)

1. `prostate_seg.ipynb` dosyasını [Google Colab](https://colab.research.google.com/)'a yükleyin veya GitHub üzerinden açın.
2. **Çalışma Zamanı → Çalışma zamanı türünü değiştir → T4 GPU** seçin.
3. Notebook'un ilk hücrelerini çalıştırarak bağımlılıkları kurun.

### Seçenek 2: Yerel ortam

Python 3.9+ ve CUDA destekli bir GPU önerilir (CPU'da çalışır ancak çok yavaştır).

```bash
git clone https://github.com/<kullanici-adi>/<repo-adi>.git
cd <repo-adi>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision numpy pillow matplotlib jupyter kaggle
```

> Notebook'un ilk hücresi ek olarak `vit_pytorch`, `einops` ve `timm` kurar. Mevcut kodda bu kütüphaneler kullanılmamaktadır; modeller doğrudan PyTorch ile yazılmıştır.

### Veri setini indirme

1. Kaggle hesabınızdan **Settings → API → Create New Token** ile `kaggle.json` dosyasını indirin.
2. Dosyayı yerleştirin ve veri setini indirin:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d lukaiktar/crta-pronobis-prostate-phantom-ultrasound-dataset
unzip -q crta-pronobis-prostate-phantom-ultrasound-dataset.zip
```

3. Notebook içindeki veri yollarını kendi dizin yapınıza göre güncelleyin:

```python
TRAIN_IMG  = "/content/ProstateUltrasoundDatasetPRONOBIS/Train/augmentedDataset"
TRAIN_MASK = "/content/ProstateUltrasoundDatasetPRONOBIS/Train/augmentedDatasetMasks"
```

> ⚠️ `kaggle.json` dosyanızı **asla** GitHub'a yüklemeyin.

## Çalıştırma

Notebook'u baştan sona sırayla çalıştırın:

| Bölüm | İçerik |
|---|---|
| 1–2 | Kurulum, kütüphaneler, veri yükleme |
| 3 | Kayıp fonksiyonları ve metrikler (Dice, IoU) |
| 4 | Model tanımları (ViT, Vanilla, MetaFormer) |
| 5 | Eğitim ve değerlendirme fonksiyonları |
| 6 | **Deney 1:** 20 konfigürasyonlu ana mimari karşılaştırması |
| 7 | **Deney 2:** Optimizer karşılaştırması |
| 8 | **Deney 3:** Aktivasyon fonksiyonu karşılaştırması (MetaFormer-Pool) |
| 9–10 | Sonuç tabloları ve görselleştirmeler |
| Ek | ViT (Base/Small/Large/Tiny) ve MetaFormer mimarileri × 7 optimizer ablasyonu |

Daha hızlı deneme için `EPOCHS` değişkenini düşürebilirsiniz. Notebook'ta tam eğitim için 30–50 epoch önerilmektedir.

## Deneysel Sonuçlar

### 1. Ana mimari karşılaştırması (20 model, Dice'a göre sıralı)

| Sıra | Model | Dice | IoU | Parametre | Süre (s) |
|---|---|---|---|---|---|
| 1 | 4_ViT_Wide | **0.9809** | 0.9626 | 19.74M | 361.6 |
| 2 | 5_Vanilla_Trans | 0.9786 | 0.9581 | 5.17M | 328.1 |
| 3 | 15_MetaAttn_GELU | 0.9775 | 0.9560 | 5.19M | 305.7 |
| 4 | 6_Vanilla_Deep | 0.9768 | 0.9548 | 6.75M | 320.9 |
| 5 | 16_ViT_Mish | 0.9755 | 0.9522 | 5.19M | 306.4 |
| 6 | 3_ViT_Deep | 0.9750 | 0.9514 | 6.77M | 331.3 |
| 7 | 19_ViT_AdamW | 0.9750 | 0.9512 | 5.19M | 315.0 |
| 8 | 17_ViT_SiLU | 0.9742 | 0.9498 | 5.19M | 307.8 |
| 9 | 9_MetaAttn_CaFormer | 0.9742 | 0.9498 | 5.19M | 309.8 |
| 10 | 18_ViT_Adam | 0.9740 | 0.9494 | 5.19M | 309.4 |
| 11 | 1_ViT_Base | 0.9731 | 0.9477 | 5.19M | 336.5 |
| 12 | 11_MetaConv_ConvFormer | 0.9652 | 0.9331 | 3.62M | 302.6 |
| 13 | 2_ViT_Small | 0.9591 | 0.9218 | 1.06M | 319.1 |
| 14 | 12_MetaIdentity | 0.9544 | 0.9137 | 3.61M | 304.1 |
| 15 | 10_MetaRand_RandFormer | 0.9539 | 0.9121 | 3.61M | 302.2 |
| 16 | 13_MetaFNet | 0.9438 | 0.8945 | 3.61M | 301.6 |
| 17 | 7_MetaPool | 0.9162 | 0.8464 | 3.61M | 306.9 |
| 18 | 8_MetaPool_Deep | 0.9113 | 0.8381 | 4.66M | 311.3 |
| 19 | 14_MetaPool_GELU | 0.8946 | 0.8105 | 3.61M | 299.5 |
| 20 | 20_ViT_SGD | 0.7756 | 0.6367 | 5.19M | 318.9 |

**Yorumlar**

- ViT-Wide en yüksek Dice'a ulaşırken en yüksek parametre sayısına (19.74M) ve en uzun eğitim süresine sahiptir.
- Vanilla Transformer ve MetaAttn-GELU, ViT-Wide'ın yaklaşık dörtte biri kadar parametreyle çok yakın doğruluk sunar.
- ViT-Small yalnızca 1.06M parametreyle 0.9591 Dice elde ederek kaynak kısıtlı sistemler için uygun bir seçenektir.
- ViT-SGD en düşük performansı (0.7756) göstermiştir.

### 2. Optimizer karşılaştırması (ViT, Dice)

ViT mimarileri: Base (dim 256 / depth 6 / heads 8), Small (192 / 4 / 6), Large (384 / 12 / 12), Tiny (128 / 3 / 4). Eğitim 20 epoch.

| Optimizer | ViT-Base | ViT-Small | ViT-Large | ViT-Tiny | Ortalama |
|---|---|---|---|---|---|
| Adam | 0.9861 | 0.9828 | 0.9870 | 0.9777 | 0.9834 |
| AdamW | 0.9859 | 0.9837 | 0.9869 | 0.9780 | 0.9836 |
| NAdam | 0.9861 | 0.9829 | **0.9874** | 0.9781 | 0.9836 |
| Adadelta | 0.9863 | 0.9861 | 0.8594 | 0.9854 | 0.9543 |
| RMSprop | 0.9866 | 0.9843 | 0.6177 | 0.9789 | 0.8919 |
| Adagrad | 0.9301 | 0.9247 | 0.9432 | 0.9093 | 0.9268 |
| SGD | 0.8597 | 0.8510 | 0.8953 | 0.8422 | 0.8620 |

**Yorumlar**

- Adam, AdamW ve NAdam, tüm model boyutlarında kararlı ve birbirine çok yakın sonuçlar verir.
- RMSprop ve Adadelta küçük/orta modellerde iyidir ancak ViT-Large'da belirgin biçimde çöker (RMSprop: 0.6177, Adadelta: 0.8594).
- SGD (momentum + Nesterov, lr=1e-4), 20 epoch'luk eğitimde tüm modellerde yetersiz kalmıştır.

### 3. Aktivasyon fonksiyonu karşılaştırması (MetaFormer-Pool, Dice / IoU)

| Aktivasyon | Dice | IoU |
|---|---|---|
| LeakyReLU | **0.9091** | **0.8343** |
| GELU | 0.9073 | 0.8314 |
| PReLU | 0.9068 | 0.8305 |
| SiLU | 0.9055 | 0.8283 |
| Mish | 0.9048 | 0.8272 |
| ReLU | 0.8983 | 0.8164 |
| ELU | 0.8941 | 0.8097 |
| Hardswish | 0.8937 | 0.8088 |

Tüm fonksiyonlar arasındaki maksimum fark Dice'ta ~0.015, IoU'da ~0.026'dır; yani aktivasyon seçimi mimarinin performansını belirgin biçimde değiştirmemektedir. LeakyReLU en iyi sonucu vermiş, GELU ise performans/maliyet dengesi açısından makul bir seçenek olarak öne çıkmıştır.

### Genel değerlendirme

Her durum için geçerli tek bir "en iyi" mimari yoktur:

- **Maksimum doğruluk** için → ViT-Wide
- **Verimlilik** için → Vanilla Transformer veya MetaAttn-GELU
- **Çok kısıtlı donanım** için → ViT-Small

## Çıktı Görselleri

Notebook çalıştırıldığında aşağıdaki grafikler oluşturulur:

| Dosya | İçerik |
|---|---|
| `mimari_karsilastirma.png` | 20 modelin Dice / IoU / parametre / süre karşılaştırması |
| `egitim_egrileri_top5.png` | En iyi 5 modelin eğitim eğrileri |
| `optimizer_karsilastirma.png` | Optimizer karşılaştırması |
| `optimizer_egitim_egrileri.png` | Optimizer eğitim eğrileri |
| `aktivasyon_karsilastirma.png` | Aktivasyon fonksiyonu karşılaştırması |
| `aktivasyon_egitim_egrileri.png` | Aktivasyon eğitim eğrileri |
| `params_vs_dice.png` | Parametre sayısı – Dice ilişkisi |
| `gorsel_karsilastirma.png` | Tahmin edilen maskelerin görsel karşılaştırması |
| `metaformer_karsilastirma.png` | MetaFormer token mixer karşılaştırması |
| `tum_modeller_karsilastirma.png` | Tüm modellerin genel karşılaştırması |

Bu görselleri repoya `images/` klasörü altında ekleyip README'ye şu şekilde dahil edebilirsiniz:

```markdown
![Mimari karşılaştırma](images/mimari_karsilastirma.png)
```

## Sınırlılıklar ve Gelecek Çalışmalar

- Çalışma **fantom** (gerçek hasta olmayan) ve küçük bir veri setiyle yapılmıştır.
- Eğitim süresi kısadır (15–20 epoch); sonuçlar tek koşuya dayanır (tekrar/istatistiksel anlamlılık analizi yoktur).
- Doğrulama kümesi hem model seçimi hem raporlama için kullanılmıştır; ayrı bir test kümesi bulunmamaktadır.
- Gelecek çalışma önerileri: gerçek klinik TRUS görüntüleri, veri artırma teknikleri, ön eğitimli (pretrained) modeller ve çok merkezli doğrulama.

## Proje Yapısı

```
.
├── prostate_seg.ipynb                       # Tüm deneyleri içeren ana notebook
├── CEM_ER_2515514026_Derin_Ogrenme.docx     # Proje raporu
├── images/                                  # (isteğe bağlı) çıktı grafikleri
└── README.md
```

## Kaynaklar

- A. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* — https://arxiv.org/abs/2010.11929
- W. Yu et al., *MetaFormer Is Actually What You Need for Vision* — https://arxiv.org/abs/2111.11418
- J. Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, 2021
- Z. Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021
- O. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015
- S. Abudalou et al., *Improving Prostate Gland Segmentation Using Transformer based Architectures* — https://arxiv.org/abs/2506.14844
- CRTA-PRONOBIS Prostate Phantom Ultrasound Dataset — https://www.kaggle.com/datasets/lukaiktar/crta-pronobis-prostate-phantom-ultrasound-dataset


