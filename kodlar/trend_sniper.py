# -*- coding: utf-8 -*-
"""
Trend Sniper (Haar Wavelet) — Python sürümü, yalnız long
=========================================================

LİSANS VE ATIF
--------------
Bu dosya, Jamallo'nun TradingView'de açık kaynak olarak yayımladığı
"Trend Sniper v2.5 - Haar Wavelet Edition [Jamallo]" göstergesinin hesaplama
mantığından türetilmiştir. Orijinal kod Mozilla Public License 2.0 (MPL-2.0)
lisansı altındadır; bu dosya da MPL-2.0 ile dağıtılır:
https://mozilla.org/MPL/2.0/
Orijinal yazar: Jamallo. Python'a aktarma ve giriş/çıkış kuralları: MMFI.

UYARI
-----
MMFI — eğitim ve araştırma amaçlıdır, yatırım tavsiyesi değildir.
Geçmiş sonuçlar geleceği garanti etmez. Kendi kararınızı kendiniz verin.

VİDEODAKİ ÖLÇÜMÜN KOŞULLARI
---------------------------
- Evren: BIST'te en likit 100 hisse (her gün son 250 günün ortanca TL hacmine göre).
- Sepet: 10 eşit yuva, adaylar 6 aylık momentuma göre sıralanır (en güçlü önce).
- Sinyal gün kapanışında, işlem ERTESİ GÜN AÇILIŞINDA; komisyon dahil (tek yön %0,09).
- Geliştirme dönemi 2011-06 → 2022, kilitli (sonradan okunan) dönem 2023 → bugün.
- Kilitli dönemde yıllık getiri %46,3 (Sharpe 1,51, en büyük düşüş −%42);
  aynı dönemde eşit ağırlıklı al-tut %9,9.
- Bu dosyayı TEK HİSSEDE çalıştırdığınızda sonuç FARKLI çıkar (sepet ve sıralama yok,
  farklı dönem, farklı veri kaynağı). Tek hisse sonucu stratejinin karnesi değildir.

KURALLAR (özet)
---------------
Göstergeler Heikin-Ashi mumlar üzerinde hesaplanır:
  * haar_trend : Haar dalgacığı (3. seviye) ile yumuşatılmış fiyat, "ölü bant" aşılınca
                 basamak atlar (aşılmazsa yatay kalır).
  * st_yavas   : yavaş SuperTrend (Heikin-Ashi, ATR 14, çarpan 9).
GİRİŞ  : (haar_trend st_yavas'ı yukarı keser)  VEYA  (trend yukarıyken haar_trend yukarı basamak atar)
         VE 12 aylık momentum > 0 VE kapanış > 200 günlük ortalama.
ÇIKIŞ  : haar_trend aşağı basamak atar VEYA st_yavas'ın altına iner.
Zarar durdur yok (çıkış sinyali bekler). Kaldıraç yok, açığa satış yok.

KULLANIM
--------
  pip install pandas numpy yfinance        (yfinance isteğe bağlı)
  python trend_sniper.py                   -> THYAO.IS örneği (yfinance ile indirir)
  python trend_sniper.py veri.csv          -> kendi CSV'niz (sütunlar: date,open,high,low,close)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# ─── Ayarlar (orijinal göstergenin varsayılanları) ───────────────────────────
HAAR_SEVIYE = 3        # Haar ayrıştırma seviyesi (3 = 8 barlık ölçek)
OLU_BANT_KAT = 2.0     # ölü bant çarpanı
ENERJI_UZ = 100        # ayrıntı enerjisi penceresi
ST_ATR_UZ = 14         # SuperTrend ATR uzunluğu
ST_CARPAN = 9.0        # yavaş SuperTrend çarpanı
ISINMA_BAR = 60        # ilk barlarda sinyal üretilmez (göstergeler otursun)
MOM_UZUN = 252         # 12 ay momentum (filtre)
MOM_KISA = 126         # 6 ay momentum (sepette aday sırası)
SMA_UZ = 200


# ─── Yardımcılar ─────────────────────────────────────────────────────────────

def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder ortalaması (Pine ta.rma). NaN değerleri atlar; ilk değer basit ortalamadır."""
    out = np.full(len(x), np.nan)
    toplam, adet = 0.0, 0
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        if adet < n:
            toplam += x[i]
            adet += 1
            if adet == n:
                out[i] = toplam / n
        else:
            out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def heikin_ashi(o, h, l, c):
    """Düz Heikin-Ashi mumları: (ha_acilis, ha_yuksek, ha_dusuk, ha_kapanis)."""
    hc = (o + h + l + c) / 4.0
    ho = np.empty(len(c))
    ho[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
    hh = np.maximum(h, np.maximum(ho, hc))
    hl = np.minimum(l, np.minimum(ho, hc))
    return ho, hh, hl, hc


def haar_trend(hc: np.ndarray, seviye: int = HAAR_SEVIYE, kat: float = OLU_BANT_KAT,
               enerji_uz: int = ENERJI_UZ) -> np.ndarray:
    """Haar dalgacığı ile yumuşatma + uyarlanır ölü bant (basamaklı çizgi)."""
    n = len(hc)
    v = hc.copy()
    d = np.zeros(n)
    gecikme = 1
    for _ in range(seviye):
        onceki = np.concatenate([v[:gecikme], v[:-gecikme]]) if n > gecikme else v.copy()
        v, d = (v + onceki) / 2.0, (v - onceki) / 2.0
        gecikme *= 2
    # ölü bant = son enerji_uz barın |ayrıntı| ortalaması x kat (Pine: ta.sma(math.abs(detail), 100) * 2)
    esik = np.full(n, np.nan)
    toplam = 0.0
    for i in range(n):
        toplam += abs(d[i])
        if i >= enerji_uz:
            toplam -= abs(d[i - enerji_uz])
        if i >= enerji_uz - 1:
            esik[i] = toplam / enerji_uz * kat
    ht = np.empty(n)
    ht[0] = v[0]
    for i in range(1, n):
        if not np.isnan(esik[i]) and abs(v[i] - ht[i - 1]) > esik[i]:
            ht[i] = v[i]          # ölü bant aşıldı: basamak
        else:
            ht[i] = ht[i - 1]     # aşılmadı: yatay
    return ht


def supertrend_ha(hh, hl, hc, carpan: float = ST_CARPAN, atr_uz: int = ST_ATR_UZ) -> np.ndarray:
    """Heikin-Ashi mumlarıyla SuperTrend (Pine'daki yavaş SuperTrend ile aynı)."""
    n = len(hc)
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum(hh[1:] - hl[1:], np.maximum(np.abs(hh[1:] - hc[:-1]), np.abs(hl[1:] - hc[:-1])))
    atr = rma(tr, atr_uz)
    orta = (hh + hl) / 2.0
    ust = orta - carpan * atr      # yukarı trendde destek
    alt = orta + carpan * atr      # aşağı trendde direnç
    st = np.full(n, np.nan)
    yon = np.ones(n, dtype=int)
    st_onceki = 0.0
    for i in range(n):
        if i > 0:
            u1 = 0.0 if np.isnan(ust[i - 1]) else ust[i - 1]
            d1 = 0.0 if np.isnan(alt[i - 1]) else alt[i - 1]
            if u1 > 0 and hc[i - 1] > st_onceki:
                ust[i] = max(ust[i], u1)
            if d1 > 0 and hc[i - 1] < st_onceki:
                alt[i] = min(alt[i], d1)
            if hc[i] > d1:
                yon[i] = 1
            elif hc[i] < u1:
                yon[i] = -1
            else:
                yon[i] = yon[i - 1]
        st[i] = ust[i] if yon[i] == 1 else alt[i]
        st_onceki = 0.0 if np.isnan(st[i]) else st[i]
    return st


# ─── Sinyaller ───────────────────────────────────────────────────────────────

def sinyaller(df: pd.DataFrame) -> pd.DataFrame:
    """df: open/high/low/close sütunlu, tarih sıralı günlük veri.

    Dönüş sütunları:
      haar, st        : göstergeler
      giris_ham       : kesişim ya da trend içi yukarı basamak
      filtre          : 12 ay momentum > 0 ve kapanış > SMA200
      al              : giris_ham ve filtre  (ertesi açılışta alınır)
      sat             : çıkış sinyali        (ertesi açılışta satılır)
      mom6            : 6 aylık momentum (sepette sıralama için)
    """
    d = df[["open", "high", "low", "close"]].astype(float).dropna()
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    _, hh, hl, hc = heikin_ashi(o, h, l, c)
    ht = haar_trend(hc)
    st = supertrend_ha(hh, hl, hc)

    ht1 = np.r_[np.nan, ht[:-1]]
    st1 = np.r_[np.nan, st[:-1]]
    gecerli = ~np.isnan(st) & ~np.isnan(st1)
    gecerli[:ISINMA_BAR] = False
    yukari = ht > st
    kesisim = yukari & (ht1 <= st1)
    basamak_yukari = yukari & (ht > ht1)
    giris = gecerli & (kesisim | basamak_yukari)
    sat = gecerli & ((ht < ht1) | (ht < st))

    kapanis = pd.Series(c, index=d.index)
    mom12 = kapanis / kapanis.shift(MOM_UZUN) - 1
    sma200 = kapanis.rolling(SMA_UZ).mean()
    filtre = ((mom12 > 0) & (kapanis > sma200)).values

    return pd.DataFrame({
        "haar": ht, "st": st, "giris_ham": giris, "filtre": filtre,
        "al": giris & filtre, "sat": sat,
        "mom6": (kapanis / kapanis.shift(MOM_KISA) - 1).values,
    }, index=d.index)


# ─── Tek hisse backtest ──────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, komisyon: float = 0.0009) -> dict:
    """Basit tek hisse testi: sinyal kapanışta, işlem ertesi açılışta, tüm para tek hissede.

    komisyon: tek yön oran (0.0009 = %0,09).
    """
    d = df[["open", "high", "low", "close"]].astype(float).dropna()
    s = sinyaller(d)
    o, c = d["open"].values, d["close"].values
    al, sat = s["al"].values, s["sat"].values
    para, adet, pozisyonda = 1.0, 0.0, False
    giris_fiyat = 0.0
    islemler, ozsermaye = [], np.empty(len(c))
    for t in range(len(c)):
        if t > 0:
            if pozisyonda and sat[t - 1]:
                para = adet * o[t] * (1 - komisyon)
                islemler.append({"cikis_tarih": d.index[t], "getiri": o[t] * (1 - komisyon) / giris_fiyat - 1})
                adet, pozisyonda = 0.0, False
            elif not pozisyonda and al[t - 1]:
                giris_fiyat = o[t] * (1 + komisyon)
                adet = para / giris_fiyat
                islemler.append({"giris_tarih": d.index[t]})
                para, pozisyonda = 0.0, True
        ozsermaye[t] = para + adet * c[t]
    eg = pd.Series(ozsermaye, index=d.index)
    kapali = [x for x in islemler if "getiri" in x]
    yil = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    gunluk = eg.pct_change().dropna()
    return {
        "ozsermaye": eg,
        "toplam_getiri": eg.iloc[-1] - 1,
        "yillik_getiri": eg.iloc[-1] ** (1 / yil) - 1,
        "al_tut_yillik": (c[-1] / c[0]) ** (1 / yil) - 1,
        "en_buyuk_dusus": (eg / eg.cummax() - 1).min(),
        "sharpe": gunluk.mean() / gunluk.std() * np.sqrt(252) if gunluk.std() > 0 else 0.0,
        "islem_sayisi": len(kapali),
        "isabet": np.mean([x["getiri"] > 0 for x in kapali]) if kapali else float("nan"),
        "acik_pozisyon": pozisyonda,
    }


# ─── Çok hisseli 10 yuvalı sepet (isteğe bağlı) ──────────────────────────────

def sepet_backtest(veriler: dict, yuva: int = 10, komisyon: float = 0.0009) -> pd.Series:
    """veriler: {sembol: DataFrame(open, high, low, close)}. Günlük getiri serisi döner.

    Her gün: önce çıkış sinyali verenler açılışta satılır; sonra boş yuvalara AL sinyali
    verenler 6 aylık momentum sırasıyla (güçlü önce) açılışta alınır (bugün satılan
    hissenin yuvası ertesi gün boşalır). Her yeni alım
    o anki toplam değerin 1/yuva'sı kadardır (nakit yetmezse kalan nakit).
    """
    sin = {k: sinyaller(v) for k, v in veriler.items()}
    idx = sorted(set().union(*[v.index for v in sin.values()]))
    idx = pd.DatetimeIndex(idx)
    sem = list(sin)
    O = pd.DataFrame({k: veriler[k]["open"].astype(float) for k in sem}).reindex(idx).values
    C = pd.DataFrame({k: veriler[k]["close"].astype(float) for k in sem}).reindex(idx).values
    # sinyal t-1 kapanışında → t açılışında işlem (her hissenin KENDİ bir önceki işlem günü)
    AL = np.zeros(O.shape, bool)
    SAT = np.zeros(O.shape, bool)
    SIRA = np.full(O.shape, np.nan)
    for j, k in enumerate(sem):
        s = sin[k]
        konum = idx.get_indexer(s.index)
        a = np.where(s["al"].values[:-1])[0]
        AL[konum[a + 1], j] = True
        SIRA[konum[a + 1], j] = s["mom6"].values[a]
        x = np.where(s["sat"].values[:-1])[0]
        SAT[konum[x + 1], j] = True

    nakit = 1.0
    tutulan = {}            # sembol sırası -> [değer, son fiyat]
    getiri = np.zeros(len(idx))
    for t in range(1, len(idx)):
        onceki_deger = nakit + sum(v[0] for v in tutulan.values())
        bugun_cikan = 0     # bugün boşalan yuva ancak yarın kullanılır
        for j in list(tutulan):
            if np.isnan(C[t, j]) or np.isnan(O[t, j]):
                continue
            deger, ref = tutulan[j]
            if SAT[t, j]:
                nakit += deger * O[t, j] * (1 - komisyon) / ref
                del tutulan[j]
                bugun_cikan += 1
            elif t == len(idx) - 1:
                nakit += deger * C[t, j] * (1 - komisyon) / ref
                del tutulan[j]
                bugun_cikan += 1
            else:
                tutulan[j] = [deger * C[t, j] / ref, C[t, j]]
        adaylar = np.where(AL[t])[0]
        puan = np.nan_to_num(SIRA[t, adaylar], nan=-1e18)
        for j in adaylar[np.argsort(-puan, kind="stable")]:
            if j in tutulan or len(tutulan) + bugun_cikan >= yuva:
                continue
            if np.isnan(O[t, j]) or np.isnan(C[t, j]):
                continue
            tutar = min(onceki_deger / yuva, nakit)
            if tutar <= 0:
                break
            nakit -= tutar
            g = O[t, j] * (1 + komisyon)
            tutulan[j] = [tutar * C[t, j] / g, C[t, j]]
        yeni_deger = nakit + sum(v[0] for v in tutulan.values())
        getiri[t] = yeni_deger / onceki_deger - 1
    return pd.Series(getiri, index=idx)


# ─── Veri ────────────────────────────────────────────────────────────────────

def veri_indir(sembol: str, baslangic: str = "2010-01-01") -> pd.DataFrame:
    """yfinance ile günlük veri (BIST için sonuna .IS ekleyin: THYAO.IS)."""
    import yfinance as yf
    d = yf.download(sembol, start=baslangic, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    return d


def csv_oku(yol: str) -> pd.DataFrame:
    d = pd.read_csv(yol)
    d.columns = [str(x).strip().lower() for x in d.columns]
    tarih = next(x for x in d.columns if x in ("date", "tarih", "time", "datetime"))
    d[tarih] = pd.to_datetime(d[tarih])
    return d.set_index(tarih).sort_index()[["open", "high", "low", "close"]]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        veri = csv_oku(sys.argv[1])
        ad = sys.argv[1]
    else:
        ad = "THYAO.IS"
        veri = veri_indir(ad)
    sonuc = backtest(veri, komisyon=0.0009)
    print(f"{ad}: {veri.index[0].date()} -> {veri.index[-1].date()}")
    print(f"  Yıllık getiri     : %{sonuc['yillik_getiri'] * 100:.1f}  (al-tut %{sonuc['al_tut_yillik'] * 100:.1f})")
    print(f"  En büyük düşüş    : %{sonuc['en_buyuk_dusus'] * 100:.1f}")
    print(f"  Sharpe            : {sonuc['sharpe']:.2f}")
    print(f"  İşlem / isabet    : {sonuc['islem_sayisi']} / %{sonuc['isabet'] * 100:.0f}")
    son = sinyaller(veri).iloc[-1]
    print(f"  Son gün           : {'AL' if son['al'] else 'SAT' if son['sat'] else 'bekle'}"
          f"  (pozisyonda: {'evet' if sonuc['acik_pozisyon'] else 'hayır'})")
    print("Not: tek hisse sonucu videodaki 10 yuvalı sepet ölçümünden farklıdır. Yatırım tavsiyesi değildir.")
