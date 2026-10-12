# -*- coding: utf-8 -*-
"""
Onaylı Likidite Süpürmesi — Python sürümü, yalnız long
=======================================================

FİKİR KAYNAĞI VE LİSANS
-----------------------
Fikir: BigBeluga'nın TradingView'de yayımladığı "Keltner Channel Trend + SMC
Liquidity Sweep [BigBeluga]" göstergesi (önemli dibin altına fitil atıp üstünde
kapanan mum = "satış tarafı likidite süpürmesi"). Teşekkürler BigBeluga.
Bu dosya o göstergenin KODUNDAN TÜRETİLMEMİŞTİR: MMFI tarafından sıfırdan yazıldı,
orijinalden satır alınmadı, Keltner kanalı ve gösterge çizimleri kullanılmaz.
Onay mumu, zarar durdur, momentum filtresi, iz süren stop ve sepet kuralları MMFI'nin
kendi ölçümünden gelir. Orijinal göstergenin kendi lisans koşulları kendi kodu için
geçerlidir; onu kullanacaksanız TradingView sayfasındaki koşullara bakın.
Bu dosyayı eğitim amaçlı serbestçe kullanabilir ve değiştirebilirsiniz; paylaşırken
"MMFI" ve fikir kaynağı olarak BigBeluga'yı belirtin.

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
- Kilitli dönemde yıllık getiri %28,0 (Sharpe 1,25, en büyük düşüş −%26);
  aynı dönemde eşit ağırlıklı al-tut %9,9 (Sharpe 0,48, en büyük düşüş −%38).
- Bu dosyayı TEK HİSSEDE çalıştırdığınızda sonuç FARKLI çıkar (sepet ve sıralama yok,
  farklı dönem, farklı veri kaynağı). Tek hisse sonucu stratejinin karnesi değildir.

KURALLAR (özet)
---------------
  * Dip (pivot)   : solundaki 5 mumun dibinden aşağıda, sağındaki 2 mumun dibinden
                    aşağıda ya da eşit olan mum dibi. 2 mum sonra kesinleşir (ileriyi görmez).
  * Süpürme       : mumun fitili bekleyen bir dibin ALTINA iner ama kapanış dibin ÜSTÜNDE.
                    Fiyatın altına indiği dip (süpürülse de kırılsa da) listeden silinir.
  * Onay          : süpürmeden sonraki 3 mum içinde kapanış süpürme mumunun TEPESİNİ aşar.
GİRİŞ  : onay mumu VE 12 aylık momentum > 0 VE kapanış > 200 günlük ortalama
         → ertesi gün açılışta al.
STOP   : süpürme mumunun dibi − 1 ATR(14) (onay günündeki ATR).
ÇIKIŞ  : 3 ATR iz süren stop (girişten beri en yüksek kapanış − 3 × ATR) ya da 250 mum.
Kaldıraç yok, açığa satış yok.

KULLANIM
--------
  pip install pandas numpy yfinance        (yfinance isteğe bağlı)
  python onayli_supurme.py                 -> THYAO.IS örneği (yfinance ile indirir)
  python onayli_supurme.py veri.csv        -> kendi CSV'niz (sütunlar: date,open,high,low,close)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# ─── Ayarlar (ölçülen ve donmuş seçim) ──────────────────────────────────────
PIVOT_SOL = 5          # dip için soldaki mum sayısı
PIVOT_SAG = 2          # dip için sağdaki mum sayısı (dip bu kadar mum sonra kesinleşir)
ONAY_BAR = 3           # süpürmeden sonra onay için beklenen en çok mum
STOP_ATR = 1.0         # stop = süpürme dibi − STOP_ATR × ATR
IZ_ATR = 3.0           # iz süren stop mesafesi (ATR)
MAX_BAR = 250          # en uzun tutma süresi (mum)
ATR_UZ = 14
ISINMA_BAR = 19        # ilk 19 mumda hiçbir şey yapılmaz (ölçümdeki 20 mumluk ısınma ile aynı)
MAX_SEVIYE = 40        # aynı anda izlenen en çok dip
MOM_UZUN = 252         # 12 ay momentum (filtre)
MOM_KISA = 126         # 6 ay momentum (sepette aday sırası)
SMA_UZ = 200


# ─── Yardımcılar ─────────────────────────────────────────────────────────────

def atr_hesap(h, l, c, n: int = ATR_UZ) -> np.ndarray:
    """ATR (Wilder yumuşatması, ilk mumdan başlar)."""
    oc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - oc), np.abs(l - oc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values


def supurme_onay(h, l, c, atr, sol: int = PIVOT_SOL, sag: int = PIVOT_SAG,
                 onay_bar: int = ONAY_BAR, stop_atr: float = STOP_ATR):
    """Onaylı süpürme sinyali (bool dizi) ve stop seviyesi (sinyal mumunda)."""
    n = len(c)
    sinyal = np.zeros(n, bool)
    stop = np.full(n, np.nan)
    seviyeler = [np.nan] * MAX_SEVIYE      # bekleyen dipler
    bek_tepe = bek_dip = 0.0               # son süpürme mumunun tepesi / dibi
    bek_son = -1                           # onay için son geçerli mum
    for i in range(max(sol + sag + 2, ISINMA_BAR), n):
        # 1) onay: bekleyen süpürme varsa ve kapanış süpürme mumunun tepesini aştıysa
        if bek_son >= i and c[i] > bek_tepe:
            sinyal[i] = True
            stop[i] = bek_dip - stop_atr * atr[i]
            bek_son = -1
        # 2) süpürme: fitil bekleyen dibin altına, kapanış üstüne
        supurdu = False
        for q in range(MAX_SEVIYE):
            x = seviyeler[q]
            if np.isnan(x) or l[i] >= x:
                continue
            if c[i] > x:
                supurdu = True
            seviyeler[q] = np.nan          # altına inilen dip artık kullanılmaz
        if supurdu:
            bek_tepe, bek_dip, bek_son = h[i], l[i], i + onay_bar
        # 3) yeni dip kesinleşti mi? (sag mum önceki mum)
        p = i - sag
        dip = all(l[p - k] > l[p] for k in range(1, sol + 1)) and \
            all(l[p + k] >= l[p] for k in range(1, sag + 1))
        if dip:
            for q in range(MAX_SEVIYE):
                if np.isnan(seviyeler[q]):
                    seviyeler[q] = l[p]
                    break
    return sinyal, stop


# ─── Sinyaller ───────────────────────────────────────────────────────────────

def sinyaller(df: pd.DataFrame) -> pd.DataFrame:
    """df: open/high/low/close sütunlu, tarih sıralı günlük veri.

    Dönüş sütunları:
      atr         : ATR(14)
      onay        : onaylı süpürme (filtresiz)
      stop        : o sinyalin stop seviyesi
      filtre      : 12 ay momentum > 0 ve kapanış > SMA200
      al          : onay ve filtre  (ertesi açılışta alınır)
      mom6        : 6 aylık momentum (sepette sıralama için)
    """
    d = df[["open", "high", "low", "close"]].astype(float).dropna()
    h, l, c = (d[k].values for k in ("high", "low", "close"))
    atr = atr_hesap(h, l, c)
    onay, stop = supurme_onay(h, l, c, atr)
    kapanis = pd.Series(c, index=d.index)
    mom12 = kapanis / kapanis.shift(MOM_UZUN) - 1
    sma200 = kapanis.rolling(SMA_UZ).mean()
    filtre = ((mom12 > 0) & (kapanis > sma200)).values
    return pd.DataFrame({
        "atr": atr, "onay": onay, "stop": stop, "filtre": filtre,
        "al": onay & filtre,
        "mom6": (kapanis / kapanis.shift(MOM_KISA) - 1).values,
    }, index=d.index)


# ─── Ortak portföy motoru ────────────────────────────────────────────────────

def _portfoy(O, H, L, C, A, AL, STOP, SIRA, yuva, komisyon, iz_atr, max_bar):
    """Satır = gün, sütun = hisse. AL/STOP/SIRA giriş gününe (sinyalin ertesi günü) yazılı.

    Her gün: önce tutulanlar (iz süren stop güncellenir → stopa değdiyse stopta ya da
    açılış daha aşağıdaysa açılışta satılır; süre dolduysa kapanışta satılır);
    sonra boş yuvalara adaylar 6 ay momentum sırasıyla açılışta alınır.
    Bugün boşalan yuva ancak yarın kullanılır. Giriş günü fiyat stopa inerse aynı gün çıkılır.
    """
    T, M = C.shape
    nakit = 1.0
    tut = {}            # hisse -> dict(deger, ref, giris, stop, tepe, bar)
    getiri = np.zeros(T)
    islemler = []
    for t in range(1, T):
        onceki = nakit + sum(x["deger"] for x in tut.values())
        bosalan = 0
        for j in list(tut):
            if np.isnan(C[t, j]) or np.isnan(O[t, j]):
                continue
            x = tut[j]
            x["bar"] += 1
            if iz_atr > 0 and not np.isnan(A[t - 1, j]):
                x["stop"] = max(x["stop"], x["tepe"] - iz_atr * A[t - 1, j])
            px = -1.0
            if L[t, j] <= x["stop"]:
                px = min(O[t, j], x["stop"])
            elif x["bar"] >= max_bar or t == T - 1:
                px = C[t, j]
            if px > 0:
                nakit += x["deger"] * px * (1 - komisyon) / x["ref"]
                islemler.append(px * (1 - komisyon) / x["giris"] - 1)
                del tut[j]
                bosalan += 1
            else:
                x["deger"] *= C[t, j] / x["ref"]
                x["ref"] = C[t, j]
                x["tepe"] = max(x["tepe"], C[t, j])
        adaylar = np.where(AL[t])[0]
        if len(adaylar):
            puan = np.nan_to_num(SIRA[t, adaylar], nan=-1e18)
            for j in adaylar[np.argsort(-puan, kind="stable")]:
                if j in tut:
                    continue
                if len(tut) + bosalan >= yuva:
                    break
                e, st = O[t, j], STOP[t, j]
                if np.isnan(e) or np.isnan(C[t, j]) or np.isnan(st) or e <= st:
                    continue
                tutar = min(onceki / yuva, nakit)
                if tutar <= 0:
                    break
                g = e * (1 + komisyon)
                nakit -= tutar
                if L[t, j] <= st:            # giriş günü stop
                    px = min(O[t, j], st)
                    nakit += tutar * px * (1 - komisyon) / g
                    islemler.append(px * (1 - komisyon) / g - 1)
                    continue
                tut[j] = dict(deger=tutar * C[t, j] / g, ref=C[t, j], giris=g, stop=st, tepe=C[t, j], bar=0)
        getiri[t] = (nakit + sum(x["deger"] for x in tut.values())) / onceki - 1
    return getiri, islemler, bool(tut)


def _giris_dizileri(sin: pd.DataFrame, konum, shape, j, AL, STOP, SIRA, uygun=None):
    """Sinyal t kapanışında → giriş hissenin KENDİ bir sonraki işlem gününde."""
    a = np.where(sin["onay"].values[:-1])[0]
    STOP[konum[a + 1], j] = sin["stop"].values[a]
    SIRA[konum[a + 1], j] = sin["mom6"].values[a]
    ok = sin["filtre"].values[a]
    if uygun is not None:
        ok = ok & uygun[a]
    AL[konum[a[ok] + 1], j] = True


# ─── Tek hisse backtest ──────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, komisyon: float = 0.0009, iz_atr: float = IZ_ATR,
             max_bar: int = MAX_BAR) -> dict:
    """Tek hisse testi: sinyal kapanışta, işlem ertesi açılışta, tüm para tek hissede.

    komisyon: tek yön oran (0.0009 = %0,09).
    """
    d = df[["open", "high", "low", "close"]].astype(float).dropna()
    s = sinyaller(d)
    O, H, L, C = (d[k].values[:, None] for k in ("open", "high", "low", "close"))
    A = s["atr"].values[:, None]
    AL = np.zeros(C.shape, bool)
    STOP = np.full(C.shape, np.nan)
    SIRA = np.full(C.shape, np.nan)
    _giris_dizileri(s, np.arange(len(d)), C.shape, 0, AL, STOP, SIRA)
    g, islemler, acik = _portfoy(O, H, L, C, A, AL, STOP, SIRA, 1, komisyon, iz_atr, max_bar)
    gunluk = pd.Series(g, index=d.index)
    eg = (1 + gunluk).cumprod()
    c = d["close"].values
    yil = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    gg = gunluk.iloc[1:]
    return {
        "ozsermaye": eg,
        "toplam_getiri": eg.iloc[-1] - 1,
        "yillik_getiri": eg.iloc[-1] ** (1 / yil) - 1,
        "al_tut_yillik": (c[-1] / c[0]) ** (1 / yil) - 1,
        "en_buyuk_dusus": (eg / eg.cummax() - 1).min(),
        "sharpe": gg.mean() / gg.std() * np.sqrt(252) if gg.std() > 0 else 0.0,
        "islem_sayisi": len(islemler),
        "isabet": float(np.mean([x > 0 for x in islemler])) if islemler else float("nan"),
        "acik_pozisyon": acik,      # son gün de kapatılır; bilgi için
    }


# ─── Çok hisseli 10 yuvalı sepet (isteğe bağlı) ──────────────────────────────

def sepet_backtest(veriler: dict, yuva: int = 10, komisyon: float = 0.0009,
                   uygun: dict | None = None) -> pd.Series:
    """veriler: {sembol: DataFrame(open, high, low, close)}. Günlük getiri serisi döner.

    uygun (isteğe bağlı): {sembol: bool Series} — o gün evrende mi (ör. en likit 100).
    Her yeni alım o anki toplam değerin 1/yuva'sı kadardır (nakit yetmezse kalan nakit).
    """
    sem = list(veriler)
    temiz = {k: veriler[k][["open", "high", "low", "close"]].astype(float).dropna() for k in sem}
    sin = {k: sinyaller(v) for k, v in temiz.items()}
    idx = pd.DatetimeIndex(sorted(set().union(*[v.index for v in temiz.values()])))
    O, H, L, C = (pd.DataFrame({k: temiz[k][c] for k in sem}).reindex(idx).values
                  for c in ("open", "high", "low", "close"))
    A = pd.DataFrame({k: sin[k]["atr"] for k in sem}).reindex(idx).values
    AL = np.zeros(C.shape, bool)
    STOP = np.full(C.shape, np.nan)
    SIRA = np.full(C.shape, np.nan)
    for j, k in enumerate(sem):
        u = None
        if uygun is not None:
            u = uygun[k].reindex(sin[k].index).fillna(False).values.astype(bool)
        _giris_dizileri(sin[k], idx.get_indexer(sin[k].index), C.shape, j, AL, STOP, SIRA, u)
    g, _, _ = _portfoy(O, H, L, C, A, AL, STOP, SIRA, yuva, komisyon, IZ_ATR, MAX_BAR)
    return pd.Series(g, index=idx)


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
    print(f"  Son gün           : {'AL (stop ' + format(son['stop'], '.2f') + ')' if son['al'] else 'bekle'}")
    print("Not: tek hisse sonucu videodaki 10 yuvalı sepet ölçümünden farklıdır. Yatırım tavsiyesi değildir.")
