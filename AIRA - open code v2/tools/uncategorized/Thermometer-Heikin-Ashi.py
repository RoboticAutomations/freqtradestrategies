import numpy as np
import pandas as pd

def ama(src, length, fast_limit, slow_limit):
    flout = 2 / (fast_limit + 1)
    slout = 2 / (slow_limit + 1)
    hh = src.rolling(window=length + 1).max()
    ll = src.rolling(window=length + 1).min()
    mltp = np.where(hh - ll != 0, np.abs(2 * src - ll - hh) / (hh - ll), 0)
    ssc = mltp * (flout - slout) + slout
    ama = np.zeros_like(src)
    for i in range(1, len(src)):
        ama[i] = ama[i-1] + (ssc[i] ** 2) * (src[i] - ama[i-1])
    return ama

def t3(src, length):
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b

    xe1 = src.ewm(span=length).mean()
    xe2 = xe1.ewm(span=length).mean()
    xe3 = xe2.ewm(span=length).mean()
    xe4 = xe3.ewm(span=length).mean()
    xe5 = xe4.ewm(span=length).mean()
    xe6 = xe5.ewm(span=length).mean()

    return c1 * xe6 + c2 * xe5 + c3 * xe4 + c4 * xe3

def kama(src, length, fast_end, slow_end):
    xvnoise = np.abs(src - src.shift(1))
    nsignal = np.abs(src - src.shift(length))
    nnoise = xvnoise.rolling(window=length).sum()
    nefratio = np.where(nnoise != 0, nsignal / nnoise, 0)
    nsmooth = (nefratio * (fast_end - slow_end) + slow_end) ** 2
    kama = np.zeros_like(src)
    for i in range(1, len(src)):
        kama[i] = kama[i-1] + nsmooth[i] * (src[i] - kama[i-1])
    return kama

def trendreg(o, h, l, c):
    return np.where(c > o, (h + c) / 2.0, (l + c) / 2.0)

def trendext(o, h, l, c):
    return np.where(c > o, h, np.where(c < o, l, c))

def habingest(o, h, l, c):
    out = (o + c) / 2 + (((c - o) / (h - l)) * np.abs((c - o) / 2))
    return np.where(np.isnan(out), c, out)

def habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    ingested = habingest(o, h, l, c)
    amaout = ama(ingested, 2, amafl, amasl)
    t3out = t3(ingested, 3)
    kamaout = kama(ingested, 2, kfl, ksl)
    return np.where(smthtype == "AMA", amaout, np.where(smthtype == "T3", t3out, kamaout))

def habopen(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    habopen = np.zeros_like(c)
    for i in range(1, len(c)):
        habopen[i] = (habopen[i-1] + habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)[i-1]) / 2
    return habopen

def habhigh(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return np.maximum.reduce([h, habopen(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)])

def hablow(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return np.minimum.reduce([l, habopen(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)])

def habmedian(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return (habhigh(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + hablow(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)) / 2

def habtypical(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return (habhigh(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + hablow(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)) / 3

def habweighted(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return (habhigh(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + hablow(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + 2 * habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)) / 4

def habaverage(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return (habopen(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + habhigh(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + hablow(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)) / 4

def habavemedbody(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return (habopen(smthtype, o, h, l, c, amafl, amasl, kfl, ksl) + habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl)) / 2

def habtrendb(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return trendreg(habopen(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), habhigh(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), hablow(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl))

def habtrendbext(smthtype, o, h, l, c, amafl, amasl, kfl, ksl):
    return trendext(habopen(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), habhigh(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), hablow(smthtype, o, h, l, c, amafl, amasl, kfl, ksl), habclose(smthtype, o, h, l, c, amafl, amasl, kfl, ksl))

# Example usage
# Replace this part with your actual data
data = {
    'open': pd.Series([1, 2, 3, 4, 5]),
    'high': pd.Series([2, 3, 4, 5, 6]),
    'low': pd.Series([1, 1.5, 2.5, 3.5, 4.5]),
    'close': pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
}

o = data['open']
h = data['high']
l = data['low']
c = data['close']

ama_fast_limit = 6
ama_slow_limit = 16
kama_fast_end = 2
kama_slow_end = 30

# Choose smoothing type: "AMA", "T3", or "KAMA"
smoothing_type = "AMA"

habopen_values = habopen(smoothing_type, o, h, l, c, ama_fast_limit, ama_slow_limit, kama_fast_end, kama_slow_end)
print(habopen_values)
