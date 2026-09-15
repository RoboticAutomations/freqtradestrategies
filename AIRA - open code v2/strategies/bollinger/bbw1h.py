import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from datetime import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

class WhaleWallHunter(IStrategy):
    """
    WhaleWallHunter V2 (Revisi Safety)
    
    Logika Perbaikan:
    1. TREND FILTER: Hanya Long di atas EMA 200, Hanya Short di bawah EMA 200.
    2. HARD STOP: Stoploss fix -5% (tanpa dynamic SL yang berisiko).
    3. TIME LIMIT: Keluar otomatis jika 2 jam tidak profit (ROI).
    4. VOLATILITY FILTER: Tidak masuk saat BB Width > 10% (Market Menggila).
    """

    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # Izinkan Shorting
    can_short = True 

    # 1. STOPLOSS FIX (SAFETY NET)
    # Kita kunci di 5%. Jika market crash, langsung keluar.
    stoploss = -0.03 
    
    # 2. ROI (TIME LIMIT)
    # - Menit 0-60: Target profit tinggi (via custom_exit)
    # - Menit 60: Profit 1% ambil saja.
    # - Menit 120 (2 Jam): Jika masih impas (0%), keluar paksa (Dead Trade).
    minimal_roi = {
        "0": 1.0,   
        "60": 0.01, 
        "120": 0    
    }

    # Buffer Jarak Aman TP (0.5% sebelum dinding)
    EXIT_BUFFER = 0.005
    
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ==========================================
        # 1. INDIKATOR 1H (INFORMATIVE)
        # ==========================================
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        
        # Bollinger Bands 1H
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_upper'] = bollinger['upper']
        informative['bb_lower'] = bollinger['lower']
        informative['bb_mid']   = bollinger['mid']
        
        # Hitung Lebar BB & Zona 30%
        bb_width_val = informative['bb_upper'] - informative['bb_lower']
        informative['zone_short_limit'] = informative['bb_upper'] - (bb_width_val * 0.30)
        informative['zone_long_limit'] = informative['bb_lower'] + (bb_width_val * 0.30)

# ... (Kode Bollinger Bands 1H sebelumnya biarkan saja) ...

        # ==========================================
        # B. ADX 4h (Resampling dari Data 1H)
        # ==========================================
        
        # 1. Buat salinan data 1h dan set index ke tanggal (wajib untuk resampling)
        resampled_data = informative.set_index('date')
        
        # 2. Ubah (Resample) data 1h menjadi candle 4h
        # Kita perlu Open, High, Low, Close yang valid untuk perhitungan ADX
        inform_4h = resampled_data.resample('4h').agg({
            'open': 'first',
            'high': 'max',  # High 4h adalah High tertinggi dari 6 candle 1h
            'low': 'min',   # Low 4h adalah Low terendah dari 6 candle 1h
            'close': 'last',
            'volume': 'sum'
        })
        
        # 3. Hitung ADX menggunakan data 4h yang baru dibuat
        inform_4h['adx_4h'] = ta.ADX(inform_4h, timeperiod=14)
        
        # 4. Bersihkan kolom tidak perlu (sisakan hanya adx_4h) agar ringan saat merge
        inform_4h = inform_4h[['adx_4h']]
        
        # 5. Gabungkan kembali ke data informative asli (1h)
        # Kita merge berdasarkan index waktu (date)
        informative = informative.merge(inform_4h, left_on='date', right_index=True, how='left')
        
        # 6. Isi nilai kosong (Forward Fill)
        # Karena data 4h cuma muncul setiap 6 baris sekali di data 1h, baris kosongnya harus diisi
        # dengan nilai terakhir yang tersedia.
        informative['adx_4h'] = informative['adx_4h'].ffill()
        
        # 7. LOGIKA ADX FALLING (Menggunakan data 4h)
        # Shift(1) di sini artinya kita membandingkan dengan ADX 4h periode sebelumnya
        # (Karena sudah di-ffill, shift(1) di candle 1h akan mengecek perubahan nilai 4h terakhir)
        informative['adx_falling'] = np.where(informative['adx_4h'] < informative['adx_4h'].shift(1), 1, 0)

        # ... (Lanjut ke merge_informative_pair seperti biasa) ...
        # Merge ke 5m
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # ==========================================
        # 2. INDIKATOR 5M (MAIN)
        # ==========================================
        
        # MFI 5m (Trigger)
        dataframe['mfi'] = ta.MFI(dataframe)
        
        # EMA 200 (Trend Filter Wajib)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # BB Width 5m (Volatility Filter)
        # Kita pakai data 1h yang sudah di-merge ('bb_upper_1h') untuk cek volatilitas global
        dataframe['bb_width_1h'] = (dataframe['bb_upper_1h'] - dataframe['bb_lower_1h']) / dataframe['bb_mid_1h']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # V3: LOGIKA LEBIH LONGGAR (Supaya ada trade)
        
        dataframe.loc[
            (
                # 1. Trend & Volatility (DILONGGARKAN)
                (dataframe['adx_falling_1h'] == 1) &       # Market tenang/Sideways (Tetap dipakai)
                (dataframe['bb_width_1h'] < 0.20) &        # Diubah: Izinkan volatilitas sampai 20%
                
                # HAPUS EMA 200: Biarkan bot trading di kondisi sideways murni
                # (dataframe['close'] > dataframe['ema_200']) & 

                # 2. Location & Trigger (DILONGGARKAN)
                (dataframe['close'] < dataframe['zone_long_limit_1h']) & # Harga di Zona Bawah 30%
                (dataframe['mfi'] < 25)                                  # Diubah: MFI < 25 (lebih mudah masuk)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # 1. Trend & Volatility
                (dataframe['adx_falling_1h'] == 1) &
                (dataframe['bb_width_1h'] < 0.20) &
                
                # HAPUS EMA 200
                # (dataframe['close'] < dataframe['ema_200']) & 
                
                # 2. Location & Trigger
                (dataframe['close'] > dataframe['zone_short_limit_1h']) & # Harga di Zona Atas 30%
                (dataframe['mfi'] > 75)                                   # Diubah: MFI > 75 (lebih mudah masuk)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # SAYA HAPUS custom_stoploss AGAR MENGGUNAKAN HARD STOP -5% DI ATAS
    # Ini untuk mencegah SL 'lari' saat crash.

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        
        tag = "front_run_wall"

        # ---------------------------------------
        # TP POSISI LONG (Target: BB Upper 1H)
        # ---------------------------------------
        if trade.is_short is False:
            # Target Dinamis: Keluar sedikit di bawah Resistance
            target_price = last_candle['bb_upper_1h'] * (1 - self.EXIT_BUFFER)

            if current_rate >= target_price:
                return tag

        # ---------------------------------------
        # TP POSISI SHORT (Target: BB Lower 1H)
        # ---------------------------------------
        else:
            # Target Dinamis: Keluar sedikit di atas Support
            target_price = last_candle['bb_lower_1h'] * (1 + self.EXIT_BUFFER)

            if current_rate <= target_price:
                return tag

        return None