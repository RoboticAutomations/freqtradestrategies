import numpy as np
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import threading
import time
import math
import pygame
import logging

class MUSE(IStrategy):
    """
    Musical EMA Crossover Strategy
    
    Trading Logic:
    - Buy when fast EMA crosses above slow EMA
    - Sell when fast EMA crosses below slow EMA
    
    Musical Logic:
    - Open price -> Base note (pitch)
    - High price -> Note goes up (melody direction)
    - Low price -> Note goes down (melody direction)
    - Close price -> Final note in sequence
    - Volume -> Note duration and intensity
    """
    
    # Strategy parameters
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }
    
    stoploss = -0.10
    timeframe = '5m'
    
    # EMA periods
    fast_ema = 9
    slow_ema = 21
    
    # Musical parameters
    base_note_duration = 0.2  # seconds
    sample_rate = 44100
    
    # Musical note frequencies (in Hz) - C4 to C7 range
    NOTE_FREQUENCIES = {
        'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
        'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
        'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
        'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25,
        'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99,
        'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
        'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51,
        'E6': 1318.51, 'F6': 1396.91, 'F#6': 1479.98, 'G6': 1567.98,
        'G#6': 1661.22, 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53,
        'C7': 2093.00
    }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.note_names = list(self.NOTE_FREQUENCIES.keys())
        self.music_playing = False
        self.music_thread = None
        self.last_played_candle = None
        self.init_pygame()
        
    def init_pygame(self):
        """Initialize pygame mixer for audio"""
        try:
            pygame.mixer.pre_init(frequency=self.sample_rate, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            self.music_playing = True
        except Exception as e:
            logging.error(f"Failed to initialize pygame mixer: {e}")
            self.music_playing = False
    
    def generate_tone(self, frequency, duration, volume=0.5):
        """Generate a sine wave tone"""
        frames = int(duration * self.sample_rate)
        arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
        
        # Add fade in/out to prevent clicks
        fade_frames = int(0.01 * self.sample_rate)  # 10ms fade
        if fade_frames > 0:
            arr[:fade_frames] *= np.linspace(0, 1, fade_frames)
            arr[-fade_frames:] *= np.linspace(1, 0, fade_frames)
        
        # Convert to stereo
        arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
        arr = (arr * volume * 32767).astype(np.int16)
        
        return arr
    
    def play_note(self, frequency, duration, volume=0.5):
        """Play a single note"""
        if not self.music_playing:
            return
            
        try:
            tone = self.generate_tone(frequency, duration, volume)
            sound = pygame.sndarray.make_sound(tone)
            sound.play()
            time.sleep(duration)
        except Exception as e:
            logging.error(f"Error playing note: {e}")
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add technical indicators and musical mappings"""
        
        # EMA indicators
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.fast_ema)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.slow_ema)
        
        # Musical mappings
        dataframe = self.add_musical_indicators(dataframe)
        
        return dataframe
    
    def add_musical_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Add musical interpretations of OHLCV data"""
        
        if len(dataframe) == 0:
            return dataframe
            
        # Price to note mapping (normalize to note range)
        price_range = dataframe['high'].max() - dataframe['low'].min()
        price_min = dataframe['low'].min()
        
        # Map prices to note indices (0-48 for our note range)
        def price_to_note_index(price):
            if price_range > 0:
                normalized = (price - price_min) / price_range
                return int(normalized * (len(self.note_names) - 1))
            return len(self.note_names) // 2
        
        # Open price -> Base note
        dataframe['open_note_idx'] = dataframe['open'].apply(price_to_note_index)
        dataframe['open_note'] = dataframe['open_note_idx'].apply(lambda x: self.note_names[x])
        
        # High price -> Peak note (melody goes up)
        dataframe['high_note_idx'] = dataframe['high'].apply(price_to_note_index)
        dataframe['high_note'] = dataframe['high_note_idx'].apply(lambda x: self.note_names[x])
        
        # Low price -> Valley note (melody goes down)
        dataframe['low_note_idx'] = dataframe['low'].apply(price_to_note_index)
        dataframe['low_note'] = dataframe['low_note_idx'].apply(lambda x: self.note_names[x])
        
        # Close price -> Final note
        dataframe['close_note_idx'] = dataframe['close'].apply(price_to_note_index)
        dataframe['close_note'] = dataframe['close_note_idx'].apply(lambda x: self.note_names[x])
        
        # Volume to duration mapping
        if dataframe['volume'].max() > dataframe['volume'].min():
            volume_max = dataframe['volume'].max()
            volume_min = dataframe['volume'].min()
            
            def volume_to_duration(volume):
                normalized = (volume - volume_min) / (volume_max - volume_min)
                return self.base_note_duration * (0.5 + normalized * 1.5)  # 0.5x to 2x base duration
            
            dataframe['note_duration'] = dataframe['volume'].apply(volume_to_duration)
        else:
            dataframe['note_duration'] = self.base_note_duration
        
        # Price change direction affects melody direction
        dataframe['price_change'] = dataframe['close'] - dataframe['open']
        dataframe['melody_direction'] = np.where(dataframe['price_change'] > 0, 'up', 'down')
        
        # Volatility (high-low range) affects note intensity
        dataframe['volatility'] = dataframe['high'] - dataframe['low']
        
        if dataframe['volatility'].max() > dataframe['volatility'].min():
            vol_max = dataframe['volatility'].max()
            vol_min = dataframe['volatility'].min()
            
            def volatility_to_volume(volatility):
                normalized = (volatility - vol_min) / (vol_max - vol_min)
                return 0.3 + normalized * 0.4  # Volume between 0.3 and 0.7
            
            dataframe['note_volume'] = dataframe['volatility'].apply(volatility_to_volume)
        else:
            dataframe['note_volume'] = 0.5
        
        return dataframe
    
    def play_candle_music(self, candle_data):
        """Play musical representation of a single candle"""
        
        try:
            # Get musical parameters
            open_freq = self.NOTE_FREQUENCIES[candle_data['open_note']]
            high_freq = self.NOTE_FREQUENCIES[candle_data['high_note']]
            low_freq = self.NOTE_FREQUENCIES[candle_data['low_note']]
            close_freq = self.NOTE_FREQUENCIES[candle_data['close_note']]
            
            duration = candle_data['note_duration']
            volume = candle_data['note_volume']
            melody_direction = candle_data['melody_direction']
            
            # Create a melody sequence based on OHLC
            quarter_duration = duration / 4
            
            # Play the candle as a 4-note sequence: Open -> High/Low -> Low/High -> Close
            if melody_direction == 'up':
                # Rising melody: Open -> Low -> High -> Close
                self.play_note(open_freq, quarter_duration, volume)
                self.play_note(low_freq, quarter_duration, volume * 0.8)
                self.play_note(high_freq, quarter_duration, volume * 1.2)
                self.play_note(close_freq, quarter_duration, volume)
            else:
                # Falling melody: Open -> High -> Low -> Close
                self.play_note(open_freq, quarter_duration, volume)
                self.play_note(high_freq, quarter_duration, volume * 1.2)
                self.play_note(low_freq, quarter_duration, volume * 0.8)
                self.play_note(close_freq, quarter_duration, volume)
                
        except Exception as e:
            logging.error(f"Error playing candle music: {e}")
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions and trigger music"""
        
        # EMA crossover buy signal
        dataframe.loc[
            (
                (dataframe['ema_fast'] > dataframe['ema_slow']) &
                (dataframe['ema_fast'].shift(1) <= dataframe['ema_slow'].shift(1)) &
                (dataframe['volume'] < 0)
            ),
            'enter_long'] = 1
        
        # Play music for new candles
        self.play_latest_candle_music(dataframe)
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions"""
        
        # EMA crossover sell signal
        dataframe.loc[
            (
                (dataframe['ema_fast'] < dataframe['ema_slow']) &
                (dataframe['ema_fast'].shift(1) >= dataframe['ema_slow'].shift(1)) &
                (dataframe['volume'] < 0)
            ),
            'exit_long'] = 1
        
        return dataframe
    
    def play_latest_candle_music(self, dataframe: DataFrame):
        """Play music for the latest candle if it's new"""
        
        if len(dataframe) == 0 or not self.music_playing:
            return
            
        latest_candle = dataframe.iloc[-1]
        candle_time = latest_candle.name
        
        # Only play if this is a new candle
        if self.last_played_candle != candle_time:
            self.last_played_candle = candle_time
            
            # Play music in a separate thread to avoid blocking trading
            if self.music_thread is None or not self.music_thread.is_alive():
                self.music_thread = threading.Thread(
                    target=self.play_candle_music,
                    args=(latest_candle,)
                )
                self.music_thread.daemon = True
                self.music_thread.start()
    
    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """Custom exit logic (optional)"""
        return None
    
    def __del__(self):
        """Cleanup when strategy is destroyed"""
        if hasattr(self, 'music_playing') and self.music_playing:
            try:
                pygame.mixer.stop()
            except:
                pass