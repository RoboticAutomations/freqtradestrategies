
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from pandas import DataFrame
import warnings
import pandas as pd
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
import math
import logging
from scipy.signal import find_peaks, find_peaks_cwt
import warnings
import talib.abstract as ta
from math import ceil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
import time

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


logger = logging.getLogger(__name__)

class CustomPeakFinder:
    def _cwt(self, data, wavelet, widths, dtype=None, **kwargs):
        # Determine output type
        if dtype is None:
            if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
                dtype = np.complex128
            else:
                dtype = np.float64
    
        output = np.empty((len(widths), len(data)), dtype=dtype)
        for ind, width in enumerate(widths):
            N = np.min([10 * width, len(data)])
            wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
            #output[ind] = convolve(data, wavelet_data, mode='same')
            # Custom convolution-like operation without look-ahead
            output[ind] = np.correlate(data, wavelet_data, mode='same')
        return output

    def _ricker(self, points, a):
        A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
        wsq = a**2
        vec = np.arange(0, points) - (points - 1.0) / 2
        xsq = vec**2
        mod = (1 - xsq / wsq)
        gauss = np.exp(-xsq / (2 * wsq))
        total = A * mod * gauss
        return total
    
    def _identify_ridge_lines(self, matr, max_distances, gap_thresh):
        if len(max_distances) < matr.shape[0]:
            raise ValueError('Max_distances must have at least as many rows '
                             'as matr')
    
        all_max_cols = self._boolrelextrema(matr, np.greater, axis=1, order=1)
        # Highest row for which there are any relative maxima
        has_relmax = np.nonzero(all_max_cols.any(axis=1))[0]
        if len(has_relmax) == 0:
            return []
        start_row = has_relmax[-1]
        # Each ridge line is a 3-tuple:
        # rows, cols,Gap number
        ridge_lines = [[[start_row],
                       [col],
                       0] for col in np.nonzero(all_max_cols[start_row])[0]]
        final_lines = []
        rows = np.arange(start_row - 1, -1, -1)
        cols = np.arange(0, matr.shape[1])
        for row in rows:
            this_max_cols = cols[all_max_cols[row]]
    
            # Increment gap number of each line,
            # set it to zero later if appropriate
            for line in ridge_lines:
                line[2] += 1
    
            # XXX These should always be all_max_cols[row]
            # But the order might be different. Might be an efficiency gain
            # to make sure the order is the same and avoid this iteration
            prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
            # Look through every relative maximum found at current row
            # Attempt to connect them with existing ridge lines.
            for ind, col in enumerate(this_max_cols):
                # If there is a previous ridge line within
                # the max_distance to connect to, do so.
                # Otherwise start a new one.
                line = None
                if len(prev_ridge_cols) > 0:
                    diffs = np.abs(col - prev_ridge_cols)
                    closest = np.argmin(diffs)
                    if diffs[closest] <= max_distances[row]:
                        line = ridge_lines[closest]
                if line is not None:
                    # Found a point close enough, extend current ridge line
                    line[1].append(col)
                    line[0].append(row)
                    line[2] = 0
                else:
                    new_line = [[row],
                                [col],
                                0]
                    ridge_lines.append(new_line)
    
            # Remove the ridge lines with gap_number too high
            # XXX Modifying a list while iterating over it.
            # Should be safe, since we iterate backwards, but
            # still tacky.
            for ind in range(len(ridge_lines) - 1, -1, -1):
                line = ridge_lines[ind]
                if line[2] > gap_thresh:
                    final_lines.append(line)
                    del ridge_lines[ind]
    
        out_lines = []
        for line in (final_lines + ridge_lines):
            sortargs = np.array(np.argsort(line[0]))
            rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
            rows[sortargs] = line[0]
            cols[sortargs] = line[1]
            out_lines.append([rows, cols])
    
        return out_lines

    def _filter_ridge_lines(self, cwt, ridge_lines, window_size=None, min_length=None,
                            min_snr=1, noise_perc=10):
        num_points = cwt.shape[1]
        if min_length is None:
            min_length = np.ceil(cwt.shape[0] / 4)
        if window_size is None:
            window_size = np.ceil(num_points / 20)

        window_size = int(window_size)
        hf_window, odd = divmod(window_size, 2)

        row_one = cwt[0, :]
        noises = np.empty_like(row_one)
        for ind, val in enumerate(row_one):
            window_start = max(ind - hf_window, 0)
            window_end = min(ind + hf_window + odd, num_points)
            noises[ind] = np.percentile(row_one[window_start:window_end], noise_perc)

        def filt_func(line):
            if len(line[0]) < min_length:
                return False
            snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])
            if snr < min_snr:
                return False
            return True

        return list(filter(filt_func, ridge_lines))

    def find_peaks_cwt(self, vector, widths, wavelet=None, max_distances=None,
                       gap_thresh=None, min_length=None,
                       min_snr=1, noise_perc=10, window_size=None):
        widths = np.array(widths, copy=False, ndmin=1)
    
        if gap_thresh is None:
            gap_thresh = np.ceil(widths[0])
        if max_distances is None:
            max_distances = widths / 4.0
        if wavelet is None:
            wavelet = self._ricker
    
        cwt_dat = self._cwt(vector, wavelet, widths)
        ridge_lines = self._identify_ridge_lines(cwt_dat, max_distances, gap_thresh)
        filtered = self._filter_ridge_lines(cwt_dat, ridge_lines, min_length=min_length,
                                       window_size=window_size, min_snr=min_snr,
                                       noise_perc=noise_perc)
        max_locs = np.asarray([x[1][0] for x in filtered])
        max_locs.sort()
    
        return max_locs
    
    def _boolrelextrema(self, data, comparator, axis=0, order=1, mode='clip'):
        if (int(order) != order) or (order < 1):
            raise ValueError('Order must be an int >= 1')
    
        datalen = data.shape[axis]
        locs = np.arange(0, datalen)
    
        results = np.ones(data.shape, dtype=bool)
        main = data.take(locs, axis=axis, mode=mode)
        
        for shift in range(1, order + 1):
            plus = data.take(locs + shift, axis=axis, mode=mode)
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            
            if not results.any():
                return results
        
        return results

    def _arg_x_as_expected(self, value):
        value = np.asarray(value, order='C', dtype=np.float64)
        if value.ndim != 1:
            raise ValueError('`x` must be a 1-D array')
        return value

    def _arg_peaks_as_expected(self, value):
        value = np.asarray(value)
        if value.size == 0:
            # Empty arrays default to np.float64 but are valid input
            value = np.array([], dtype=np.intp)
        try:
            # Safely convert to C-contiguous array of type np.intp
            value = value.astype(np.intp, order='C', casting='safe', subok=False, copy=False)
        except TypeError as e:
            raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
        if value.ndim != 1:
            raise ValueError('`peaks` must be a 1-D array')
        return value

    def _arg_wlen_as_expected(self, value):
        if value is None:
            # _peak_prominences expects an intp; -1 signals that no value was
            # supplied by the user
            value = -1
        elif 1 < value:
            # Round up to a positive integer
            if isinstance(value, float):
                value = math.ceil(value)
            value = np.intp(value)
        else:
            raise ValueError(f'`wlen` must be larger than 1, was {value}')
        return value

    def _local_maxima_1d(self, x):
        midpoints = []
        left_edges = []
        right_edges = []
        m = 0  # Pointer to the end of the valid area in allocated arrays
    
        i = 1  # Pointer to the current sample, the first one can't be maxima
        i_max = len(x) - 1  # Last sample can't be maxima
    
        while i < i_max:
            # Test if the previous sample is smaller
            if x[i - 1] < x[i]:
                i_ahead = i + 1  # Index to look ahead of the current sample
    
                # Find the next sample that is unequal to x[i]
                while i_ahead < i_max and x[i_ahead] == x[i]:
                    i_ahead += 1
    
                # Maxima is found if the next unequal sample is smaller than x[i]
                if x[i_ahead] < x[i]:
                    left_edges.append(i)
                    right_edges.append(i_ahead - 1)
                    midpoints.append((left_edges[m] + right_edges[m]) // 2)
                    m += 1
                    # Skip samples that can't be a maximum
                    i = i_ahead
            i += 1
    
        return midpoints, left_edges, right_edges

    def _local_minima_1d(self, x):
        midpoints = []
        left_edges = []
        right_edges = []
        m = 0  # Pointer to the end of the valid area in allocated arrays
    
        i = 1  # Pointer to the current sample, the first one can't be minima
        i_max = len(x) - 1  # Last sample can't be minima
    
        while i < i_max:
            # Test if the previous sample is larger
            if x[i - 1] > x[i]:
                i_ahead = i + 1  # Index to look ahead of the current sample
    
                # Find the next sample that is unequal to x[i]
                while i_ahead < i_max and x[i_ahead] == x[i]:
                    i_ahead += 1
    
                # Minima is found if the next unequal sample is larger than x[i]
                if x[i_ahead] > x[i]:
                    left_edges.append(i)
                    right_edges.append(i_ahead - 1)
                    midpoints.append((left_edges[m] + right_edges[m]) // 2)
                    m += 1
                    # Skip samples that can't be a minimum
                    i = i_ahead
            i += 1
    
        return midpoints, left_edges, right_edges

    def select_by_peak_distance(self, peaks, priority, distance):
        peaks_size = len(peaks)
        # Round up because the actual peak distance can only be a natural number
        distance_ = int(ceil(distance))
        keep = np.ones(peaks_size, dtype=np.uint8)  # Prepare an array of flags
    
        # Create a map from `i` (index for `peaks` sorted by `priority`) to `j` (index
        # for `peaks` sorted by position). This allows iterating `peaks` and `keep`
        # with `j` by order of `priority` while still maintaining the ability to
        # step to neighboring peaks with (`j` + 1) or (`j` - 1).
        priority_to_position = np.argsort(priority)
    
        # Highest priority first -> iterate in reverse order (decreasing)
        for i in range(peaks_size - 1, -1, -1):
            # "Translate" `i` to `j` which points to the current peak whose
            # neighbors are to be evaluated
            j = priority_to_position[i]
            if keep[j] == 0:
                # Skip evaluation for a peak already marked as "don't keep"
                continue
    
            k = j - 1
            # Flag "earlier" peaks for removal until the minimal distance is exceeded
            while 0 <= k and peaks[j] - peaks[k] < distance_:
                keep[k] = 0
                k -= 1
    
            k = j + 1
            # Flag "later" peaks for removal until the minimal distance is exceeded
            while k < peaks_size and peaks[k] - peaks[j] < distance_:
                keep[k] = 0
                k += 1
    
        return keep.astype(bool)  # Return as a boolean array

    def _arg_wlen_as_expected(self, value):
        if value is None:
            # _peak_prominences expects an intp; -1 signals that no value was
            # supplied by the user
            value = -1
        elif 1 < value:
            # Round up to a positive integer
            if isinstance(value, float):
                value = math.ceil(value)
            value = np.intp(value)
        else:
            raise ValueError(f'`wlen` must be larger than 1, was {value}')
        return value

    def _arg_x_as_expected(self, value):
        value = np.asarray(value, order='C', dtype=np.float64)
        if value.ndim != 1:
            raise ValueError('`x` must be a 1-D array')
        return value    

    def _peak_prominences(self, x, peaks, wlen):
        show_warning = False
        prominences = np.empty(len(peaks), dtype=np.float64)
        left_bases = np.empty(len(peaks), dtype=np.intp)
        right_bases = np.empty(len(peaks), dtype=np.intp)
    
        for peak_nr in range(len(peaks)):
            peak = peaks[peak_nr]
            i_min = 0
            i_max = len(x) - 1
            if not i_min <= peak <= i_max:
                raise ValueError("peak {} is not a valid index for `x`".format(peak))
    
            if wlen >= 2:
                # Adjust the window around the evaluated peak (within bounds);
                # if wlen is even, the resulting window length is implicitly
                # rounded to the next odd integer
                i_min = max(peak - wlen // 2, i_min)
                i_max = min(peak + wlen // 2, i_max)
    
            # Find the left base in the interval [i_min, peak]
            i = left_bases[peak_nr] = peak
            left_min = x[peak]
            while i_min <= i and x[i] <= x[peak]:
                if x[i] < left_min:
                    left_min = x[i]
                    left_bases[peak_nr] = i
                i -= 1
    
            # Find the right base in the interval [peak, i_max]
            i = right_bases[peak_nr] = peak
            right_min = x[peak]
            while i <= i_max and x[i] <= x[peak]:
                if x[i] < right_min:
                    right_min = x[i]
                    right_bases[peak_nr] = i
                i += 1
    
            prominences[peak_nr] = x[peak] - max(left_min, right_min)
            if prominences[peak_nr] == 0:
                show_warning = True
    
        if show_warning:
            logger.warning("some peaks have a prominence of 0")
            
        # Return as ndarrays
        return prominences, left_bases, right_bases

    def _unpack_condition_args(self, interval, x, peaks):
        try:
            imin, imax = interval
        except (TypeError, ValueError):
            imin, imax = (interval, None)
    
        # Reduce arrays if arrays
        if isinstance(imin, np.ndarray):
            if imin.size != x.size:
                raise ValueError('array size of lower interval border must match x')
            imin = imin[peaks]
        if isinstance(imax, np.ndarray):
            if imax.size != x.size:
                raise ValueError('array size of upper interval border must match x')
            imax = imax[peaks]
    
        return imin, imax

    def peak_prominences(self, x, peaks, wlen=None):
        x = self._arg_x_as_expected(x)
        peaks = self._arg_peaks_as_expected(peaks)
        wlen = self._arg_wlen_as_expected(wlen)
        return _peak_prominences(x, peaks, wlen)

    def peak_widths(self, x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
        x = _arg_x_as_expected(x)
        peaks = _arg_peaks_as_expected(peaks)
        if prominence_data is None:
            wlen = _arg_wlen_as_expected(wlen)
            prominence_data = _peak_prominences(x, peaks, wlen=wlen)
        return _peak_widths(x, peaks, rel_height, *prominence_data)

    def _select_by_property(self, peak_properties, pmin, pmax):
        keep = np.ones(peak_properties.size, dtype=bool)
        if pmin is not None:
            keep &= (pmin <= peak_properties)
        if pmax is not None:
            keep &= (peak_properties <= pmax)
        return keep

    def find_peaks(self, x, height=None, threshold=None, distance=None,
                   prominence=None, width=None, wlen=None, rel_height=0.5,
                   plateau_size=None):
    
        # _argmaxima1d expects array of dtype 'float64'
        x = self._arg_x_as_expected(x)
        if distance is not None and distance < 1:
            raise ValueError('`distance` must be greater or equal to 1')
    
        peaks, left_edges, right_edges = self._local_maxima_1d(x)
        properties = {}
    
        if plateau_size is not None:
            # Evaluate plateau size
            plateau_sizes = right_edges - left_edges + 1
            pmin, pmax = self._unpack_condition_args(plateau_size, x, peaks)
            keep = self._select_by_property(plateau_sizes, pmin, pmax)
            peaks = np.array(peaks)[keep]  # Convert to NumPy array before boolean indexing
            properties["plateau_sizes"] = plateau_sizes
            properties["left_edges"] = left_edges
            properties["right_edges"] = right_edges
            properties = {key: array[keep] for key, array in properties.items()}
    
        if height is not None:
            # Evaluate height condition
            peak_heights = x[peaks]
            hmin, hmax = self._unpack_condition_args(height, x, peaks)
            keep = self._select_by_property(peak_heights, hmin, hmax)
            peaks = np.array(peaks)[keep]  # Convert to NumPy array before boolean indexing
            properties["peak_heights"] = peak_heights
            properties = {key: array[keep] for key, array in properties.items()}
    
        if threshold is not None:
            # Evaluate threshold condition
            tmin, tmax = self._unpack_condition_args(threshold, x, peaks)
            keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
                x, peaks, tmin, tmax)
            peaks = np.array(peaks)[keep]  # Convert to NumPy array before boolean indexing
            properties["left_thresholds"] = left_thresholds
            properties["right_thresholds"] = right_thresholds
            properties = {key: array[keep] for key, array in properties.items()}
    
        if distance is not None:
            # Evaluate distance condition
            keep = self._select_by_peak_distance(peaks, x[peaks], distance)
            peaks = np.array(peaks)[keep]  # Convert to NumPy array before boolean indexing
            properties = {key: array[keep] for key, array in properties.items()}
    
        if prominence is not None or width is not None:
            # Calculate prominence (required for both conditions)
            wlen = self._arg_wlen_as_expected(wlen)
            properties.update(zip(
                ['prominences', 'left_bases', 'right_bases'],
                self._peak_prominences(x, peaks, wlen=wlen)
            ))
    
        if prominence is not None:
            # Evaluate prominence condition
            pmin, pmax = self._unpack_condition_args(prominence, x, peaks)
            keep = self._select_by_property(properties['prominences'], pmin, pmax)
            peaks = np.array(peaks)[keep]  # Convert to NumPy array before boolean indexing
            properties = {key: array[keep] for key, array in properties.items()}
    
        if width is not None:
            # Calculate widths
            properties.update(zip(
                ['widths', 'width_heights', 'left_ips', 'right_ips'],
                _peak_widths(x, peaks, rel_height, properties['prominences'],
                             properties['left_bases'], properties['right_bases'])
            ))
            # Evaluate width condition
            wmin, wmax = self._unpack_condition_args(width, x, peaks)
            keep = self._select_by_property(properties['widths'], wmin, wmax)
            peaks = np.array(peaks)[keep]  # Convert to NumPy array before boolean indexing
            properties = {key: array[keep] for key, array in properties.items()}
    
        return peaks, properties

class ARIMA(IStrategy):
    INTERFACE_VERSION = 2

#     # ROI table:
#     minimal_roi = {
#         "0": 0.195,
#         "39": 0.10600000000000001,
#         "91": 0.04,
#         "210": 0
#     }

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    use_custom_stoploss = True


    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '15m'
    startup_candle_count = 400
    process_only_new_candles = True
    
    # DCA
    position_adjustment_enable = False

    # Custom Entry
    last_entry_price = None

    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(150, 200, default=184, space='buy', optimize=True)
    up = DecimalParameter(low=1.020, high=1.025, default=1.02, decimals=3 ,space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3 ,space='buy', optimize=True, load=True)
    enable1 = BooleanParameter(default=True, space="buy", optimize=False)
    enable2 = BooleanParameter(default=True, space="buy", optimize=False)
    enable3 = BooleanParameter(default=True, space="buy", optimize=False)
    enable4 = BooleanParameter(default=True, space="buy", optimize=False)
    # enable5 = BooleanParameter(default=True, space="buy", optimize=False)
    # enable6 = BooleanParameter(default=True, space="buy", optimize=False)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)
    rekt = DecimalParameter(low=1.01, high=1.10, default=1.05, decimals=2 ,space='buy', optimize=True, load=True)
    atr_length = IntParameter(10, 30, default=14, space='buy', optimize=True, load=True)

    # Modulus    
    # peaks = IntParameter(70, 120, default=120, space='buy', optimize=True) ### initial smallest window
    # bull_bear = IntParameter(120, 160, default=155, space='buy', optimize=True)
    # trend = DecimalParameter(low=25, high=40, default=29.6, decimals=1 ,space='buy', optimize=True, load=True)
    # volatility = DecimalParameter(low=30, high=50, default=38.4, decimals=1 ,space='buy', optimize=True, load=True)
    # sensitivity = IntParameter(7, 15, default=11, space='buy', optimize=True, load=True)
    # lookback_candles = IntParameter(30, 60, default=55, space='buy', optimize=True, load=True)
    # atr = IntParameter(3, 7, default=5, space='buy', optimize=True, load=True)
    perc_target_buy = DecimalParameter(low=1, high=5, default=1, decimals=1 ,space='buy', optimize=True, load=True)
    perc_target_sell = DecimalParameter(low=1, high=5, default=1, decimals=1 ,space='sell', optimize=True, load=True)

    # DCA
    # initial_safety_order_trigger = DecimalParameter(low=-0.02, high=-0.015, default=-0.016, decimals=3 ,space='buy', optimize=True, load=True)
    # max_safety_orders = IntParameter(1, 6, default=2, space='buy', optimize=True)
    # max_dca_multiplier = IntParameter(1, 6, default=2, space='buy', optimize=True)
    # safety_order_step_scale = DecimalParameter(low=1.1, high=1.4, default=1.3, decimals=2 ,space='buy', optimize=True, load=True)
    # safety_order_volume_scale = DecimalParameter(low=1.1, high=1.5, default=1.2, decimals=1 ,space='buy', optimize=True, load=True)

    # Unclog Function
    days = IntParameter(2, 7, default=4, space='sell', optimize=True)
    loss = DecimalParameter(-0.07, -0.04, default=-0.055, space='sell', optimize=True)

    ### trailing stop loss optimiziation ###
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.03, high=0.06, default=0.04, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.012, default=0.01, decimals=3, space='sell', optimize=True, load=True)
    moon = IntParameter(80, 90, default=85, space='sell', optimize=True)


    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['rsi'] < self.moon.value:

            for stop3 in self.tsl_target3.range:
                if (current_profit > stop3):
                    for stop3a in self.ts3.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                        return stop3a 
            for stop2 in self.tsl_target2.range:
                if (current_profit > stop2):
                    for stop2a in self.ts2.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
                        return stop2a 
            for stop1 in self.tsl_target1.range:
                if (current_profit > stop1):
                    for stop1a in self.ts1.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
                        return stop1a 
            for stop0 in self.tsl_target0.range:
                if (current_profit > stop0):
                    for stop0a in self.ts0.range:
                        self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
                        return stop0a 
        else:
            for stop0 in self.tsl_target0.range:
                if (current_profit > stop0):
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                    return 0.99

        return self.stoploss

    # def adjust_trade_position(self, trade: Trade, current_time: datetime,
    #                           current_rate: float, current_profit: float, min_stake: float,
    #                           max_stake: float, **kwargs):
    #     if current_profit > self.initial_safety_order_trigger.value:
    #         logger.info(f"{trade.pair} - Current Profit: {current_profit} Trigger: {self.initial_safety_order_trigger.value}")
    #         return None

    #     dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

    #     count_of_buys = 0
    #     for order in trade.orders:
    #         if order.ft_is_open or order.ft_order_side != 'buy':
    #             continue
    #         if order.status == "closed":
    #             count_of_buys += 1

    #     if 1 <= count_of_buys <= self.max_safety_orders.value:
            
    #         safety_order_trigger = abs(self.initial_safety_order_trigger.value) + (abs(self.initial_safety_order_trigger.value) * self.safety_order_step_scale.value * (math.pow(self.safety_order_step_scale.value,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale.value - 1))

    #         if current_profit <= (-1 * abs(safety_order_trigger)):
    #             try:
    #                 stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
    #                 stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value,(count_of_buys - 1))
    #                 amount = stake_amount / current_rate
    #                 logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
    #                 return stake_amount
    #             except Exception as exception:
    #                 logger.debug(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
    #                 return None
    #         else:
    #             stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
    #             stake_amount = stake_amount * math.pow(self.safety_order_volume_scale.value,(count_of_buys - 1))
    #             logger.info(f"{trade.pair} Next Safety Order #{count_of_buys} @ Trigger -{safety_order_trigger} Current Profit: {current_profit}")    
    #             return None

    #     logger.info(f"{trade.pair} - Current Profit: {current_profit} All safety orders used")     
    #     return None

    # ### Custom Functions ###
    # # This is called when placing the initial order (opening trade)
    # # Let unlimited stakes leave funds open for DCA orders
    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                         proposed_stake: float, min_stake: Optional[float], max_stake: float,
    #                         leverage: float, entry_tag: Optional[str], side: str,
    #                         **kwargs) -> float:

    #     # We need to leave most of the funds for possible further DCA orders
    #     # This also applies to fixed stakes
    #     total_stake = proposed_stake / (self.max_dca_multiplier.value + self.max_safety_orders.value)
    #     return total_stake

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < self.loss.value and (current_time - trade.open_date_utc).days >= self.days.value:
            return 'unclog'

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}") 

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        # Handle freak events
        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} ROI is below 0")
            self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        size = len(dataframe) - 10
        train, test = model_selection.train_test_split(dataframe['OHLC4'], train_size=size)

        # Fit ARIMA model
        start_time = time.time()
        arima_model = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                     max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                     stepwise=True, suppress_warnings=True, D=10, max_D=20,
                     error_action='ignore')

        fitting_time = time.time() - start_time

        # Forecast future values with confidence intervals
        start_time = time.time()
        future_forecast, conf_int = arima_model.predict(n_periods=test.shape[0], return_conf_int=True)
        inference_time = time.time() - start_time
        # logger.info(f"{pair} Fitting time: {fitting_time:.2f} seconds")
        # logger.info(f"{pair} Inference time: {inference_time:.2f} seconds")

        # Extract upper and lower confidence intervals
        lower_confidence, upper_confidence = conf_int[:, 0], conf_int[:, 1]
        logger.info(f"{pair} - Fitting time: {fitting_time:.2f} seconds | Inference time: {inference_time:.2f} seconds | " \
             f"Current Price: {dataframe['OHLC4'].iloc[-1]:.7f} | {self.timeframe} Future Forecast: {future_forecast.iloc[-1]:.7f}")
        rmse = np.sqrt(mean_squared_error(test, future_forecast))
        accuracy_perc = 100 * (1 - (rmse / dataframe['OHLC4'].iloc[-1]))
        reward = ((future_forecast.iloc[-1] / dataframe['OHLC4'].iloc[-1]) - 1) * 100

        dataframe['accuracy'] = accuracy_perc
        dataframe['reward'] = reward
        dataframe['rmse'] = rmse

        if future_forecast.iloc[-1] > dataframe['OHLC4'].iloc[-1]:
            direction = 'Up'
            dataframe['decision'] = 1
        else:
            direction = 'Down'
            dataframe['decision'] = -1


        logger.info(f"{pair} - Test RMSE: {rmse:.3f} | Accuracy: {accuracy_perc:.2f}% | Potential Profit: {reward:.2}% | Trend: {direction}")
        # print(type(future_forecast), )
        # # Create a DataFrame with the forecasted values and confidence intervals
        # forecast_df = pd.DataFrame({
        #     'arima_predictions': future_forecast,
        #     'lower_confidence': lower_confidence,
        #     'upper_confidence': upper_confidence
        # })


        dataframe['arima_predictions'] = pd.Series(future_forecast)
        dataframe['lower_confidence'] = pd.Series(lower_confidence)
        dataframe['upper_confidence'] = pd.Series(upper_confidence)

        # print(dataframe['arima_predictions'].iloc[-1])

        peak_finder = CustomPeakFinder()

        """
        widths = np.arange(1, 10)
        peaks = peak_finder.find_peaks_cwt(dataframe['close'], widths=widths)
        
        # Identify troughs using the modified local_minima_1d method
        troughs, _, _ = peak_finder._local_minima_1d(dataframe['close'])
        
        # Create a new column for the signal, initially set to 0
        dataframe['peak'] = 0

        # Generate sell signals when the index is a peak
        dataframe.loc[dataframe.index.isin(peaks), 'peak'] = 1

        # Generate buy signals when the index is a trough
        dataframe.loc[dataframe.index.isin(troughs), 'peak'] = -1
        """
        
        ptc_target_buy = self.perc_target_buy.value
        ptc_target_sell = self.perc_target_sell.value

        high_peaks, high_properties = peak_finder.find_peaks(dataframe['OHLC4'].values, height=None, threshold=None,
                                                 distance=None, prominence=dataframe['OHLC4'].values / 100 * ptc_target_sell,
                                                 wlen=100, plateau_size=None)

        lower_peaks, low_properties = peak_finder.find_peaks(-dataframe['OHLC4'].values, height=None, threshold=None,
                                                 distance=None, prominence=dataframe['OHLC4'].values / 100 * ptc_target_buy,
                                                 wlen=100, plateau_size=None)

        # dataframe['peak_prominance'] = float(0)
        dataframe['peak'] = float(0)

        # dataframe.iloc[lower_peaks, dataframe.columns.get_loc('peak_prominance')] = low_properties["prominences"].astype('float')
        # dataframe.iloc[high_peaks, dataframe.columns.get_loc('peak_prominance')] = -high_properties["prominences"].astype('float')
        
        dataframe.iloc[lower_peaks, dataframe.columns.get_loc('peak')] = -1
        dataframe.iloc[high_peaks, dataframe.columns.get_loc('peak')] = 1
        
        # positive_values = dataframe['peak_prominance'].where(dataframe['peak_prominance'] > 0)
        # negative_values = dataframe['peak_prominance'].where(dataframe['peak_prominance'] < 0)

        # # Calculate the expanding mean for positive values
        # dataframe['positive_expanding_mean'] = positive_values.expanding().mean()

        # # Calculate the expanding mean for negative values
        # dataframe['negative_expanding_mean'] = negative_values.expanding().mean()

        # # Fill NaN values with the previous non-null values
        # dataframe['positive_expanding_mean'].ffill(inplace=True)
        # dataframe['negative_expanding_mean'].ffill(inplace=True)


        # Print the identified peaks
        #print("Identified Peaks:", peaks)
        
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_length.value)
        # dataframe['vol_z_score'] = (dataframe['volume'] - dataframe['volume'].rolling(window=30).mean()) / dataframe['volume'].rolling(window=30).std()
        # dataframe['vol_anomaly'] = np.where(dataframe['vol_z_score'] > 3, 1, 0)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']
        # dataframe['ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())


        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value

        # dataframe['sma_pc'] = abs((dataframe['sma'] - dataframe['sma'].shift(1)) / dataframe['sma']) * 100
        # dataframe['atr_pcnt'] = (qtpylib.atr(dataframe, window = self.atr.value)) / dataframe['ha_close']
       
        # dataframe['modulation'] = 1 + (dataframe['sma_pc'] * self.trend.value) + (dataframe['atr_pcnt'] * self.volatility.value)

        # min_window = self.peaks.value 
        # max_window = self.bull_bear.value

        # # Set minimum and maximum window size
        # dataframe['order'] = (dataframe['modulation'] * self.peaks.value).round().fillna(self.bull_bear.value).astype(int)
        # dataframe['order'] = np.where(dataframe['order'] > max_window, max_window, dataframe['order'])
        # dataframe['order'] = np.where(dataframe['order'] < min_window, min_window, dataframe['order'])
        
        # if not dataframe['order'].empty:
        #     order = dataframe['order'].iloc[-1]
        # else:
        #     order = self.bear.value

        # peak prominence

        
        # prom = dataframe['peak_prominance'].iloc[-1]
        # if prom != 0:
        #     logger.info(f"*** {pair} *** Prominence 0: {prom}")
        # prom1 = dataframe['peak_prominance'].iloc[-2]
        # if prom1 != 0:
        #     logger.info(f"*** {pair} *** Prominence 1: {prom1}")
        # prom2 = dataframe['peak_prominance'].iloc[-3]
        # if prom2 != 0:
        #     logger.info(f"*** {pair} *** Prominence 2: {prom2}")
        # prom3 = dataframe['peak_prominance'].iloc[-4]
        # if prom3 != 0:
        #     logger.info(f"*** {pair} *** Prominence 3: {prom3}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition1 = (
            (dataframe['decision'] == 1) &
            (dataframe['peak'].shift() == -1) &
            (dataframe['volume'] > 0)
            # (dataframe['close'] < dataframe['sma_dn'])
        )

        dataframe.loc[condition1, 'enter_long'] = 1
        dataframe.loc[condition1, 'enter_tag'] = 'buy'

        # condition2 = (
        #         (dataframe['peak_prominance'].shift() >= dataframe['positive_expanding_mean'].shift()) &
        # #         (dataframe['order'] < (self.peaks.value * self.quick.value)) &
        #         (self.enable2.value == True) &
        #         (dataframe['volume'] > 0)
        # #         (dataframe['close'] < dataframe['sma'])
        #     )

        # dataframe.loc[condition2, 'enter_long'] = 1
        # dataframe.loc[condition2, 'enter_tag'] = 'minima_check_1'

        # condition3 = (
        #         (dataframe['peak_prominance'].shift() >= dataframe['positive_expanding_mean'].shift()) &
        #         (self.enable3.value == True) &
        #         (dataframe['volume'] > 0) &
        #         (dataframe['close'].shift() < dataframe['sma'].shift())
        #     )

        # dataframe.loc[condition3, 'enter_long'] = 1
        # dataframe.loc[condition3, 'enter_tag'] = 'minima_check_below_2'

        # condition4 = (
        #         (dataframe['peak_prominance'].shift() >= dataframe['positive_expanding_mean'].shift()) &
        #         (self.enable4.value == True) &
        #         (dataframe['volume'] > 0) &
        #         (dataframe['close'].shift() < dataframe['sma'].shift())
        #     )

        # dataframe.loc[condition4, 'enter_long'] = 1
        # dataframe.loc[condition4, 'enter_tag'] = 'minima_check_below_1'

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition5 = (
            (dataframe['decision'] == -1) &
            (dataframe['peak'].shift() == 1) &
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition5, 'exit_long'] = 1
        dataframe.loc[condition5, 'exit_tag'] = 'maxima_check_2'

        # condition6 = (
        #         (dataframe['peak_prominance'].shift(3) < 0)&
        #         (dataframe['close'].shift(3) < dataframe['sma'].shift(3))
                
        #     )

        # dataframe.loc[condition6, 'exit_long'] = 1
        # dataframe.loc[condition6, 'exit_tag'] = 'maxima_check_below_sma_2'

        # condition7 = (
        #         (dataframe['peak_prominance'].shift(2) <= dataframe['negative_expanding_mean'].shift(2)) 
        #     )

        # dataframe.loc[condition7, 'exit_long'] = 1
        # dataframe.loc[condition7, 'exit_tag'] = 'maxima_check_2'

        # condition8 = (
        #         (dataframe['peak_prominance'].shift(2) < 0)&
        #         (dataframe['close'].shift(2) < dataframe['sma'].shift(2))
                
        #     )

        # dataframe.loc[condition8, 'exit_long'] = 1
        # dataframe.loc[condition8, 'exit_tag'] = 'maxima_check_below_sma_2'

        # condition9 = (
        #         (dataframe['peak_prominance'].shift() <= dataframe['negative_expanding_mean'].shift()) 
        #     )

        # dataframe.loc[condition9, 'exit_long'] = 1
        # dataframe.loc[condition9, 'exit_tag'] = 'maxima_check_1'

        # condition10 = (
        #         (dataframe['peak_prominance'].shift() < 0)&
        #         (dataframe['close'].shift() < dataframe['sma'].shift())
                
        #     )

        # dataframe.loc[condition10, 'exit_long'] = 1
        # dataframe.loc[condition10, 'exit_tag'] = 'maxima_check_below_sma_1'

        return dataframe

