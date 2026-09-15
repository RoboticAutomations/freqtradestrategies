import requests
import json
import re
from urllib.parse import urlparse, urlunparse

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from datetime import datetime, timezone, timedelta

from pandas import DataFrame
from typing import Any, Tuple, Dict, List, Literal, Optional
import logging
import time
import numpy as np
from freqtrade.persistence import Trade
import functools

# Import prompt templates
from prompting import SYSTEM_PROMPT_INDICATORS_ONLY, USER_PROMPT_ENTER_FULL, USER_PROMPT_EXIT_FULL

# deps
from pydantic import BaseModel, Field, computed_field, ConfigDict, field_validator

try:
    from instructor import Instructor
    import instructor
except ImportError:
    Instructor = Any
    instructor = None




MarketSide = Optional[Literal["LONG", "SHORT"]]
TradeAction = Literal["LONG_ENTER", "SHORT_ENTER", "NEUTRAL", "HOLD", "LONG_EXIT", "SHORT_EXIT"]


class TradingRecommendation(BaseModel):
    """Structured trading recommendation returned by the LLM."""

    model_config = ConfigDict(extra="ignore")

    action: TradeAction = Field(
        description="Recommended trading action based on technical analysis"
    )

    summary: str = Field(
        description=f"Concise justification (MAXIMUM {1000} characters) linking indicators to recommended action",
        #max_length=10000
    )


class GPTTraderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_profit: float = 0.03
    target_duration: int = 100
    stoploss: float = Field(default=0.04, validation_alias="stop_loss")

    # LLM configuration
    llm_offline: bool = True
    llm_provider: Literal["openai_compat", "instructor"] = "openai_compat"
    llm_base_url: Optional[str] = None
    llm_model: str = Field(default="gpt-3.5-turbo", min_length=1)
    llm_api_key: Optional[str] = ""
    llm_timeout_s: float = 560.0
    llm_temperature: float = 0.8
    llm_trace: bool = True
    llm_trace_max_chars: int = 30000

    # Advanced LLM control parameters
    max_tokens: Optional[int] = Field(default=2000, ge=1, le=32768, description="Maximum tokens in response")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Reduce repetition")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Encourage new topics")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible outputs")

    # Context and prompt control
    max_context_chars: int = Field(default=10000, ge=1000, le=50000, description="Maximum context length")

    # Error handling and retry
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    retry_delay_s: float = Field(default=1.0, ge=0.1, le=60.0, description="Delay between retries")
    fallback_model: Optional[str] = Field(default=None, description="Fallback model on failure")

    # Rate limiting
    requests_per_minute: Optional[int] = Field(default=None, ge=1, le=1000, description="Rate limit")

    # Response validation
    strict_json_parsing: bool = Field(default=True, description="Strict JSON validation")
    response_timeout_s: float = Field(default=30.0, ge=1.0, le=300.0, description="Response parsing timeout")

    # Model-specific parameters
    model_specific_params: Dict[str, Any] = Field(default_factory=dict, description="Extra parameters for specific models")
    use_structured_outputs: bool = Field(default=False, description="Use structured outputs if available")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, v: Any) -> str:
        """Normalize friendly provider names from config to supported internal values.

        Accepts common aliases like 'LLMStudio'/'lmstudio'/'ollama' and maps them to 'openai_compat'.
        """
        if v is None:
            return "openai_compat"
        if isinstance(v, str):
            key = v.strip().lower()
            aliases_openai_compat = {
                "openai_compat",
                "openai-compat",
                "openai",
                "lmstudio",
                "llmstudio",
                "lms",
                "ollama",
            }
            if key in aliases_openai_compat:
                return "openai_compat"
            if key == "instructor":
                return "instructor"
        return v

def safe_json_loads(json_text: str, strict: bool = False, sanitize: bool = False) -> dict:
    """
    Parse JSON with control character handling.

    Args:
        json_text: JSON string to parse
        strict: If False, allows control characters in strings (default: False)
        sanitize: If True, removes problematic control characters (default: False)
                 Note: Only removes non-printable characters, NOT JSON structural characters

    Returns:
        Parsed JSON as dict

    Raises:
        json.JSONDecodeError: If JSON is invalid
    """
    if sanitize:
        import re
        # Only remove problematic control characters that break JSON parsing
        # Preserve structural JSON characters: {, }, [, ], ", :, ,
        # Remove only: NULL, non-breaking spaces, and other non-printable control chars
        json_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', json_text)

    return json.loads(json_text, strict=strict)


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat.completions client (works with OpenAI/LMStudio/Ollama OpenAI API).

    Expects an endpoint like:
    - OpenAI: https://api.openai.com/v1
    - LMStudio: http://localhost:1234/v1
    - Ollama (OpenAI compat): http://localhost:11434/v1
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        timeout_s: float,
        temperature: float,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
        requests_per_minute: Optional[int] = None,
        model_specific_params: Optional[Dict[str, Any]] = None,
        fallback_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.requests_per_minute = requests_per_minute
        self.model_specific_params = model_specific_params or {}
        self.fallback_model = fallback_model
        self._last_request_time = 0.0

    def chat_completions_url(self) -> str:
        """Return a best-effort OpenAI-compatible chat completions URL.

        Supports base URLs like:
        - http://host:1234        -> http://host:1234/v1/chat/completions
        - http://host:1234/v1     -> http://host:1234/v1/chat/completions
        - https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
        """
        parsed = urlparse(self.base_url)
        path = (parsed.path or "").rstrip("/")

        # If user already provided the full endpoint, respect it
        if path.endswith("/chat/completions"):
            fixed_path = path
        else:
            if not path:
                path = "/v1"
            elif not path.endswith("/v1"):
                path = f"{path}/v1"
            fixed_path = f"{path}/chat/completions"

        return urlunparse(parsed._replace(path=fixed_path))

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        
        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        # Handle both with and without language specifier
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # If it's already clean JSON, return it
        if text.startswith("{") and text.endswith("}"):
            return text
        
        # Otherwise search for JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        
        raise ValueError("LLM did not return a JSON object")

    def complete_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        import time

        url = self.chat_completions_url()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        if self.seed is not None:
            payload["seed"] = self.seed

        # Add model-specific parameters
        payload.update(self.model_specific_params)

        if "openai.com" in (urlparse(self.base_url).netloc or ""):
            payload["response_format"] = {"type": "json_object"}

        # Rate limiting
        if self.requests_per_minute:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            min_interval = 60.0 / self.requests_per_minute
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)

        last_exception = None
        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        for model_attempt in models_to_try:
            payload["model"] = model_attempt

            for attempt in range(self.max_retries):
                try:
                    self._last_request_time = time.time()
                    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)

                    resp.raise_for_status()
                    return resp.json()

                except requests.HTTPError as e:
                    body = (resp.text or "").strip()
                    if len(body) > 2000:
                        body = body[:2000] + "\n... (truncated)"
                    last_exception = requests.HTTPError(
                        f"{e} response_body={body}",
                        response=resp,
                    )
                except Exception as e:
                    last_exception = e

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_s * (2 ** attempt))  # Exponential backoff

        # If all retries and models failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unknown error in LLM request")

    def complete_json(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        import json
        import re

        data = self.complete_raw(messages)
        content = data["choices"][0]["message"]["content"]

        if self.strict_json_parsing:
            json_text = self._extract_json(content)
            return safe_json_loads(json_text)
        else:
            # Try strict parsing first, fallback to lenient
            try:
                json_text = self._extract_json(content)
                return safe_json_loads(json_text)
            except (ValueError, json.JSONDecodeError):
                # Fallback: try to parse any JSON-like content
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    try:
                        return safe_json_loads(match.group(0))
                    except json.JSONDecodeError:
                        pass
                raise ValueError("LLM did not return valid JSON")

logger = logging.getLogger(__name__)


class DarkProphetLLMAIV3(IFreqaiModel):
    """
    Base model for letting an LLM take full control of your bot...
    """

    def _clean_kitchen(self, dk: FreqaiDataKitchen) -> None:
        """Populate required FreqAI placeholders.

        FreqAI expects these keys to exist for persistence, even when the model
        does not train a classical ML estimator.
        """
        dk.data_dictionary["train_features"] = DataFrame(np.zeros(1))
        dk.data_dictionary["train_dates"] = DataFrame(np.zeros(1))

        zero_labels = {label: 0 for label in dk.label_list}
        dk.data.setdefault("labels_mean", zero_labels)
        dk.data.setdefault("labels_std", zero_labels)

    def _ensure_label_stats(self, dk: FreqaiDataKitchen, labels: List[str]) -> None:
        """Ensure label mean/std entries exist for FreqAI pipelines.

        Some FreqAI paths expect `dk.data['labels_mean']` and `dk.data['labels_std']`
        to contain all labels used during prediction, even for placeholder labels.
        """
        labels_mean = dk.data.setdefault("labels_mean", {})
        labels_std = dk.data.setdefault("labels_std", {})
        for label in labels:
            labels_mean.setdefault(label, 0)
            labels_std.setdefault(label, 0)

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.gpt_config = GPTTraderConfig(**self.freqai_info.get('GPTTrader', {}))

        # Do not annotate with optional Instructor/SearchResponse types here,
        # since those may be replaced with Any when optional deps are missing.
        self._llm_openai_compat = None
        self._llm_instructor = None

        if self.gpt_config.llm_provider == "instructor":
            if instructor is None:
                raise ImportError("llm_provider='instructor' requires the 'instructor' package")

            if self.gpt_config.llm_offline:
                raise ValueError(
                    "llm_offline=true is not supported with llm_provider='instructor'. "
                    "Use llm_provider='openai_compat' for offline servers (LMStudio/Ollama)."
                )

            if not (self.gpt_config.llm_api_key or "").strip():
                raise ValueError("llm_offline=false requires llm_api_key to be set")

            self._llm_instructor = instructor.from_provider(
                model=self.gpt_config.llm_model,
                api_key=self.gpt_config.llm_api_key,
            )
        else:
            base_url = self.gpt_config.llm_base_url

            if self.gpt_config.llm_offline:
                if not base_url:
                    raise ValueError(
                        "llm_offline=true requires llm_base_url (example: http://localhost:1234/v1 for LMStudio)"
                    )
            else:
                if not (self.gpt_config.llm_api_key or "").strip():
                    raise ValueError("llm_offline=false requires llm_api_key to be set")
                if not base_url:
                    base_url = "https://api.openai.com/v1"

            self._llm_openai_compat = OpenAICompatLLM(
                base_url=base_url,
                model=self.gpt_config.llm_model,
                api_key=self.gpt_config.llm_api_key,
                timeout_s=self.gpt_config.llm_timeout_s,
                temperature=self.gpt_config.llm_temperature,
                max_tokens=self.gpt_config.max_tokens,
                top_p=self.gpt_config.top_p,
                frequency_penalty=self.gpt_config.frequency_penalty,
                presence_penalty=self.gpt_config.presence_penalty,
                seed=self.gpt_config.seed,
                max_retries=self.gpt_config.max_retries,
                retry_delay_s=self.gpt_config.retry_delay_s,
                requests_per_minute=self.gpt_config.requests_per_minute,
                model_specific_params=self.gpt_config.model_specific_params,
                fallback_model=self.gpt_config.fallback_model,
            )

        if self.gpt_config.llm_trace and self._llm_openai_compat is not None:
            logger.warning(
                "LLM_TRACE init: offline=%s provider=%s endpoint_url=%s model=%s",
                self.gpt_config.llm_offline,
                self.gpt_config.llm_provider,
                self._llm_openai_compat.chat_completions_url(),
                self.gpt_config.llm_model,
            )

        # Prefer the strict JSON system prompt for maximum reliability with local LLMs.
        self.cached_system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT_INDICATORS_ONLY,
        }

        logging.info("DarkProphetLLMAIV2 initialized (technical analysis only)")


    def train(
            self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Build a custom prompt asking the LLM to take an action based on
        the current technical indicator context and (if present) open position.
        """
        logger.info(f"GPTTrader training for {pair}")

        indicator_context = self._build_indicator_context(unfiltered_df)

        side, profit, duration = self.get_state_info(pair)

        if not side:
            logger.info(f"No position for {pair}, getting entry recommendation")
            result = self._get_entry_recommendation(pair, indicator_context)
        else:
            logger.info(f"Position exists for {pair}: {side}, profit: {profit:.2%}, duration: {duration} candles")
            result = self._get_exit_recommendation(pair, indicator_context, side=side, profit=profit, duration=duration)

        if self.gpt_config.llm_trace:
            logger.info(
                "LLM_TRACE final action for %s: %s summary=%s",
                pair,
                result.action,
                result.summary,
            )

        self._clean_kitchen(dk)

        # Convert to dict with Python native types for serialization
        return {
            "action": str(result.action),
            "summary": str(result.summary)
        }

    def _build_indicator_context(self, df: DataFrame) -> str:

        if df is None or df.empty:
            logger.warning("_build_indicator_context: DataFrame is None or empty")
            return "(no dataframe)"

        all_columns = list(df.columns)
        indicator_columns = [c for c in all_columns if c.startswith('%-')]
        ohlcv_columns = [c for c in all_columns if c in ['date', 'open', 'high', 'low', 'close', 'volume']]
        other_columns = [c for c in all_columns if not c.startswith('%-') and c not in ohlcv_columns]

        if self.gpt_config.llm_trace:
            logger.info("="*80)
            logger.info("LLM_TRACE _build_indicator_context: STRATEGY INDICATORS RECEIVED")
            logger.info("="*80)
            logger.info(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
            logger.info(f"OHLCV columns ({len(ohlcv_columns)}): {ohlcv_columns}")
            logger.info(f"Indicator columns ({len(indicator_columns)}): {sorted(indicator_columns)}")
            logger.info(f"Other columns ({len(other_columns)}): {other_columns[:20]}{'...' if len(other_columns) > 20 else ''}")
            logger.info("-"*80)

        last = df.iloc[-1]

        def safe_get(col: str, decimals: int = 6) -> str:
            """Safely get column value with formatting."""
            if col not in df.columns:
                return "N/A"
            v = last.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.{decimals}f}"
            return str(v)

        lines = []

        lines.append("## PRICE ACTION")
        lines.append(f"- Date: {safe_get('date', 0)}")
        lines.append(f"- Open: {safe_get('open', 2)} | High: {safe_get('high', 2)} | Low: {safe_get('low', 2)} | Close: {safe_get('close', 2)}")
        lines.append(f"- Volume: {safe_get('volume', 0)}")
        lines.append(f"- Return (1 candle): {safe_get('%-return_1', 6)} ({float(safe_get('%-return_1', 6).replace('N/A', '0')) * 100:.3f}%)")
        lines.append(f"- Volatility (20): {safe_get('%-volatility_20', 6)}")

        if "close" in df.columns:
            closes = df["close"].tail(6).tolist()
            lines.append(f"- Recent Closes (6): {[round(c, 2) for c in closes]}")

        lines.append("")
        lines.append("## TREND ANALYSIS (EMAs)")
        lines.append(f"- EMA Fast (12): {safe_get('%-ema_fast', 2)}")
        lines.append(f"- EMA Slow (26): {safe_get('%-ema_slow', 2)}")
        lines.append(f"- EMA Mid (50): {safe_get('%-ema_mid', 2)}")
        lines.append(f"- EMA Long (200): {safe_get('%-ema_long', 2)}")
        lines.append(f"- EMA Diff (fast-slow normalized): {safe_get('%-ema_diff', 6)}")
        lines.append(f"- EMA Mid Slope (10 periods): {safe_get('%-ema_mid_slope_10', 6)}")

        # EMA Alignment flags
        ema_trend = safe_get('%-ema_trend_mid_long', 0)
        price_above_mid = safe_get('%-price_above_ema_mid', 0)
        price_above_long = safe_get('%-price_above_ema_long', 0)
        lines.append(f"- EMA Alignment: Mid>Long={ema_trend} | Price>Mid={price_above_mid} | Price>Long={price_above_long}")

        # Trend interpretation
        try:
            close_val = float(last.get('close', 0))
            ema_fast_val = float(last.get('%-ema_fast', 0))
            ema_long_val = float(last.get('%-ema_long', 0))
            if close_val > ema_fast_val > ema_long_val:
                trend_state = "STRONG UPTREND ↑↑"
            elif close_val > ema_long_val:
                trend_state = "UPTREND ↑"
            elif close_val < ema_fast_val < ema_long_val:
                trend_state = "STRONG DOWNTREND ↓↓"
            elif close_val < ema_long_val:
                trend_state = "DOWNTREND ↓"
            else:
                trend_state = "RANGING ↔"
            lines.append(f"- **Trend State**: {trend_state}")
        except:
            pass

        # SECTION 3: SUPERTREND

        lines.append("")
        lines.append("## SUPERTREND (14,3)")
        lines.append(f"- SuperTrend Line: {safe_get('%-supertrend_14_3', 2)}")
        st_dir = safe_get('%-supertrend_dir_14_3', 0)
        st_dir_text = "BULLISH ↑" if st_dir == "1" else "BEARISH ↓" if st_dir == "-1" else st_dir
        lines.append(f"- SuperTrend Direction: {st_dir_text}")
        lines.append(f"- Distance from ST (ATR units): {safe_get('%-supertrend_dist_atr_14_3', 2)}")


        # SECTION 4: MOMENTUM OSCILLATORS

        lines.append("")
        lines.append("## MOMENTUM OSCILLATORS")

        # RSI with interpretation
        rsi_val = safe_get('%-rsi_14', 2)
        try:
            rsi_num = float(rsi_val)
            if rsi_num < 30:
                rsi_state = "OVERSOLD "
            elif rsi_num < 40:
                rsi_state = "Near oversold"
            elif rsi_num > 70:
                rsi_state = "OVERBOUGHT "
            elif rsi_num > 60:
                rsi_state = "Near overbought"
            else:
                rsi_state = "Neutral"
            lines.append(f"- RSI (14): {rsi_val} [{rsi_state}]")
        except:
            lines.append(f"- RSI (14): {rsi_val}")

        # RSI Divergences
        bull_div = safe_get('%-bullish_div_rsi_48', 0)
        bear_div = safe_get('%-bearish_div_rsi_48', 0)
        if bull_div == "1":
            lines.append(f"BULLISH DIVERGENCE DETECTED")
        if bear_div == "1":
            lines.append(f"BEARISH DIVERGENCE DETECTED")

        # CCI
        cci_val = safe_get('%-cci_20', 2)
        try:
            cci_num = float(cci_val)
            if cci_num < -100:
                cci_state = "OVERSOLD"
            elif cci_num > 100:
                cci_state = "OVERBOUGHT"
            else:
                cci_state = "Neutral"
            lines.append(f"- CCI (20): {cci_val} [{cci_state}]")
        except:
            lines.append(f"- CCI (20): {cci_val}")

        # MFI
        mfi_val = safe_get('%-mfi_14', 2)
        try:
            mfi_num = float(mfi_val)
            if mfi_num < 20:
                mfi_state = "OVERSOLD money flowing out"
            elif mfi_num > 80:
                mfi_state = "OVERBOUGHT money flowing in"
            else:
                mfi_state = "Neutral"
            lines.append(f"- MFI (14): {mfi_val} [{mfi_state}]")
        except:
            lines.append(f"- MFI (14): {mfi_val}")


        #  5: ADX / TREND STRENGTH

        lines.append("## TREND STRENGTH (ADX/DI)")
        adx_val = safe_get('%-adx_20', 2)
        di_plus = safe_get('%-di_plus_20', 2)
        di_minus = safe_get('%-di_minus_20', 2)

        try:
            adx_num = float(adx_val)
            if adx_num < 20:
                adx_state = "WEAK/NO TREND (ranging)"
            elif adx_num < 40:
                adx_state = "MODERATE TREND"
            elif adx_num < 60:
                adx_state = "STRONG TREND"
            else:
                adx_state = "VERY STRONG TREND"
            lines.append(f"- ADX (20): {adx_val} [{adx_state}]")
        except:
            lines.append(f"- ADX (20): {adx_val}")

        lines.append(f"- DI+ (20): {di_plus}")
        lines.append(f"- DI- (20): {di_minus}")

        try:
            if float(di_plus) > float(di_minus):
                lines.append(f"- DI Signal: BULLISH (DI+ > DI-)")
            else:
                lines.append(f"- DI Signal: BEARISH (DI- > DI+)")
        except:
            pass


        # 6: VOLATILITY (ATR, Bollinger)

        lines.append("")
        lines.append("##  VOLATILITY")
        lines.append(f"- ATR (14 Wilder): {safe_get('%-atr_14_wilder', 4)}")
        lines.append(f"- ATR (20 ADX): {safe_get('%-atr_for_adx_20', 4)}")

        # Bollinger Bands
        lines.append(f"- BB Upper (20,2): {safe_get('%-bb_upper_20', 2)}")
        lines.append(f"- BB Mid (20): {safe_get('%-bb_mid_20', 2)}")
        lines.append(f"- BB Lower (20,2): {safe_get('%-bb_lower_20', 2)}")
        lines.append(f"- BB Width: {safe_get('%-bb_width_20', 4)}")

        # %B position
        bb_pct = safe_get('%-bb_percent_b_20', 4)
        try:
            bb_pct_num = float(bb_pct)
            if bb_pct_num > 1.0:
                bb_state = "ABOVE UPPER BAND (overbought)"
            elif bb_pct_num > 0.8:
                bb_state = "Near upper band"
            elif bb_pct_num < 0.0:
                bb_state = "BELOW LOWER BAND (oversold)"
            elif bb_pct_num < 0.2:
                bb_state = "Near lower band"
            else:
                bb_state = "Inside bands"
            lines.append(f"- BB %B Position: {bb_pct} [{bb_state}]")
        except:
            lines.append(f"- BB %B Position: {bb_pct}")
#

        # 7: DONCHIAN CHANNELS (Breakout)

        lines.append("")
        lines.append("##  DONCHIAN CHANNELS (32)")
        lines.append(f"- Donchian High: {safe_get('%-donchian_high_32', 2)}")
        lines.append(f"- Donchian Low: {safe_get('%-donchian_low_32', 2)}")

        dc_pos = safe_get('%-donchian_pos_32', 4)
        try:
            dc_pos_num = float(dc_pos)
            if dc_pos_num > 0.9:
                dc_state = "NEAR BREAKOUT HIGH ↑"
            elif dc_pos_num > 0.7:
                dc_state = "Upper range"
            elif dc_pos_num < 0.1:
                dc_state = "NEAR BREAKOUT LOW ↓"
            elif dc_pos_num < 0.3:
                dc_state = "Lower range"
            else:
                dc_state = "Mid-range"
            lines.append(f"- Donchian Position (0-1): {dc_pos} [{dc_state}]")
        except:
            lines.append(f"- Donchian Position (0-1): {dc_pos}")


        #  8: REGIME DETECTION

        lines.append("")
        lines.append("##  MARKET REGIME")

        # Choppiness Index
        chop = safe_get('%-chop_14', 2)
        try:
            chop_num = float(chop)
            if chop_num > 61.8:
                chop_state = "CHOPPY/RANGING (avoid trends)"
            elif chop_num < 38.2:
                chop_state = "TRENDING (momentum strategies)"
            else:
                chop_state = "Transitional"
            lines.append(f"- Choppiness (14): {chop} [{chop_state}]")
        except:
            lines.append(f"- Choppiness (14): {chop}")

        # Efficiency Ratio
        er = safe_get('%-eff_ratio_20', 4)
        try:
            er_num = float(er)
            if er_num > 0.6:
                er_state = "HIGH EFFICIENCY (strong directional move)"
            elif er_num < 0.3:
                er_state = "LOW EFFICIENCY (noisy/sideways)"
            else:
                er_state = "Moderate"
            lines.append(f"- Efficiency Ratio (20): {er} [{er_state}]")
        except:
            lines.append(f"- Efficiency Ratio (20): {er}")


        # 9: VOLUME ANALYSIS

        lines.append("")
        lines.append("##  VOLUME ANALYSIS")

        # Volume Z-scores
        vol_z_48 = safe_get('%-volume_z_48', 2)
        vol_z_96 = safe_get('%-volume_z_96', 2)
        try:
            vol_z_num = float(vol_z_48)
            if vol_z_num > 2.0:
                vol_state = "VERY HIGH VOLUME SPIKE"
            elif vol_z_num > 1.0:
                vol_state = "Above average"
            elif vol_z_num < -1.0:
                vol_state = "Below average (low interest)"
            else:
                vol_state = "Normal"
            lines.append(f"- Volume Z-score (48): {vol_z_48} [{vol_state}]")
        except:
            lines.append(f"- Volume Z-score (48): {vol_z_48}")
        lines.append(f"- Volume Z-score (96): {vol_z_96}")

        # OBV
        lines.append(f"- OBV: {safe_get('%-obv', 0)}")
        lines.append(f"- OBV Z-score (48): {safe_get('%-obv_z_48', 2)}")

        # CMF
        cmf = safe_get('%-cmf_20', 4)
        try:
            cmf_num = float(cmf)
            if cmf_num > 0.1:
                cmf_state = "ACCUMULATION (buying pressure)"
            elif cmf_num < -0.1:
                cmf_state = "DISTRIBUTION (selling pressure)"
            else:
                cmf_state = "Neutral"
            lines.append(f"- CMF (20): {cmf} [{cmf_state}]")
        except:
            lines.append(f"- CMF (20): {cmf}")


        # 10: VWAP (Fair Value)

        lines.append("")
        lines.append("##  VWAP (Fair Value)")
        lines.append(f"- VWAP (48 / 12h): {safe_get('%-vwap_48', 2)}")
        lines.append(f"- VWAP (96 / 24h): {safe_get('%-vwap_96', 2)}")

        vwap_dist_48 = safe_get('%-vwap_dist_48_atr', 2)
        vwap_dist_96 = safe_get('%-vwap_dist_96_atr', 2)
        try:
            vwap_d = float(vwap_dist_48)
            if vwap_d > 1.5:
                vwap_state = "OVEREXTENDED ABOVE (mean reversion risk)"
            elif vwap_d < -1.5:
                vwap_state = "OVEREXTENDED BELOW (bounce potential)"
            else:
                vwap_state = "Near fair value"
            lines.append(f"- VWAP Distance 48 (ATR): {vwap_dist_48} [{vwap_state}]")
        except:
            lines.append(f"- VWAP Distance 48 (ATR): {vwap_dist_48}")
        lines.append(f"- VWAP Distance 96 (ATR): {vwap_dist_96}")


        # 11: MARKET STRUCTURE (Pivots)

        lines.append("")
        lines.append("##  MARKET STRUCTURE (Pivots)")
        lines.append(f"- Last Pivot High: {safe_get('%-pivot_high', 2)}")
        lines.append(f"- Last Pivot Low: {safe_get('%-pivot_low', 2)}")
        lines.append(f"- Distance to Pivot High (ATR): {safe_get('%-pivot_high_dist_atr', 2)}")
        lines.append(f"- Distance to Pivot Low (ATR): {safe_get('%-pivot_low_dist_atr', 2)}")


        #  12: VOLUME PROFILE

        lines.append("")
        lines.append("##  VOLUME PROFILE (96)")
        lines.append(f"- POC (Point of Control): {safe_get('%-poc_96', 2)}")
        lines.append(f"- LVN (Low Volume Node): {safe_get('%-lvn_96', 2)}")
        lines.append(f"- Distance to POC (ATR): {safe_get('%-poc_dist_96_atr', 2)}")
        lines.append(f"- Distance to LVN (ATR): {safe_get('%-lvn_dist_96_atr', 2)}")

        # 13: MARKET REGIME DETECTION

        lines.append("")
        lines.append("## 🎭 MARKET REGIME DETECTION (Advanced AI)")

        # Regime Classification
        regime = safe_get('%-market_regime', 0)
        regime_conf = safe_get('%-regime_confidence', 2)
        regime_transition = safe_get('%-regime_transition', 0)

        # Regime emoji and interpretation
        regime_emoji = {
            'STRONG_TREND': '🚀',
            'WEAK_TREND': '📈',
            'RANGING_CALM': '😴',
            'RANGING_VOLATILE': '⚡',
            'BREAKOUT_SETUP': '💎',
            'CHAOTIC': '🌪️',
            'TRANSITIONAL': '🔄',
            'UNKNOWN': '❓'
        }

        regime_strategy = {
            'STRONG_TREND': 'TREND FOLLOWING optimal. Ride momentum, wide stops.',
            'WEAK_TREND': 'CAUTIOUS TREND. Tighter stops, partial profits.',
            'RANGING_CALM': 'MEAN REVERSION. Buy support, sell resistance.',
            'RANGING_VOLATILE': 'HIGH RISK. Avoid or use tight stops.',
            'BREAKOUT_SETUP': 'BREAKOUT WATCH. Compression before expansion.',
            'CHAOTIC': 'STAY OUT. Too unpredictable for reliable trades.',
            'TRANSITIONAL': 'WAIT FOR CLARITY. Regime shifting.',
            'UNKNOWN': 'INSUFFICIENT DATA.'
        }

        emoji = regime_emoji.get(regime, '❓')
        strategy = regime_strategy.get(regime, 'Assess carefully.')

        lines.append(f"- **Current Regime**: {emoji} {regime} (Confidence: {regime_conf}%)")
        lines.append(f"- **Strategy**: {strategy}")

        if regime_transition == "1":
            lines.append(f" **REGIME TRANSITION DETECTED** - Market shifting, be cautious!")

        # Component Scores
        trend_str = safe_get('%-trend_strength', 2)
        range_sc = safe_get('%-range_score', 2)
        vol_sc = safe_get('%-volatility_score', 2)
        vol_regime = safe_get('%-volatility_regime', 0)
        trend_dir = safe_get('%-trend_direction', 0)

        # DEBUG: Check if scores are invalid
        if trend_str == "N/A" or range_sc == "N/A" or vol_sc == "N/A":
            lines.append("")
            lines.append("⚠️ **WARNING: Regime scores are N/A (insufficient data or calculation error)**")
            lines.append(f"  - Trend Strength: {trend_str}")
            lines.append(f"  - Range Score: {range_sc}")
            lines.append(f"  - Volatility Score: {vol_sc}")
            lines.append("  - This typically happens in the first ~288 candles (REGIME_LOOKBACK_LONG)")
            lines.append("  - Market Regime will be 'UNKNOWN' until sufficient data is available")
            lines.append("")

        try:
            ts_num = float(trend_str)
            if ts_num < 20:
                ts_interp = "NO TREND"
            elif ts_num < 40:
                ts_interp = "WEAK"
            elif ts_num < 60:
                ts_interp = "MODERATE"
            elif ts_num < 80:
                ts_interp = "STRONG"
            else:
                ts_interp = "VERY STRONG"
            lines.append(f"- Trend Strength: {trend_str}/100 [{ts_interp}]")
        except:
            lines.append(f"- Trend Strength: {trend_str}/100")

        try:
            rs_num = float(range_sc)
            if rs_num < 30:
                rs_interp = "TRENDING"
            elif rs_num < 50:
                rs_interp = "TRANSITIONAL"
            elif rs_num < 70:
                rs_interp = "RANGING"
            else:
                rs_interp = "EXTREMELY CHOPPY"
            lines.append(f"- Range/Chop Score: {range_sc}/100 [{rs_interp}]")
        except:
            lines.append(f"- Range/Chop Score: {range_sc}/100")

        lines.append(f"- Volatility Score: {vol_sc}/100 [{vol_regime}]")

        try:
            td_num = float(trend_dir)
            if td_num > 0:
                td_text = "BULLISH ↑"
            elif td_num < 0:
                td_text = "BEARISH ↓"
            else:
                td_text = "NEUTRAL ↔"
            lines.append(f"- Trend Direction: {td_text}")
        except:
            lines.append(f"- Trend Direction: {trend_dir}")

        # Regime-specific advice
        lines.append("")
        lines.append("### 💡 Regime-Specific Trading Rules:")

        if regime in ['STRONG_TREND', 'WEAK_TREND']:
            lines.append("- ✅ Favor trend-following entries (LONG in uptrend, SHORT in downtrend)")
            lines.append("- ✅ Use wider stops to avoid noise")
            lines.append("- ✅ Let profits run, trend has momentum")
            lines.append("- ⛔ Avoid counter-trend trades")
        elif regime in ['RANGING_CALM', 'RANGING_VOLATILE']:
            lines.append("- ✅ Mean reversion strategies (buy lows, sell highs)")
            lines.append("- ✅ Use support/resistance, VWAP, pivot points")
            lines.append("- ✅ Tight stops (range-bound moves)")
            lines.append("- ⛔ Avoid trend-following, no sustained momentum")
        elif regime == 'BREAKOUT_SETUP':
            lines.append("- ⚠️ COMPRESSION PHASE - breakout imminent")
            lines.append("- ✅ Watch for volume spike + price breakout")
            lines.append("- ✅ Enter on breakout confirmation with volume")
            lines.append("- ⛔ Avoid range-bound trades, wait for breakout")
        elif regime == 'CHAOTIC':
            lines.append("- 🚫 STAY OUT - market too unpredictable")
            lines.append("- 🚫 High volatility + no clear direction")
            lines.append("- 🚫 Extremely high risk of whipsaw")
            lines.append("- ⏸️ Wait for regime to stabilize")
        elif regime == 'TRANSITIONAL':
            lines.append("- ⚠️ Market regime changing")
            lines.append("- 🔍 Wait for new regime to establish")
            lines.append("- 📊 Monitor next few candles for confirmation")
            lines.append("- ⏸️ Reduce position size or stay flat")

        #  14: TIME CONTEXT

        lines.append("")
        lines.append("## 🕐 TIME CONTEXT")
        lines.append(f"- Day of Week: {safe_get('%-day_of_week', 0)} (0=Mon, 6=Sun)")
        lines.append(f"- Hour of Day (UTC): {safe_get('%-hour_of_day', 0)}")

        #  15: SUMMARY / QUICK SIGNALS

        lines.append("")
        lines.append("## 🎯 QUICK SIGNAL SUMMARY")

        # Build quick summary
        signals = []
        try:
            rsi_num = float(safe_get('%-rsi_14', 2))
            if rsi_num < 35:
                signals.append("RSI OVERSOLD ✅")
            elif rsi_num > 65:
                signals.append("RSI OVERBOUGHT ⛔")
        except:
            pass

        try:
            st_dir = float(safe_get('%-supertrend_dir_14_3', 0))
            signals.append(f"SuperTrend {'BULL ✅' if st_dir > 0 else 'BEAR ⛔'}")
        except:
            pass

        try:
            chop_num = float(safe_get('%-chop_14', 2))
            if chop_num > 61.8:
                signals.append("CHOPPY MARKET ⚠️")
            elif chop_num < 38.2:
                signals.append("TRENDING MARKET ✅")
        except:
            pass

        try:
            cmf_num = float(safe_get('%-cmf_20', 4))
            if cmf_num > 0.1:
                signals.append("CMF ACCUMULATION ✅")
            elif cmf_num < -0.1:
                signals.append("CMF DISTRIBUTION ⛔")
        except:
            pass
        try:
            adx_num = float(safe_get('%-adx_20', 2))
            di_plus_num = float(safe_get('%-di_plus_20', 2))
            di_minus_num = float(safe_get('%-di_minus_20', 2))
            if adx_num > 25:
                if di_plus_num > di_minus_num:
                    signals.append("ADX STRONG BULLISH ✅")
                else:
                    signals.append("ADX STRONG BEARISH ⛔")
        except:
            pass

        if bull_div == "1":
            signals.append("BULLISH DIVERGENCE ✅")
        if bear_div == "1":
            signals.append("BEARISH DIVERGENCE ⛔")

        if signals:
            lines.append(f"- Signals: {' | '.join(signals)}")
        else:
            lines.append("- Signals: Mixed/Neutral")


        # LOG: Final context statistics

        context_text = "\n".join(lines)

        # Truncate if exceeds max_context_chars
        if len(context_text) > self.gpt_config.max_context_chars:
            context_text = context_text[:self.gpt_config.max_context_chars - 100] + "\n\n[CONTEXT TRUNCATED FOR LENGTH]"

        if self.gpt_config.llm_trace:
            # Estimate tokens (rough: ~4 chars per token for English)
            estimated_tokens = len(context_text) // 4

            logger.info("-"*80)
            logger.info("LLM_TRACE _build_indicator_context: CONTEXT BUILT")
            logger.info(f"Context length: {len(context_text)} chars (~{estimated_tokens} tokens)")
            logger.info(f"Number of lines: {len(lines)}")
            logger.info("-"*80)

            # Log key indicator values for debugging
            logger.info("KEY INDICATOR VALUES (last candle):")
            key_indicators = [
                ('%-supertrend_dir_14_3', 'SuperTrend Dir'),
                ('%-rsi_14', 'RSI'),
                ('%-adx_20', 'ADX'),
                ('%-di_plus_20', 'DI+'),
                ('%-di_minus_20', 'DI-'),
                ('%-cmf_20', 'CMF'),
                ('%-chop_14', 'Choppiness'),
                ('%-trend_strength', 'Trend Strength'),
                ('%-range_score', 'Range Score'),
                ('%-volatility_score', 'Volatility Score'),
                ('%-volatility_regime', 'Volatility Regime'),
                ('%-market_regime', 'Market Regime'),
                ('%-regime_confidence', 'Regime Confidence'),
                ('%-ema_diff', 'EMA Diff'),
            ]
            for col, name in key_indicators:
                if col in df.columns:
                    val = last.get(col)
                    if isinstance(val, (float, np.floating)) and not np.isnan(val):
                        logger.info(f"  {name}: {val:.4f}")
                    else:
                        logger.info(f"  {name}: {val}")
                else:
                    logger.info(f"  {name}: [MISSING from strategy]")
            logger.info("="*80)

        return context_text

    def _get_entry_recommendation(
        self,
        pair: str,
        indicators_context: str
    ) -> TradingRecommendation:
        """Get entry recommendation from LLM based on technical indicators."""
        target_profit = self.gpt_config.target_profit * 100
        target_duration = self.gpt_config.target_duration
        stop_loss = self.gpt_config.stoploss * 100

        # Always use FULL prompt for detailed analysis
        user_prompt = USER_PROMPT_ENTER_FULL.format(
            indicators=indicators_context,
            target_profit=target_profit,
            target_duration=target_duration,
            stop_loss=stop_loss,
        )

        if self.gpt_config.llm_trace:
            logger.info("\n" + "="*80)
            logger.info(f"LLM_TRACE ENTRY REQUEST for {pair}")
            logger.info("="*80)
            logger.info(f"Target Profit: {target_profit}% | Target Duration: {target_duration} candles | Stop Loss: {stop_loss}%")
            logger.info(f"User prompt length: {len(user_prompt)} chars (~{len(user_prompt)//4} tokens)")
            logger.info(f"Indicator context length: {len(indicators_context)} chars")

        return self._llm_recommendation(pair, user_prompt, "entry")

    def _get_exit_recommendation(
        self,
        pair: str,
        indicators_context: str,
        side: str,
        profit: float,
        duration: int
    ) -> TradingRecommendation:
        """Get exit recommendation from LLM based on technical indicators and position status."""
        target_profit = self.gpt_config.target_profit * 100
        target_duration = self.gpt_config.target_duration
        stop_loss = self.gpt_config.stoploss * 100

        # Always use FULL prompt for detailed analysis
        user_prompt = USER_PROMPT_EXIT_FULL.format(
            indicators=indicators_context,
            side=side.upper(),
            profit=profit * 100,
            duration=duration,
            target_profit=target_profit,
            target_duration=target_duration,
            stop_loss=stop_loss,
        )

        if self.gpt_config.llm_trace:
            logger.info("\n" + "="*80)
            logger.info(f"LLM_TRACE EXIT REQUEST for {pair}")
            logger.info("="*80)
            logger.info(f"Position: {side.upper()} | Profit: {profit*100:.2f}% | Duration: {duration} candles")
            logger.info(f"Target Profit: {target_profit}% | Target Duration: {target_duration} candles | Stop Loss: {stop_loss}%")
            logger.info(f"User prompt length: {len(user_prompt)} chars (~{len(user_prompt)//4} tokens)")
            logger.info(f"Indicator context length: {len(indicators_context)} chars")

        return self._llm_recommendation(pair, user_prompt, "exit", side=side.upper())

    def _llm_recommendation(
        self, pair: str, user_prompt: str, context: str, side: Optional[str] = None
    ) -> TradingRecommendation:
        """Call LLM and parse structured response."""
        messages = [self.cached_system_message, {"role": "user", "content": user_prompt}]

        if self.gpt_config.llm_trace:
            system_chars = len(self.cached_system_message.get("content", ""))
            user_chars = len(user_prompt)
            total_chars = system_chars + user_chars
            estimated_tokens = total_chars // 4

            logger.info("-"*80)
            logger.info(f"LLM_TRACE _llm_recommendation: SENDING TO LLM")
            logger.info("-"*80)
            logger.info(f"Pair: {pair} | Context: {context} | Side: {side}")
            logger.info(f"System prompt: {system_chars} chars (~{system_chars//4} tokens)")
            logger.info(f"User prompt: {user_chars} chars (~{user_chars//4} tokens)")
            logger.info(f"TOTAL: {total_chars} chars (~{estimated_tokens} tokens)")
            logger.info(f"LLM endpoint: {self._llm_openai_compat.chat_completions_url() if self._llm_openai_compat else 'instructor'}")
            logger.info(f"Model: {self.gpt_config.llm_model} | Temperature: {self.gpt_config.llm_temperature}")

            # Log full messages if within limit
            display_chars = min(total_chars, self.gpt_config.llm_trace_max_chars)
            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "unknown")
                if len(content) > display_chars:
                    logger.info(f"  [{role}] ({len(content)} chars): {content[:display_chars]}... (truncated)")
                else:
                    logger.info(f"  [{role}] ({len(content)} chars): {content}")

        def _sanitize_summary(text: str) -> str:
            t = (text or "").strip()
            # Remove non-printable control chars except common whitespace (\t, \n, \r)
            t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
            # Keep it short; model enforces max_length=300
            return t[:300]

        def _parse_action_from_text(raw_text: str) -> TradeAction:
            # Strip common markdown wrappers and normalize
            cleaned = (raw_text or "").strip()
            cleaned = re.sub(r"```[a-zA-Z0-9_-]*", "```", cleaned)
            cleaned = cleaned.replace("```", " ")
            cleaned = cleaned.replace("**", "")
            cleaned_u = cleaned.upper()

            # Prefer explicit full action tokens anywhere in the text
            m = re.search(r"\b(LONG_ENTER|SHORT_ENTER|LONG_EXIT|SHORT_EXIT|NEUTRAL|HOLD)\b", cleaned_u)
            if m:
                token = m.group(1)
                # HOLD is only valid for exit/position management; for entry treat HOLD as NEUTRAL.
                if token == "HOLD" and context == "entry":
                    return "NEUTRAL"
                return token

            first_line = cleaned_u.split("\n", 1)[0].strip()
            # Common shorthand
            if "NEUTRAL" in first_line or "WAIT" in first_line:
                return "NEUTRAL"
            if "HOLD" in first_line:
                return "HOLD" if context == "exit" else "NEUTRAL"

            if context == "entry":
                if "LONG" in first_line or "BUY" in first_line:
                    return "LONG_ENTER"
                if "SHORT" in first_line or "SELL" in first_line:
                    return "SHORT_ENTER"
                return "NEUTRAL"

            # context == "exit"
            if "EXIT" in first_line or "CLOSE" in first_line:
                if (side or "").upper() == "SHORT":
                    return "SHORT_EXIT"
                if (side or "").upper() == "LONG":
                    return "LONG_EXIT"
                return "NEUTRAL"

            if "LONG" in first_line:
                return "LONG_EXIT" if context == "exit" else "LONG_ENTER"
            if "SHORT" in first_line:
                return "SHORT_EXIT" if context == "exit" else "SHORT_ENTER"

            return "NEUTRAL"

        try:
            if self.gpt_config.llm_provider == "instructor":
                response = self._llm_instructor.chat.completions.create(
                    model=self.gpt_config.llm_model,
                    messages=messages,
                    response_model=TradingRecommendation,
                    temperature=self.gpt_config.llm_temperature,
                    max_retries=2,
                )
                result = response
            else:
                if self.gpt_config.llm_trace:
                    logger.info(f"LLM_TRACE sending request for {pair} to OpenAI-compatible endpoint")

                data = self._llm_openai_compat.complete_raw(messages)
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                # Try JSON anywhere in output
                try:
                    json_text = OpenAICompatLLM._extract_json(str(content))
                    raw_response = safe_json_loads(json_text, strict=False, sanitize=False)  # Explicitly False
                    result = TradingRecommendation(**raw_response)
                except Exception as e:
                    # 2) Fallback: plain text / markdown output
                    logger.warning(
                        f"LLM response is not valid JSON: %s for %s (%s). Using text fallback. Raw (first {self._llm_openai_compat.max_tokens}): %s",
                        e,
                        pair,
                        context,
                        str(content).strip()[:self._llm_openai_compat.max_tokens],
                    )
                    action = _parse_action_from_text(str(content))
                    result = TradingRecommendation(
                        action=action,
                        summary=_sanitize_summary(str(content)),
                    )

            if self.gpt_config.llm_trace:
                logger.info("-"*80)
                logger.info(f"LLM_TRACE _llm_recommendation: RESPONSE RECEIVED")
                logger.info(
                    f"Raw (first {self._llm_openai_compat.max_tokens}) for %s (%s): %s",
                    pair,
                    context,
                    str(content).strip()[:self._llm_openai_compat.max_tokens],
                )
                logger.info("-"*80)
                logger.info(f"Action: {result.action}")
                logger.info(f"Summary sanitized: {result.summary}")
                logger.info("="*80 + "\n")

            return result

        except Exception as e:
            logger.error(f"LLM recommendation failed for {pair}: {e}", exc_info=True)
            # Return neutral recommendation on error
            return TradingRecommendation(
                action="NEUTRAL",
                summary=f"Error: {str(e)[:200]}"
            )

    def predict(
            self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ):
        """
        Call LLM on each new candle to get fresh recommendations based on latest indicators.
        """
        pair = dk.pair

        logger.debug(f"GPTTrader predict for {pair} on new candle")

        # Build context from latest candle indicators
        indicator_context = self._build_indicator_context(unfiltered_df)

        # Get current position state
        side, profit, duration = self.get_state_info(pair)

        # Call LLM with fresh context
        if not side:
            logger.debug(f"No position for {pair}, getting entry recommendation")
            result = self._get_entry_recommendation(pair, indicator_context)
        else:
            logger.debug(f"Position exists for {pair}: {side}, profit: {profit:.2%}, duration: {duration} candles")
            result = self._get_exit_recommendation(
                pair, indicator_context, 
                side=side, profit=profit, duration=duration
            )

        if self.gpt_config.llm_trace:
            logger.info(
                "LLM_TRACE final action for %s: %s summary=%s",
                pair,
                result.action,
                result.summary,
            )

        # Convert LLM recommendation to strategy signals
        trade_action: TradeAction = result.action

        results = {
            "expert_long_enter": bool(trade_action == 'LONG_ENTER'),
            "expert_long_exit": bool(trade_action == 'LONG_EXIT'),
            "expert_short_enter": bool(trade_action == 'SHORT_ENTER'),
            "expert_short_exit": bool(trade_action == 'SHORT_EXIT'),
            "expert_neutral": bool(trade_action == 'NEUTRAL'),
            "expert_opinion": str(result.summary)
        }

        dk.data['extra_returns_per_train'] = results
        logger.debug(f"prediction dump: {dk.data['extra_returns_per_train']}")

        # Return zeros for classical predictions (not used by this strategy)
        if "&-empty" not in dk.label_list:
            dk.label_list.append("&-empty")
        self._ensure_label_stats(dk, labels=["&-empty"])
        zeros = len(unfiltered_df.index)
        return DataFrame(np.zeros(zeros), columns=["&-empty"]), np.ones(zeros)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        return None

    def get_state_info(self, pair: str) -> Tuple[MarketSide, float, int]:
        """
        State info during dry/live (not backtesting) which is fed back
        into the model.
        :param pair: str = COIN/STAKE to get the environment information for
        :return:
        :market_side: MarketSide = representing short, long, or neutral for
            pair
        :current_profit: float = unrealized profit of the current trade
        :trade_duration: int = the number of candles that the trade has
            been open for
        """

        exchange = self.data_provider._exchange

        if not exchange:
            logger.error('No exchange available.')
            return None, 0, 0

        open_trades = Trade.get_trades_proxy(is_open=True)

        pair_trades = [trade for trade in open_trades if trade.pair == pair]

        if len(pair_trades) == 0:
            return None, 0, 0

        trade = pair_trades[0]

        current_rate = exchange.get_rate(pair, refresh=False, side="exit", is_short=trade.is_short)

        now = datetime.now(timezone.utc).timestamp()
        trade_duration = int((now - trade.open_date_utc.timestamp()) / self.base_tf_seconds)
        current_profit = trade.calc_profit_ratio(current_rate)
        market_side: MarketSide = "SHORT" if trade.is_short else "LONG"

        return market_side, current_profit, int(trade_duration)


def date_str(date: datetime) -> str:
    return date.strftime('%Y-%m-%d %H:%M')

def now_str():
    return date_str(datetime.now(timezone.utc))

