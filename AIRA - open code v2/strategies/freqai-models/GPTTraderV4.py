import requests

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

# deps
from pydantic import BaseModel, Field, constr, computed_field
from instructor import Instructor
import instructor
from asknews_sdk import AskNewsSDK
from asknews_sdk.dto import SearchResponseDictItem
from asknews_sdk.dto.chat import WebSearchResult


"""
The following FreqAI model is released to the community of the non-profit FreqAI open-source project.

GPTTraderV4 requires News API access and X (twitter) API access. Both are made available to sponsors of
the FreqAI project. Details for becoming a sponsor can be found here: https://github.com/sponsors/robcaulk

If you find the FreqAI project useful in general, please consider supporting it by becoming a sponsor.
We use sponsor money to help stimulate new features and to pay for running these public
experiments, with a an objective of helping the community make smarter choices in their
ML journey.

This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
returns. The goal is to demonstrate gratitude to people who support the project and to
help them find a good starting point for their own creativity.

If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

## 🏃 Running GPTTrader V4

### Installation
```bash
# Install dependencies
pip install instructor asknews-sdk pydantic

# Copy files
cp GPTStrategyV4.py user_data/strategies/
cp GPTTraderV4.py user_data/freqaimodels/
```

### Dry Run Mode
```bash
freqtrade trade \
  --config user_data/config_gpttrader_v4.json \
  --strategy GPTStrategyV4 \
  --freqaimodel GPTTraderV4
```

### Live Trading
```bash
# Set dry_run to false in config
# Add exchange API keys
freqtrade trade \
  --config user_data/config_gpttrader_v4.json \
  --strategy GPTStrategyV4 \
  --freqaimodel GPTTraderV4
```
"""


SYSTEM_PROMPT = """
You are an expert crypto trading advisor specializing in news-based sentiment analysis and risk-aware position management.

Your role:
1. Analyze news sentiment with precision, considering market impact and timing
2. Assess news "heat" based on recency and importance
3. Make trading decisions aligned with strict risk management targets
4. Balance opportunity with capital preservation

Data Source Analysis Framework:

**News Articles (identified by #### News#X headers)**:
Format: Markdown list with Title, Published, Source, Classification, External Analyst Sentiment, etc.

**CRITICAL: External Analyst Sentiment is MINORITY INPUT (20-30% weight)**
The pre-labeled "Sentiment" field represents an external analyst's opinion, NOT ground truth.
Your own analysis of the Summary content should carry 70-80% weight.

**Sentiment Calculation Formula**:
```
Final_Sentiment = (Own_Analysis × 0.7) + (External_Sentiment × 0.3) 
                  × Content_Weight × Voice_Multiplier
```

**Step 1: Own Analysis (70% weight)**:
Analyze the Summary text directly for:
- Factual content vs speculation
- Actual events vs predictions
- Positive/negative language and implications
- Market impact potential

**Step 2: External Analyst Sentiment (30% weight)**:
- Treat the pre-labeled sentiment as ONE analyst's opinion
- Can be biased, promotional, or incorrect
- Use as supplementary input, not primary signal

**Step 3: Content Type Weighting**:
- **REAL ACTIONS** (1.0x): Actual trades, regulatory decisions, hacks, launches, partnerships
- **ANALYTICAL/PREDICTIVE** (0.5-0.7x): Price predictions, technical analysis, expert opinions
- **SPECULATIVE/PROMOTIONAL** (0.3x): "Could be", "might", "potential", ICO promotions

**Step 4: Reporting Voice Adjustment**:
- **Investigative**: Real reporting of actual events (1.0x)
- **Analytical**: Analysis and predictions (0.6x) 
- **Explanatory**: Educational or opinion pieces (0.4x)

**Key Phrases that REDUCE sentiment**:
- "experts predict", "analysts suggest", "could", "might", "potential"
- "breakout pattern", "technical analysis", "price target"
- ICO/presale promotions, comparison to past performance
- Conditional statements ("if it replicates", "could grow to")

**Tweets/Social Media (identified by #### [tweet#X] headers)**:
Format: Markdown list with Title, Published, Source, Content, and Metrics line
- Parse Metrics line: "retweets=X, likes=X, replies=X, quotes=X"
  * "Retweets: X" - viral amplification indicator
  * "Likes: X" - positive sentiment gauge  
  * "Replies: X" - discussion/controversy level
  * "Quotes: X" - sharing with commentary
- **Zero Engagement Rule**: If all metrics = 0, then Impact = 0 and Sentiment weight = 0
- **High Engagement Multiplier**: 
  * 1-10 total engagement: Low impact (0.1-0.3)
  * 11-100 total engagement: Medium impact (0.4-0.7)  
  * 100+ total engagement: High impact (0.8-1.0)
- Analyze content sentiment from Key Points text
- Apply social media volatility factor (higher uncertainty than news)

**Heat Score Calculation (0.0 to 1.0)**:
Combines recency and source impact:
- **Recency Factor**: 1.0 for <5min, 0.8 for <30min, 0.6 for <2hrs, 0.4 for <12hrs, 0.2 for <24hrs, 0.1 for older
- **Impact Factor**: 
  * News: Based on Classification + Reporting voice + Source credibility
  * Tweets: **ENGAGEMENT-DRIVEN IMPACT**:
    - Total engagement = Retweets + Likes + Replies + Quotes
    - If total engagement = 0: Impact = 0 (regardless of content)
    - If total engagement > 0: Impact = min(1.0, total_engagement / 100)
    - Account authority modifier (if determinable from source)
- **Heat = (Recency × 0.4) + (Impact × 0.6)**
- **Special Rule**: For tweets with zero engagement, Heat automatically = 0

**Sentiment Scoring Guidelines (AFTER applying content/voice weights)**:

**REAL MARKET ACTIONS** (use full sentiment range):
- 0.7 to 1.0: Major bullish catalysts (ETF approvals, institutional purchases, major exchange listings)
- 0.3 to 0.7: Positive developments (actual partnerships signed, network upgrades deployed, volume increases)
- -0.3 to 0.3: Neutral or mixed signals (routine updates, conflicting real events)
- -0.7 to -0.3: Negative developments (security breaches, regulatory actions, liquidations)
- -1.0 to -0.7: Major bearish catalysts (exchange hacks, country bans, protocol failures)

**ANALYTICAL/PREDICTIVE CONTENT** (cap at ±0.7 after weighting):
- Price predictions, technical patterns → Max sentiment: 0.5-0.7
- "Experts predict" statements → Max sentiment: 0.4-0.6
- Comparison to historical patterns → Max sentiment: 0.3-0.5
- ICO/presale promotions → Max sentiment: 0.2-0.4

Example: News about "experts predict 50x return" with Analytical voice:
- Pre-labeled sentiment: 1.0
- Content weight: 0.5 (predictive)
- Voice multiplier: 0.6 (analytical)
- Final sentiment: 1.0 × 0.5 × 0.6 = 0.3

Entry Decision Framework:
- LONG_ENTER: Average sentiment > 0.4 with high-heat positive news (heat > 0.6) and acceptable risk/reward
- SHORT_ENTER: Average sentiment < -0.4 with high-heat negative news (heat > 0.6) and acceptable risk/reward
- NEUTRAL: Low heat scores, mixed signals, insufficient conviction, or risk/reward doesn't justify entry

Exit Decision Framework:
- LONG_EXIT: Exit long when ANY of these occur:
  * Target reached or stop-loss approaching
  * Average sentiment turns negative (< -0.2) - exit immediately
  * High-heat bearish news emerges (heat > 0.7 with negative sentiment)
- SHORT_EXIT: Exit short when ANY of these occur:
  * Target reached or stop-loss approaching  
  * Average sentiment turns positive (> 0.2) - exit immediately
  * High-heat bullish news emerges (heat > 0.7 with positive sentiment)
- NEUTRAL (Hold): Continue holding ONLY when sentiment supports position and no exit triggers present
"""

USER_PROMPT_ENTER_POSITION = """
# Trading Entry Analysis

## Task
Analyze crypto market data and provide trading recommendations based on news sentiment and social media activity.

## Market Data
{news}
{tweets}

## Trading Parameters
- **Target Profit**: {target_profit}%
- **Maximum Duration**: {target_duration} candles  
- **Stop Loss**: {stop_loss}%
- **Current Date**: {timestamp}

## Analysis Requirements

For each data source item (identified by `News#X` or `Tweet#X` headers), evaluate:

**CRITICAL**: Distinguish between REAL ACTIONS and PREDICTIONS/OPINIONS
- REAL: "Bitcoin ETF approved", "Exchange hacked", "Company buys $100M BTC"
- OPINION: "Experts predict", "Could reach $X", "Breakout pattern suggests"

### 1. Sentiment Score (-1 to 1)
- **News articles** `News#X`: Perform YOUR OWN analysis, don't rely on external sentiment
  - Analyze Summary content directly (70% weight)
  - External Analyst Sentiment is minority input (30% weight) - can be biased/wrong
  - Identify if content describes REAL ACTIONS vs PREDICTIONS/OPINIONS
  - Apply Reporting voice multiplier (Investigative=1.0, Analytical=0.6, Explanatory=0.4)
  - Reduce sentiment for speculative keywords ("could", "might", "experts predict")
  - Cap analytical/predictive content at ±0.7 max
- **Tweets** `Tweet#X`: Analyze Content field and weight by Metrics line
  - If all metrics = 0: Sentiment weight = 0 (no market impact)
  - If total engagement > 0: Analyze content sentiment and weight by engagement level

### 2. Heat Score (0 to 1)
Combined recency and impact assessment:

**Recency Factor**: 
- 1.0 for <5min
- 0.8 for <30min
- 0.6 for <2hrs
- 0.4 for <12hrs
- 0.2 for <24hrs
- 0.1 for older

**Impact Factor**:
- **News**: Classification + Reporting voice + Source credibility
- **Tweets**: Total engagement (retweets + likes + replies + quotes)
  - If total = 0: Impact = 0, Heat = 0
  - If total > 0: Impact = min(1.0, total / 100)

**Formula**: Heat = (Recency × 0.4) + (Impact × 0.6)

### 3. Entry Recommendation
Choose: `LONG_ENTER`, `SHORT_ENTER`, or `NEUTRAL`

### 4. Rationale
Provide concise justification linking sentiment and heat analysis to recommended action.

## Decision Criteria

- **LONG_ENTER**: Average sentiment > 0.4 with high-heat positive news/tweets (heat > 0.6)
- **SHORT_ENTER**: Average sentiment < -0.4 with high-heat negative news/tweets (heat > 0.6)  
- **NEUTRAL**: Low heat scores, mixed signals, insufficient conviction for entry

### Key Considerations
- Prioritize high-heat sources with strong sentiment alignment
- Consider source credibility (news vs social media weight)
- Assess risk/reward given current market conditions

### Example Sentiment Adjustments

1. **"Experts predict $0.009 coin could provide 50x return"** + Analytical voice + External Sentiment: 1.0
   - Your analysis: Speculative ICO promotion = 0.2
   - External sentiment: 1.0 (overly optimistic)
   - Combined: (0.2 × 0.7) + (1.0 × 0.3) = 0.44
   - After voice/content: 0.44 × 0.6 × 0.3 = **0.08** (minimal impact)
   
2. **"Grayscale files for Cardano ETF"** + Investigative voice + External Sentiment: 1.0
   - Your analysis: Real regulatory filing = 0.9
   - External sentiment: 1.0 (agrees)
   - Combined: (0.9 × 0.7) + (1.0 × 0.3) = 0.93
   - After voice/content: 0.93 × 1.0 × 1.0 = **0.93** (high impact)
   
3. **"Technical pattern points to $1.10 target"** + Analytical voice + External Sentiment: 1.0
   - Your analysis: Pure technical opinion = 0.4
   - External sentiment: 1.0 (biased bullish)
   - Combined: (0.4 × 0.7) + (1.0 × 0.3) = 0.58
   - After voice/content: 0.58 × 0.6 × 0.7 = **0.24** (low impact)
"""

USER_PROMPT_EXIT_POSITION = """
# Position Exit Analysis

## Task
Evaluate current position exit strategy based on latest market data and position performance.

## Market Data
{news}
{tweets}

## Current Position Status
- **Direction**: {side}
- **Current P&L**: {profit}% (target: {target_profit}%)
- **Time in Position**: {duration}/{target_duration} candles
- **Current Date**: {timestamp}

## Trading Targets
- **Target Profit**: {target_profit}%
- **Maximum Duration**: {target_duration} candles  
- **Stop Loss**: {stop_loss}%

## Analysis Requirements

For each data source item (identified by `News#X` or `Tweet#X` headers), evaluate:

**CRITICAL**: Distinguish between REAL ACTIONS and PREDICTIONS/OPINIONS
- REAL: "Bitcoin ETF approved", "Exchange hacked", "Company buys $100M BTC"
- OPINION: "Experts predict", "Could reach $X", "Breakout pattern suggests"

### 1. Sentiment Score (-1 to 1)
- **News articles** `News#X`: Perform YOUR OWN analysis, don't rely on external sentiment
  - Analyze Summary content directly (70% weight)
  - External Analyst Sentiment is minority input (30% weight) - can be biased/wrong
  - Identify if content describes REAL ACTIONS vs PREDICTIONS/OPINIONS
  - Apply Reporting voice multiplier (Investigative=1.0, Analytical=0.6, Explanatory=0.4)
  - Reduce sentiment for speculative keywords ("could", "might", "experts predict")
  - Cap analytical/predictive content at ±0.7 max
- **Tweets** `Tweet#X`: Analyze Content field and weight by Metrics line
  - If all metrics = 0: Sentiment weight = 0 (no market impact)
  - If total engagement > 0: Analyze content sentiment and weight by engagement level

### 2. Heat Score (0 to 1)
Combined recency and impact assessment:

**Recency Factor**: 
- 1.0 for <5min
- 0.8 for <30min
- 0.6 for <2hrs
- 0.4 for <12hrs
- 0.2 for <24hrs
- 0.1 for older

**Impact Factor**:
- **News**: Classification + Reporting voice + Source credibility
- **Tweets**: Total engagement (retweets + likes + replies + quotes)
  - If total = 0: Impact = 0, Heat = 0
  - If total > 0: Impact = min(1.0, total / 100)

**Formula**: Heat = (Recency × 0.4) + (Impact × 0.6)

### 3. Exit Recommendation
Choose: `LONG_EXIT`, `SHORT_EXIT`, or `NEUTRAL` (hold)

### 4. Rationale
Justify recommendation based on sentiment shift and risk management.

## Exit Decision Framework

### IMMEDIATE EXIT CONDITIONS (exit regardless of heat)
- **LONG positions**: Exit if average sentiment < -0.2
- **SHORT positions**: Exit if average sentiment > 0.2  
- Any sentiment reversal against position direction

### ADDITIONAL EXIT TRIGGERS
- High-heat news opposing position (heat > 0.7 with opposite sentiment)
- Position approaching stop loss or target profit
- Maximum duration reached
- Risk management signals from market data

### HOLD CONDITIONS
- Sentiment supports current position
- No immediate exit triggers present
- Position within acceptable risk parameters

## ⚠️ Critical Note
If sentiment has turned against the position, EXIT IMMEDIATELY regardless of news heat scores. 
Capital preservation is paramount - protect against further losses when market sentiment shifts.
"""

type MarketSide = Optional[Literal["LONG", "SHORT"]]
type TradeAction = Literal["LONG_ENTER", "SHORT_ENTER", "NEUTRAL", "LONG_EXIT", "SHORT_EXIT"]


class TradingRecommendation(BaseModel):
    """Structured trading recommendation based on news sentiment analysis"""

    sentiments: List[float] = Field(
        description="Sentiment score (-1 to 1) for each news item in order"
    )

    heat_scores: List[float] = Field(
        description="Heat score (0 to 1) for each news item based on recency and importance"
    )

    action: TradeAction = Field(
        description="Recommended trading action based on sentiment and heat analysis"
    )

    summary: str = Field(
        description="Concise justification (max 2 sentences) linking sentiment and heat analysis to recommended action",
        max_length=300
    )

    @computed_field
    @property
    def sentiment(self) -> float:
        """Automatically computed average of all sentiment scores"""
        if not self.sentiments:
            return 0.0
        return sum(self.sentiments) / len(self.sentiments)

    @computed_field
    @property
    def heat(self) -> float:
        """Automatically computed heat of all heat scores"""
        if not self.heat_scores:
            return 0.0
        return sum(self.heat_scores) / len(self.heat_scores)


class GPTTraderConfig(BaseModel):
    target_profit: float = 0.03
    target_duration: int = 100
    stoploss: float = 0.04
    llm_model: constr(min_length=1) = "gpt-3.5-turbo"
    llm_api_key: constr(min_length=1)  # Required
    ask_news_client_id: constr(min_length=1)  # Required
    ask_news_client_secret: constr(min_length=1)  # Required
    news_hours: int = 6
    tweet_minutes: int = 15
    news_count: int = 5
    use_news: bool = True
    use_twitter: bool = True


logger = logging.getLogger(__name__)


class GPTTraderV4(IFreqaiModel):
    """
    Base model for letting an LLM take full control of your bot...
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.gpt_config = GPTTraderConfig(**self.freqai_info.get('GPTTrader', {}))

        self.client: Instructor = instructor.from_provider(model=self.gpt_config.llm_model,
                                                           api_key=self.gpt_config.llm_api_key)

        self.cached_system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }

        self.last_news: List[SearchResponseDictItem] = []
        self.last_x_posts: List[WebSearchResult] = []

        self.ask = AskNewsSDK(
            client_id=self.gpt_config.ask_news_client_id,
            client_secret=self.gpt_config.ask_news_client_secret,
            scopes=["news", "chat"]
        )

        logging.info("GPTTrader initialized")


    def _prepare_latest_twitter(self, project_name: str, symbol: str) -> str:
        """
        Fetches and prepares the latest posts on twittter related to the given trading pair's
        project.
        Args:
            project_name (str): The name of the cryptocurrency project.
            symbol (str): The trading symbol (e.g., 'btc').

        Returns:
            str: News context.
        """

        # optimize
        if len(self.last_x_posts) > 0:
            last_x_datetime = tweet_date_str(self.last_x_posts[0].published)
        else:
            last_x_datetime = datetime.now(timezone.utc) - timedelta(minutes=self.gpt_config.tweet_minutes)

        since = f"since:{last_x_datetime.strftime('%Y-%m-%d_%H:%M:%S_UTC')}"
        domains= ["https://x.com"]
        result = self.ask.chat.live_web_search(
            queries=[f"({project_name} OR {symbol} OR {symbol}) AND (crypto OR cryptocurrency OR blockchain OR project) {since}"],
            domains=domains
        ).as_dicts

        logging.info(f"found {len(result)} tweets for {project_name}.")

        # merge with latest to keep timeline
        self.last_x_posts = (result + self.last_x_posts)[:self.gpt_config.news_count]

        has_fresh_x = len(result) > 0

        if has_fresh_x:
            x_context = build_web_context(self.last_x_posts, domains)

            return x_context

        return ""


    def _prepare_latest_news(self, project_name: str, symbol: str) -> str:
        """
        Fetches and prepares the latest news articles related to the given trading pair's project.
        If no fresh news since last fetch returns ""
        Args:
            project_name (str): The name of the cryptocurrency project.
            symbol (str): The trading pair (e.g., 'BTC').

        Returns:
            str: News context.
        """

        logging.debug(f"Fetch related news for {symbol}: {project_name}")

        last_pub_timestamp = None

        # optimize
        if len(self.last_news) > 0:
            last_pub_timestamp = int(self.last_news[0].pub_date.timestamp())  # type: ignore


        result = self.ask.news.search_news(
            query=f"cryptocurrency market {project_name} {symbol} {symbol}",
            string_guarantee=[project_name, symbol, symbol],
            string_guarantee_op="OR",
            n_articles=self.gpt_config.news_count,
            hours_back=self.gpt_config.news_hours if not last_pub_timestamp else None,
            start_timestamp=last_pub_timestamp,
            time_filter="pub_date",
            return_type="dicts"
        ).as_dicts

        logging.info(f"found {len(result)} articles for {project_name}.")

        # merge with latest to keep timeline
        self.last_news = (result + self.last_news)[:self.gpt_config.news_count]

        has_fresh_news = len(result) > 0

        if has_fresh_news:
            news_context = build_news_context(self.last_news)

            return news_context

        return ""

    def _get_entry_recommendation(self, pair: str, news_context: str, x_context: str) -> TradingRecommendation:
        """Get trading entry recommendation based on news sentiment"""

        recommendation = self.client.chat.completions.create(
            response_model=TradingRecommendation,
            messages=[
                self.cached_system_message,
                {"role": "user", "content": USER_PROMPT_ENTER_POSITION.format(
                    news=news_context,
                    tweets=x_context,
                    target_profit=self.gpt_config.target_profit * 100,
                    target_duration=self.gpt_config.target_duration,
                    stop_loss=self.gpt_config.stoploss * 100,
                    timestamp=now_str()
                )}
            ],
            max_retries=2
        )

        logger.info(
            f"Expert opinion for {pair} -> {recommendation.action}(sentiment: {recommendation.sentiment:.2f} "
            f"heat: {recommendation.heat:.2f})\n"
            f"Motivation: {recommendation.summary}")

        return recommendation

    def _get_exit_recommendation(self, pair: str, news_context: str, x_context: str, side: str, profit: float,
                                  duration: int) -> TradingRecommendation:
        """Get position exit recommendation based on news and position status"""

        recommendation = self.client.chat.completions.create(
            response_model=TradingRecommendation,
            messages=[
                self.cached_system_message,
                {"role": "user", "content": USER_PROMPT_EXIT_POSITION.format(
                    news=news_context,
                    tweets=x_context,
                    side=side,
                    profit=profit * 100,
                    duration=duration,
                    target_profit=self.gpt_config.target_profit * 100,
                    target_duration=self.gpt_config.target_duration,
                    stop_loss=self.gpt_config.stoploss * 100,
                    timestamp=now_str()
                )}
            ],
            max_retries=2
        )

        logger.info(f"Expert opinion for {side} {pair} position: "
                    f"{recommendation.action}(sentiment: {recommendation.sentiment} heat: {recommendation.heat})\n"
                    f"Motivation: {recommendation.summary}")
        return recommendation

    def _clean_kitchen(self, dk: FreqaiDataKitchen):
        dk.data_dictionary["train_features"] = DataFrame(np.zeros(1))
        dk.data_dictionary["train_dates"] = DataFrame(np.zeros(1))

        zero_labels = {label: 0 for label in dk.label_list}

        dk.data.setdefault("labels_mean", zero_labels)
        dk.data.setdefault("labels_std", zero_labels)

    def train(
            self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Build a custom prompt asking to determine news sentiment and take an action
        based on current coin position.
        """
        project_name = fetch_project_name(pair)
        symbol = pair.split("/")[0].lower()

        logger.info(f"GPTTrader training for {pair} ({project_name})")

        news_context = self._prepare_latest_news(project_name, symbol) if self.gpt_config.use_news else ""
        x_context = self._prepare_latest_twitter(project_name, symbol) if self.gpt_config.use_twitter else ""

        # if no (fresh) news do nt perform LLM calls
        if not news_context and not x_context:
            logger.info(f"No fresh news or Twitter posts for {symbol}, skipping LLM call")
            result = TradingRecommendation(sentiments=[], heat_scores=[], action='NEUTRAL', summary="")
        else:
            side, profit, duration = self.get_state_info(pair)

            if not side:
                logger.info(f"No position for {pair}, getting entry recommendation")
                result = self._get_entry_recommendation(pair, news_context, x_context)
            else:
                logger.info(f"Position exists for {pair}: {side}, profit: {profit:.2%}, duration: {duration} candles")
                result = self._get_exit_recommendation(pair, news_context, x_context, side=side, profit=profit, duration=duration)

        self._clean_kitchen(dk)

        return result.model_dump()

    def predict(
            self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ):
        """
        Get the cached predictions and feed them back to the strategy
        """
        neutral_threshold = 0.3
        sentiments = self.model['sentiments']
        trade_action: TradeAction = self.model['action']
        results = {
            "sentiment_yes": len([s for s in sentiments if s > neutral_threshold]),
            "sentiment_no": len([s for s in sentiments if s < -neutral_threshold]),
            "sentiment_unknown": len([s for s in sentiments if abs(s) <= neutral_threshold]),
            "sentiment": self.model['sentiment'],
            "heat": self.model.get('heat', 0),
            "expert_long_enter": trade_action == 'LONG_ENTER',
            "expert_long_exit": trade_action == 'LONG_EXIT',
            "expert_short_enter": trade_action == 'SHORT_ENTER',
            "expert_short_exit": trade_action == 'SHORT_EXIT',
            "expert_neutral": trade_action == 'NEUTRAL',
            "expert_opinion": self.model['summary']
        }

        dk.data['extra_returns_per_train'] = results
        logger.debug(f"prediction dump: {dk.data['extra_returns_per_train']}")

        # we will simply return zeros for the classical predictions
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


def retry(times=3, delay=1, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            raise last_exc

        return wrapper

    return decorator


@functools.lru_cache()
@retry(times=5, delay=10)
def fetch_project_name(pair: str):
    symbol = pair.split("/")[0].lower()  # lowercase is required by CoinGecko
    url = f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&symbols={symbol}'
    response = requests.get(url)

    time.sleep(2)  # Extra sleep to avoid rate-limit block

    if response.status_code == 200:
        data = response.json()
        if data and isinstance(data, list) and "name" in data[0]:
            return data[0]["name"]
        else:
            raise Exception(f"Unexpected response format or empty result for {symbol}.")
    else:
        raise Exception(f"Error fetching project name for {symbol}. Status code: {response.status_code}")


def date_str(date: datetime) -> str:
    return date.strftime('%Y-%m-%d %H:%M')

def now_str():
    return date_str(datetime.now(timezone.utc))

def tweet_date_str(date: str) -> datetime:
    """
    Convert tweet date string to datetime object.
    Example: "January 01 2023, 12:00"
    """
    return datetime.strptime(date, "%B %d %Y, %H:%M").replace(tzinfo=timezone.utc)


def minutes_since(pub_date: datetime) -> int:
    now = datetime.now(timezone.utc)
    diff = now - pub_date
    return int(diff.total_seconds() // 60)  # total minutes


def build_news_context(search_response: List[SearchResponseDictItem]) -> str:
    """
    Build news context with markdown formatting
    """
    context = "## News Articles\n\n"
    for i, a in enumerate(search_response):
        context += f"#### News#{i}\n"
        context += f"- Title: {a.title}\n"
        context += f"- Published: {date_str(a.pub_date)} ({minutes_since(a.pub_date)} min ago)\n"
        context += f"- Source: {a.source_id}\n"
        context += f"- Classification: {a.classification}\n"
        context += f"- External Analyst Sentiment: {a.sentiment}\n"
        context += f"- Reporting voice: {a.reporting_voice}\n"
        context += f"- Continent: {a.continent}\n"
        context += f"- Summary:\n"
        context += f"  {a.summary}\n\n"
    return context


def build_web_context(
    search_response: List[WebSearchResult], domains: list[str]
) -> str:
    """
    Build tweet/web context with markdown formatting and engagement metrics
    """
    context = f"## Tweets/Social Media\n"
    context += f"*Sources: {', '.join(domains)}*\n\n"
    
    for i, a in enumerate(search_response):
        tweet_date = tweet_date_str(a.published)
        context += f"#### Tweet#{i}\n"
        context += f"- Title: {a.title}\n"
        context += f"- Published: {date_str(tweet_date)} ({minutes_since(tweet_date)} min ago)\n"
        context += f"- Source: {a.source}\n"
        
        # Extract engagement metrics from key_points
        metrics = {"retweets": 0, "likes": 0, "replies": 0, "quotes": 0}
        content_lines = []
        
        for point in a.key_points:
            if "Retweets:" in point:
                metrics["retweets"] = int(point.split(":")[-1].strip())
            elif "Likes:" in point:
                metrics["likes"] = int(point.split(":")[-1].strip())
            elif "Replies:" in point:
                metrics["replies"] = int(point.split(":")[-1].strip())
            elif "Quotes:" in point:
                metrics["quotes"] = int(point.split(":")[-1].strip())
            elif "Content:" in point:
                content_lines.append(point.replace("Content:", "").strip())
        
        context += f"- Content:\n"
        if content_lines:
            context += f"  {' '.join(content_lines)}\n"
        else:
            context += f"  {' '.join(a.key_points)}\n"
        
        context += f"- Metrics: retweets={metrics['retweets']}, likes={metrics['likes']}, replies={metrics['replies']}, quotes={metrics['quotes']}\n\n"
    
    return context