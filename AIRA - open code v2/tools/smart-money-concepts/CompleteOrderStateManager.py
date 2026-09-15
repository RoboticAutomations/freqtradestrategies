from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import requests

logger = logging.getLogger(__name__)

class TradingState(Enum):
    NO_SIGNAL = "no_signal"
    PENDING_LIMIT = "pending_limit"
    ACTIVE_POSITION = "active_position"
    SIGNAL_EXPIRED = "signal_expired"

class CompleteOrderStateManager:
    def __init__(self):
        self.last_signals = {}  # Track last detected signals
        self.current_states = {}  # Track current trading states
        self.pending_orders = {}  # Track pending limit orders
        self.active_positions = {}  # Track active positions
        
    def update_last_signal(self, pair: str, signal_data: dict):
        """Update the last detected entry signal"""
        self.last_signals[pair] = {
            'signal': signal_data,
            'detected_time': datetime.now(),
            'used': False,  # Whether this signal was used for entry
            'valid': True   # Whether signal is still valid
        }
        
    def update_trading_state(self, pair: str, state: TradingState, state_data: dict = None):
        """Update current trading state"""
        self.current_states[pair] = {
            'state': state,
            'data': state_data or {},
            'updated_time': datetime.now()
        }
        
    def get_comprehensive_decision(self, pair: str, current_market_data: dict):
        """Get comprehensive decision based on current state"""
        
        current_state = self.current_states.get(pair, {}).get('state', TradingState.NO_SIGNAL)
        last_signal = self.last_signals.get(pair)
        
        # Prepare decision data
        decision_data = {
            'pair': pair,
            'current_state': current_state.value,
            'last_signal': last_signal,
            'current_market_data': current_market_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get decision from DeepSeek
        return self.analyze_complete_state_with_deepseek(decision_data)
    
    def analyze_complete_state_with_deepseek(self, decision_data: dict):
        """Comprehensive DeepSeek analysis for ICT methodology"""
        
        current_state = decision_data['current_state']
        last_signal = decision_data['last_signal']
        pair = decision_data['pair']
        
        # Create state-specific prompt
        if current_state == TradingState.NO_SIGNAL.value:
            scenario_text = "NO RECENT SIGNALS"
            possible_decisions = "ENTER_NEW (create new ICT signal) or WAIT (wait for liquidity sweep + MSS)"
            
        elif current_state == TradingState.PENDING_LIMIT.value:
            scenario_text = "PENDING LIMIT ORDER"
            possible_decisions = "CANCEL (cancel order) or KEEP_WAITING (keep order)"
            
        elif current_state == TradingState.ACTIVE_POSITION.value:
            scenario_text = "ACTIVE POSITION"
            possible_decisions = "HOLD (hold position), CLOSE_NOW (close now), or PARTIAL_EXIT (partial exit)"
            
        else:  # SIGNAL_EXPIRED
            scenario_text = "SIGNAL EXPIRED"
            possible_decisions = "ENTER_NEW (create new ICT signal) or WAIT (wait for new setup)"
        
        # Build comprehensive prompt for ICT methodology
        prompt = f"""
        You are an **ICT (Inner Circle Trader) methodology expert**. Analyze institutional order flow and provide optimal decisions based on liquidity sweeps and market structure shifts.

        === CURRENT SITUATION ===
        Pair: {pair}
        Status: {scenario_text}
        Time: {decision_data['timestamp']}

        === LATEST SIGNAL INFORMATION ===
        """
        
        if last_signal:
            signal_age = (datetime.now() - last_signal['detected_time']).total_seconds() / 60
            prompt += f"""
        Last signal: {last_signal['signal'].get('action', 'N/A')}
        Entry: ${last_signal['signal'].get('entry', 'N/A')}
        SL: ${last_signal['signal'].get('sl', 'N/A')}
        TP: ${last_signal['signal'].get('tp', 'N/A')}
        Signal age: {signal_age:.1f} minutes
        Used: {'Yes' if last_signal['used'] else 'No'}
        """
        else:
            prompt += "\nNo recent signals"
        
        prompt += f"""

        === CURRENT MARKET DATA ===
        Current price: ${decision_data['current_market_data'].get('current_price', 'N/A')}
        Latest OHLCV: {json.dumps(decision_data['current_market_data'].get('timeframes', []), indent=2)}

        === ICT ANALYSIS REQUIREMENTS ===
        1. **Liquidity Sweep Analysis**: 
           - Recent highs/lows taken out (liquidity grab)
           - External/Internal liquidity sweeps
           - Relative Equal Highs/Lows (REH/REL) swept
        
        2. **Market Structure Shift (MSS)**:
           - Break of structure (BOS) - continuation
           - Change of character (CHOCH) - reversal
           - Higher highs/lower lows broken
           - Shift from bullish to bearish or vice versa
        
        3. **Institutional Order Flow**:
           - Order blocks (OB) - last opposing candle before move
           - Fair Value Gaps (FVG) - 3-candle imbalance
           - Breaker blocks - broken structure turned support/resistance
           - Mitigation blocks - old highs/lows acting as support/resistance
        
        4. **Premium/Discount Arrays**:
           - Is price in premium (sell side) or discount (buy side)?
           - Fibonacci retracements (62%, 79%, 88.6%)
           - Equilibrium levels (50%)
        
        5. **Multi-Timeframe Confluence**:
           - Higher timeframe bias and structure
           - Lower timeframe entry models
           - Session alignments (London, New York)

        === ICT ENTRY REQUIREMENTS (MANDATORY) ===
        **ONLY ENTER AFTER BOTH CONDITIONS:**
        1. **LIQUIDITY SWEEP**: Price must sweep highs/lows to grab liquidity
        2. **MARKET STRUCTURE SHIFT**: Clear MSS/BOS/CHOCH must occur after sweep
        
        **ENTRY PROCESS:**
        - Wait for liquidity sweep (external liquidity taken)
        - Confirm market structure shift in opposite direction
        - Enter on pullback to order block/FVG in direction of new structure
        - Target opposing liquidity or next significant level

        === REQUIRED DECISION ===
        {possible_decisions}

        === DECISION DETAILS ===
        """
        
        if current_state == TradingState.NO_SIGNAL.value:
            prompt += """
        **ENTER_NEW**: Create new ICT signal
        - Conditions: BOTH liquidity sweep + MSS completed, clear order block/FVG entry
        - Include: Specific Entry (OB/FVG), SL (beyond invalidation), TP (opposing liquidity)
        
        **WAIT**: Wait for proper ICT setup
        - Conditions: No liquidity sweep yet, no MSS, or unclear structure
        """
        elif current_state == TradingState.PENDING_LIMIT.value:
            prompt += """
        **CANCEL**: Cancel limit order
        - Conditions: Market structure invalidated, new MSS in opposite direction
        
        **KEEP_WAITING**: Keep limit order
        - Conditions: Structure still valid, waiting for mitigation of order block/FVG
        """
        elif current_state == TradingState.ACTIVE_POSITION.value:
            prompt += """
        **HOLD**: Hold position
        - Conditions: Structure intact, moving toward target liquidity
        
        **CLOSE_NOW**: Close now
        - Conditions: Opposing MSS occurred, structure broken against position
        
        **PARTIAL_EXIT**: Partial exit
        - Conditions: Reached intermediate liquidity, signs of potential reversal
        """
        
        prompt += """

        === RESULT FORMAT (KEEP exact format like below, don't give any additional information) ===
        DECISION: [ENTER_NEW/WAIT/CANCEL/KEEP_WAITING/HOLD/CLOSE_NOW/PARTIAL_EXIT]
        CONFIDENCE: [HIGH/MEDIUM/LOW]
        REASON: [Detailed ICT analysis including liquidity sweep and MSS status]
        URGENCY: [HIGH/MEDIUM/LOW]
        
        **If ENTER_NEW, add (KEEP exact format like below, don't give any additional information):**
        NEW_SIGNAL_ACTION: [LONG/SHORT]
        NEW_SIGNAL_ENTRY: $[price]
        NEW_SIGNAL_SL: $[price]
        NEW_SIGNAL_TP: $[price]
        """

        prompt = prompt.strip()
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer <API-KEY>", #TODO replace API key
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an ICT (Inner Circle Trader) methodology expert. ONLY enter trades after confirmed liquidity sweeps followed by market structure shifts. Focus on institutional order flow and high-probability setups with clear risk management."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 5000
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                logger.info(f"""Deepseek response: {analysis_text}""")
                return self.parse_comprehensive_decision(analysis_text)
            else:
                logger.info(f"""Error Deepseek response: {response.text}""")
                return self.get_fallback_decision(current_state)
                
        except Exception as e:
            return self.get_fallback_decision(current_state)

    def parse_comprehensive_decision(self, analysis_text: str):
        """Parse comprehensive decision from DeepSeek"""
        decision_data = {
            'decision': 'WAIT',
            'confidence': 'MEDIUM',
            'reason': 'Default decision',
            'urgency': 'LOW',
            'new_signal': None
        }

        lines = analysis_text.split('\n')
        collecting_reason = False
        reason_lines = []

        for line in lines:
            line = line.strip()
            clean_line = line.replace('**', '').strip()

            if 'DECISION:' in clean_line:
                decision_data['decision'] = clean_line.split(':', 1)[1].strip()
                collecting_reason = False
            elif 'CONFIDENCE:' in clean_line:
                decision_data['confidence'] = clean_line.split(':', 1)[1].strip()
                collecting_reason = False
            elif 'REASON:' in clean_line:
                reason_part = clean_line.split(':', 1)[1].strip()
                if reason_part:
                    reason_lines = [reason_part]
                else:
                    reason_lines = []
                collecting_reason = True
            elif 'URGENCY:' in clean_line:
                if reason_lines:
                    decision_data['reason'] = ' '.join(reason_lines).strip()
                decision_data['urgency'] = clean_line.split(':', 1)[1].strip()
                collecting_reason = False
            elif 'NEW_SIGNAL_ACTION:' in clean_line:
                if not decision_data['new_signal']:
                    decision_data['new_signal'] = {}
                decision_data['new_signal']['action'] = clean_line.split(':', 1)[1].strip()
                collecting_reason = False
            elif 'NEW_SIGNAL_ENTRY:' in clean_line:
                if not decision_data['new_signal']:
                    decision_data['new_signal'] = {}
                price_str = clean_line.split(':', 1)[1].strip().replace('$', '').replace(',', '')
                try:
                    decision_data['new_signal']['entry'] = float(price_str)
                except ValueError:
                    pass
                collecting_reason = False
            elif 'NEW_SIGNAL_SL:' in clean_line:
                if not decision_data['new_signal']:
                    decision_data['new_signal'] = {}
                price_str = clean_line.split(':', 1)[1].strip().replace('$', '').replace(',', '')
                try:
                    decision_data['new_signal']['sl'] = float(price_str)
                except ValueError:
                    pass
                collecting_reason = False
            elif 'NEW_SIGNAL_TP:' in clean_line:
                if not decision_data['new_signal']:
                    decision_data['new_signal'] = {}
                price_str = clean_line.split(':', 1)[1].strip().replace('$', '').replace(',', '')
                try:
                    decision_data['new_signal']['tp'] = float(price_str)
                except ValueError:
                    pass
                collecting_reason = False
            elif collecting_reason and clean_line and not clean_line.startswith('Note:'):
                reason_lines.append(clean_line)

        if reason_lines and collecting_reason:
            decision_data['reason'] = ' '.join(reason_lines).strip()

        return decision_data

    def get_fallback_decision(self, current_state: str):
        """Fallback decision when API fails"""
        fallback_decisions = {
            TradingState.NO_SIGNAL.value: 'WAIT',
            TradingState.PENDING_LIMIT.value: 'KEEP_WAITING',
            TradingState.ACTIVE_POSITION.value: 'HOLD',
            TradingState.SIGNAL_EXPIRED.value: 'WAIT'
        }
        
        return {
            'decision': fallback_decisions.get(current_state, 'WAIT'),
            'confidence': 'LOW',
            'reason': 'API unavailable - using safe fallback',
            'urgency': 'LOW',
            'new_signal': None
        }
