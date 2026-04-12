REGISTRY = {}

from .MAE_belief_agent import MAERNNAgent
REGISTRY["MAEAgent"] = MAERNNAgent

from .belief_rnn_agent import RNNAgent as BeliefRNNAgent
REGISTRY["belief_rnn"] = BeliefRNNAgent

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

# from .transformer_rnn_agent import TransformerRNNAgent
# REGISTRY["transformer_rnn"] = TransformerRNNAgent

from .updet_agent import UPDeT
REGISTRY["updet"] = UPDeT

from .transformer import TransformerAggregationAgent
REGISTRY["transformer"] = TransformerAggregationAgent

from .central_rnn_agent import CentralRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent

from .rtc_agent import HistoryRTCs
REGISTRY["rtcs"] = HistoryRTCs

from .rnn_sd_agent import RNN_SD_Agent
REGISTRY["rnn_sd"] = RNN_SD_Agent

from .transformer_belief_agent import TransformerBeliefAgent
REGISTRY["transformer_belief"] = TransformerBeliefAgent
from .history_token_transformer_belief_agent import HistoryTokenTransformerBeliefAgent
REGISTRY["history_token_transformer_belief"] = HistoryTokenTransformerBeliefAgent
from .token_transformer_belief_agent import TokenTransformerBeliefAgent
REGISTRY["token_transformer_belief"] = TokenTransformerBeliefAgent
from .token_latent_belief_agent import TokenLatentBeliefAgent
REGISTRY["token_latent_belief"] = TokenLatentBeliefAgent
from .token_value_belief_agent import TokenValueBeliefAgent
REGISTRY["token_value_belief"] = TokenValueBeliefAgent

from .history_token_rnn_belief_agent import HistoryTokenRNNBeliefAgent
REGISTRY["history_token_rnn_belief"] = HistoryTokenRNNBeliefAgent

from .enemy_history_transformer_belief_agent import EnemyHistoryTransformerBeliefAgent
REGISTRY["enemy_history_transformer_belief"] = EnemyHistoryTransformerBeliefAgent
from .belief_slot_transformer_agent import BeliefSlotTransformerAgent
REGISTRY["belief_slot_transformer"] = BeliefSlotTransformerAgent
from .history_slot_belief_agent import HistorySlotBeliefAgent
REGISTRY["history_slot_belief"] = HistorySlotBeliefAgent
from .current_slot_belief_rnn_agent import CurrentSlotBeliefRNNAgent
REGISTRY["current_slot_belief_rnn"] = CurrentSlotBeliefRNNAgent

from .TrXL_belief_agent import TrXLBeliefAgent
REGISTRY["TrXL_belief_agent"] = TrXLBeliefAgent
REGISTRY["trxl_belief_agent"] = TrXLBeliefAgent
