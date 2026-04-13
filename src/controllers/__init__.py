REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_policy import BasicMAC as PolicyMAC
from .central_basic_controller import CentralBasicMAC
from .basic_controller_belief import BasicMAC as BeliefMAC
from .mac_mae_belief import MAEMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["policy"] = PolicyMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["belief_mac"] = BeliefMAC
REGISTRY["MAE_mac"] = MAEMAC
from .basic_controller_transformer_belief import BasicMACTransformerBelief
REGISTRY["basic_mac_transformer_belief"] = BasicMACTransformerBelief
from .history_token_transformer_belief_controller import HistoryTokenTransformerBeliefMAC
REGISTRY["history_token_transformer_belief_mac"] = HistoryTokenTransformerBeliefMAC
from .token_transformer_belief_controller import TokenTransformerBeliefMAC
REGISTRY["token_transformer_belief_mac"] = TokenTransformerBeliefMAC
from .token_latent_belief_controller import TokenLatentBeliefMAC
REGISTRY["token_latent_belief_mac"] = TokenLatentBeliefMAC
from .token_value_belief_controller import TokenValueBeliefMAC
REGISTRY["token_value_belief_mac"] = TokenValueBeliefMAC
from .token_task_belief_controller import TokenTaskBeliefMAC
REGISTRY["token_task_belief_mac"] = TokenTaskBeliefMAC
from .token_structured_belief_controller import TokenStructuredBeliefMAC
REGISTRY["token_structured_belief_mac"] = TokenStructuredBeliefMAC

from .history_token_rnn_belief_controller import HistoryTokenRNNBeliefMAC
REGISTRY["history_token_rnn_belief_mac"] = HistoryTokenRNNBeliefMAC

from .enemy_history_transformer_belief_controller import EnemyHistoryTransformerBeliefMAC
REGISTRY["enemy_history_transformer_belief_mac"] = EnemyHistoryTransformerBeliefMAC
from .belief_slot_transformer_controller import BeliefSlotTransformerMAC
REGISTRY["belief_slot_transformer_mac"] = BeliefSlotTransformerMAC
from .history_slot_belief_controller import HistorySlotBeliefMAC
REGISTRY["history_slot_belief_mac"] = HistorySlotBeliefMAC

from .TrXL_belief_controller import TrXLBeliefMAC
REGISTRY["TrXL_belief_mac"] = TrXLBeliefMAC
REGISTRY["trxl_belief_mac"] = TrXLBeliefMAC
