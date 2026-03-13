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

from .history_token_rnn_belief_controller import HistoryTokenRNNBeliefMAC
REGISTRY["history_token_rnn_belief_mac"] = HistoryTokenRNNBeliefMAC

from .enemy_history_transformer_belief_controller import EnemyHistoryTransformerBeliefMAC
REGISTRY["enemy_history_transformer_belief_mac"] = EnemyHistoryTransformerBeliefMAC
