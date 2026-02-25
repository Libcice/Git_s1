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