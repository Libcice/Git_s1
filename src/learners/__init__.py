from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .max_q_learner import MAXQLearner
from .max_q_learner_ddpg import DDPGQLearner
from .max_q_learner_sac import SACQLearner
from .q_learner_w import QLearner as WeightedQLearner
from .qatten_learner import QattenLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .shaq_learner import SHAQLearner
from .sqddpg_learner import SQDDPGLearner
from .qamix_learner import QamixLearner
from .cds_learner import CDS_Learner
from .qnam_learner import QNAM_Learner
from .dvd_learner import DVDLearner
from .q_learner_belief import QLearner_belief
from .q_belief_sep import  QLearner_belief_sep
from .belief_mae_qlearner import  MAEQLearner

REGISTRY = {}
REGISTRY["q_belief_mae"] = MAEQLearner
REGISTRY["q_belief_sep"] = QLearner_belief_sep
REGISTRY["q_learner_belief"] = QLearner_belief
REGISTRY["q_learner"] = QLearner
REGISTRY["w_q_learner"] = WeightedQLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["sac"] = SACQLearner
REGISTRY["ddpg"] = DDPGQLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["shaq_learner"] = SHAQLearner
REGISTRY["sqddpg_learner"] = SQDDPGLearner
REGISTRY["qamix_learner"] = QamixLearner
REGISTRY["cds_learner"] = CDS_Learner
REGISTRY["dvd_learner"] = DVDLearner
REGISTRY["qnam_learner"] = QNAM_Learner