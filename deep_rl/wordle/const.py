

WORDLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
WORDLE_N = 5
######################
#  Color codes used
######################
GREEN = 1
YELLOW = 0.75
GREY = 0.5
C_UNK = 0.25
######################
#  REWARDS
######################
WIN_REWARD = 30
GREEN_REWARD = 2
YELLOW_REWARD = 1
GREY_REWARD = 0.1
######################
#  LOSSES
######################
STEP_LOSS = -1
LOSS = -50

######################
# FOR CHARACTER LEVEL #
######################

int2char = {a: c for a, c in enumerate(WORDLE_CHARS)}
