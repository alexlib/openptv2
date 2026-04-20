# algorithms/constants.py
# Constants matching C header definitions in lib/include/

# tracking_frame_buf.h
POSI = 80
STR_MAX_LEN = 255
PT_UNUSED = -999
CORRES_NONE = -1
PREV_NONE = -1
NEXT_NONE = -2
PRIO_DEFAULT = 2

# track.h
MAX_TARGETS = 20000
MAX_CANDS = 4
TR_UNUSED = -1

# epi.h
MAXCAND = 200

# orientation.h
COORD_UNUSED = -1e10

# orientation.c
NUM_ITER = 80
POS_INF = 1e20
CONVERGENCE = 0.00001
