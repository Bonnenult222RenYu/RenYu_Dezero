# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from dezero.utils import get_conv_outsize


H, W = 4, 4  # Input size
KH, KW = 3, 3  # Kernel size
SH, SW = 1, 1  # Kernel stride
PH, PW = 1, 1  # Padding size

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)