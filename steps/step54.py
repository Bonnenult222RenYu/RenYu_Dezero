# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import test_mode
import dezero.functions as F
import numpy as np

x = np.ones(5)
print(x)

# When training
y = F.dropout(x)
print(y)

# When testing (predicting)
with test_mode():
    y = F.dropout(x)
    print(y)