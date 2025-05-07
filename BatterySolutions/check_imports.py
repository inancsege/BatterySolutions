import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import tensorflow
    print(f"TensorFlow version: {tensorflow.__version__}")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")

try:
    import xgboost
    print(f"XGBoost version: {xgboost.__version__}")
except ImportError as e:
    print(f"Error importing XGBoost: {e}")

try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"Error importing Scikit-learn: {e}")

try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"Error importing NumPy: {e}")

try:
    import pandas
    print(f"Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"Error importing Pandas: {e}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"Error importing Matplotlib: {e}")

try:
    import psutil
    print(f"psutil version: {psutil.__version__}")
except ImportError as e:
    print(f"Error importing psutil: {e}") 