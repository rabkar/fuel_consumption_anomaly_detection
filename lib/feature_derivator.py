import numpy as np
import pandas as pd

def komtrax_feature_derivator(X):
    X['daily_productivity'] = (X['daily_actual_work_value']/X['daily_smr_value'])
    X['delta_pe_mode'] = ((X['daily_pmode'] - X['daily_emode'])/(X['daily_pmode'] + X['daily_emode']))
    X['fuel_consumption_ratio'] = (X['daily_fuel_value']*1.1/X['daily_smr_value'])



