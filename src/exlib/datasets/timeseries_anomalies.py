import os
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class TimeseriesAnomalies(Dataset):
    """ Time series data stored in a CSV format.
        We give info like the number of rows apriori (header is not a row),
        since we are loading these big files dynamically.
    """
    def __init__(self, data_dir, num_csv_rows, num_features,
                 mode="single", window_len=-1, col_names=None):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "data.csv")
        self.exp_file = os.path.join(data_dir, "exp.csv")
        self.num_csv_rows = num_csv_rows
        self.num_features = num_features
        self.col_names = col_names

        assert mode in ["single", "window"]
        self.mode = mode
        self.window_len = window_len

        if mode == "window":
            assert window_len > 0

    def __len__(self):
        if self.mode == "single":
            return self.num_csv_rows
        elif self.mode == "window":
            return self.num_csv_rows - self.window_len + 1
        else:
            raise NotImplementedError()

    def get_one_row(self, idx):
        data_df = pd.read_csv(self.data_file, index_col=0, skiprows=idx+1, nrows=1, header=None)
        exp_df = pd.read_csv(self.exp_file, skiprows=idx+1, nrows=1, header=None)

        data_np = np.array(data_df.iloc[0])
        exp_np = np.array(exp_df.iloc[0])

        assert data_np.size == self.num_features + 1
        assert exp_np.size == self.num_features

        x = torch.from_numpy(data_np[:self.num_features])
        y = torch.tensor(data_np[self.num_features])
        a = torch.from_numpy(exp_np)
        return x, a.long(), y.long()

    def get_window(self, start_idx, count):
        xs, aa, ys = [], [], []
        for i in range(start_idx, start_idx + count):
            x, a, y = self.get_one_row(i)
            xs.append(x)
            aa.append(a)
            ys.append(y)
        xs = torch.stack(xs)
        aa = torch.stack(aa)
        ys = torch.stack(ys)
        return xs, aa.long(), ys.long()

    def __getitem__(self, idx):
        if self.mode == "single":
            return self.get_one_row(idx)
        elif self.mode == "window":
            return self.get_window(idx, self.window_len)
        else:
            raise NotImplementedError()


# 5 per line, total 51 items
swat_col_names = [
    "FIT101", "LIT101", "MV101", "P101", "P102",
    "AIT201", "AIT202", "AIT203", "FIT201", "MV201",
    "P201", "P202", "P203", "P204", "P205",
    "P206", "DPIT301", "FIT301", "LIT301", "MV301",
    "MV302", "MV303", "MV304", "P301", "P302",
    "AIT401", "AIT402", "FIT401", "LIT401", "P401",
    "P402", "P403", "P404", "UV401", "AIT501",
    "AIT502", "AIT503", "AIT504", "FIT501", "FIT502",
    "FIT503", "FIT504", "P501", "P502", "PIT501",
    "PIT502", "PIT503", "FIT601", "P601", "P602",
    "P603"
]

class SWaT(TimeseriesAnomalies):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir = data_dir,
                         num_csv_rows = 946719,
                         num_features = 51,
                         col_names = swat_col_names,
                         **kwargs)


# 5 per line, total 127 items
wadi_col_names = [
    "1_AIT_001_PV", "1_AIT_002_PV", "1_AIT_003_PV", "1_AIT_004_PV", "1_AIT_005_PV",
    "1_FIT_001_PV", "1_LS_001_AL", "1_LS_002_AL", "1_LT_001_PV", "1_MV_001_STATUS",
    "1_MV_002_STATUS", "1_MV_003_STATUS", "1_MV_004_STATUS", "1_P_001_STATUS", "1_P_002_STATUS",
    "1_P_003_STATUS", "1_P_004_STATUS", "1_P_005_STATUS", "1_P_006_STATUS", "2_DPIT_001_PV",
    "2_FIC_101_CO", "2_FIC_101_PV", "2_FIC_101_SP", "2_FIC_201_CO", "2_FIC_201_PV",
    "2_FIC_201_SP", "2_FIC_301_CO", "2_FIC_301_PV", "2_FIC_301_SP", "2_FIC_401_CO",
    "2_FIC_401_PV", "2_FIC_401_SP", "2_FIC_501_CO", "2_FIC_501_PV", "2_FIC_501_SP",
    "2_FIC_601_CO", "2_FIC_601_PV", "2_FIC_601_SP", "2_FIT_001_PV", "2_FIT_002_PV",
    "2_FIT_003_PV", "2_FQ_101_PV", "2_FQ_201_PV", "2_FQ_301_PV", "2_FQ_401_PV",
    "2_FQ_501_PV", "2_FQ_601_PV", "2_LS_001_AL", "2_LS_002_AL", "2_LS_101_AH",
    "2_LS_101_AL", "2_LS_201_AH", "2_LS_201_AL", "2_LS_301_AH", "2_LS_301_AL",
    "2_LS_401_AH", "2_LS_401_AL", "2_LS_501_AH", "2_LS_501_AL", "2_LS_601_AH",
    "2_LS_601_AL", "2_LT_001_PV", "2_LT_002_PV", "2_MCV_007_CO", "2_MCV_101_CO",
    "2_MCV_201_CO", "2_MCV_301_CO", "2_MCV_401_CO", "2_MCV_501_CO", "2_MCV_601_CO",
    "2_MV_001_STATUS", "2_MV_002_STATUS", "2_MV_003_STATUS", "2_MV_004_STATUS", "2_MV_005_STATUS",
    "2_MV_006_STATUS", "2_MV_009_STATUS", "2_MV_101_STATUS", "2_MV_201_STATUS", "2_MV_301_STATUS",
    "2_MV_401_STATUS", "2_MV_501_STATUS", "2_MV_601_STATUS", "2_P_001_STATUS", "2_P_002_STATUS",
    "2_P_003_SPEED", "2_P_003_STATUS", "2_P_004_SPEED", "2_P_004_STATUS", "2_PIC_003_CO",
    "2_PIC_003_PV", "2_PIC_003_SP", "2_PIT_001_PV", "2_PIT_002_PV", "2_PIT_003_PV",
    "2_SV_101_STATUS", "2_SV_201_STATUS", "2_SV_301_STATUS", "2_SV_401_STATUS", "2_SV_501_STATUS",
    "2_SV_601_STATUS", "2A_AIT_001_PV", "2A_AIT_002_PV", "2A_AIT_003_PV", "2A_AIT_004_PV",
    "2B_AIT_001_PV", "2B_AIT_002_PV", "2B_AIT_003_PV", "2B_AIT_004_PV", "3_AIT_001_PV",
    "3_AIT_002_PV", "3_AIT_003_PV", "3_AIT_004_PV", "3_AIT_005_PV", "3_FIT_001_PV",
    "3_LS_001_AL", "3_LT_001_PV", "3_MV_001_STATUS", "3_MV_002_STATUS", "3_MV_003_STATUS",
    "3_P_001_STATUS", "3_P_002_STATUS", "3_P_003_STATUS", "3_P_004_STATUS", "LEAK_DIFF_PRESSURE",
    "PLANT_START_STOP_LOG", "TOTAL_CONS_REQUIRED_FLOW",
]

class WADI(TimeseriesAnomalies):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir = data_dir,
                         num_csv_rows = 1382402,
                         num_features = 127,
                         col_names = wadi_col_names,
                         **kwargs)


# 5 per line, total 86 items
hai_col_names = [
    "P1_B2004", "P1_B2016", "P1_B3004", "P1_B3005", "P1_B4002",
    "P1_B4005", "P1_B400B", "P1_B4022",  "P1_FCV01D", "P1_FCV01Z",
    "P1_FCV02D", "P1_FCV02Z", "P1_FCV03D", "P1_FCV03Z", "P1_FT01",
    "P1_FT01Z", "P1_FT02", "P1_FT02Z", "P1_FT03", "P1_FT03Z",
    "P1_LCV01D", "P1_LCV01Z", "P1_LIT01", "P1_PCV01D", "P1_PCV01Z",
    "P1_PCV02D", "P1_PCV02Z", "P1_PIT01", "P1_PIT01_HH", "P1_PIT02",
    "P1_PP01AD", "P1_PP01AR", "P1_PP01BD", "P1_PP01BR", "P1_PP02D",
    "P1_PP02R", "P1_PP04", "P1_PP04SP", "P1_SOL01D", "P1_SOL03D",
    "P1_STSP", "P1_TIT01", "P1_TIT02", "P1_TIT03", "P2_24Vdc",
    "P2_ATSW_Lamp", "P2_AutoGO", "P2_AutoSD", "P2_Emerg", "P2_MASW",
    "P2_MASW_Lamp", "P2_ManualGO", "P2_ManualSD", "P2_OnOff", "P2_RTR",
    "P2_SCO", "P2_SCST", "P2_SIT01", "P2_TripEx", "P2_VIBTR01",
    "P2_VIBTR02", "P2_VIBTR03", "P2_VIBTR04", "P2_VT01", "P2_VTR01",
    "P2_VTR02", "P2_VTR03", "P2_VTR04", "P3_FIT01", "P3_LCP01D",
    "P3_LCV01D", "P3_LH01", "P3_LIT01", "P3_LL01", "P3_PIT01",
    "P4_HT_FD", "P4_HT_PO", "P4_HT_PS", "P4_LD", "P4_ST_FD",
    "P4_ST_GOV", "P4_ST_LD", "P4_ST_PO", "P4_ST_PS", "P4_ST_PT01",
    "P4_ST_TT01",
]

class HAI(TimeseriesAnomalies):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir = data_dir,
                         num_csv_rows = 1365602,
                         num_features = 86,
                         col_names = hai_col_names,
                         **kwargs)


