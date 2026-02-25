# ============================================
# Install & Import Dependencies
# ============================================
import pandas as pd
import numpy as np
from scipy.io import loadmat
from datetime import datetime, timedelta

# image processing dependencies
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from pyts.preprocessing import InterpolationImputer

# fastai data loading and training dependencies
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback
import dill
from PIL import Image
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)


# ============================================
# Helper Function: MATLAB datenum → datetime
# ============================================
def matlab2datetime(matlab_datenum):
    return (
        datetime.fromordinal(int(matlab_datenum))
        + timedelta(days=matlab_datenum % 1)
        - timedelta(days=366)
    )


# ============================================
# Load .mat Dataset
# ============================================
data = loadmat("NEUSTG_19502020_12stations.mat")

lat = data["lattg"].flatten()
lon = data["lontg"].flatten()
sea_level = data["sltg"]
station_names = [s[0] for s in data["sname"].flatten()]
time = data["t"].flatten()
time_dt = np.array([matlab2datetime(t) for t in time])

# ============================================
# Select Target Stations
# ============================================
SELECTED_STATIONS = [
    "Annapolis",
    "Atlantic_City",
    "Charleston",
    "Washington",
    "Wilmington",
    "Eastport",
    "Portland",
    "Sewells_Point",
    "Sandy_Hook",
    "The_Battery",
    "Lewes",
    "Fernandina_Beach",
]

selected_idx = [station_names.index(st) for st in SELECTED_STATIONS]
selected_names = [station_names[i] for i in selected_idx]
selected_lat = lat[selected_idx]
selected_lon = lon[selected_idx]
selected_sea_level = sea_level[:, selected_idx]  # time × selected_stations

# ============================================
# Build Preview DataFrame
# ============================================
df_preview = pd.DataFrame(
    {
        "time": np.tile(time_dt[:5], len(selected_names)),
        "station_name": np.repeat(selected_names, 5),
        "latitude": np.repeat(selected_lat, 5),
        "longitude": np.repeat(selected_lon, 5),
        "sea_level": selected_sea_level[:5, :].T.flatten(),
    }
)

# ============================================
# Print Data Head
# ============================================
print(f"Number of stations: {len(selected_names)}")
print(f"Sea level shape (time x stations): {selected_sea_level.shape}")
# df_preview.head()

# ============================================
# Convert Hourly → Daily per Station
# ============================================
# Convert time to pandas datetime
time_dt = pd.to_datetime(time_dt).round("h")

# Build hourly DataFrame for selected stations
df_hourly = pd.DataFrame(
    {
        "time": np.tile(time_dt, len(selected_names)),
        "station_name": np.repeat(selected_names, len(time_dt)),
        "latitude": np.repeat(selected_lat, len(time_dt)),
        "longitude": np.repeat(selected_lon, len(time_dt)),
        "sea_level": selected_sea_level.flatten(),
    }
)


# ============================================
# Compute Flood Threshold per Station
# ============================================

threshold_mat = loadmat("Seed_Coastal_Stations_Thresholds.mat")
names = [s[0] for s in data["sname"].flatten()]
thresholds = threshold_mat["thminor_stnd"].flatten()
threshold_df = pd.DataFrame({"station_name": names, "flood_threshold": thresholds})

df_hourly = df_hourly.merge(
    threshold_df[["station_name", "flood_threshold"]], on="station_name", how="left"
)

# ============================================
# Daily Aggregation + Flood Flag
# ============================================
df_daily = (
    df_hourly.groupby(["station_name", pd.Grouper(key="time", freq="D")])
    .agg(
        {
            "sea_level": "mean",
            "latitude": "first",
            "longitude": "first",
            "flood_threshold": "first",
        }
    )
    .reset_index()
)

# Flood flag: 1 if any hourly value exceeded threshold that day
hourly_max = (
    df_hourly.groupby(["station_name", pd.Grouper(key="time", freq="D")])["sea_level"]
    .max()
    .reset_index()
)
df_daily = df_daily.merge(
    hourly_max, on=["station_name", "time"], suffixes=("", "_max")
)
df_daily["flood"] = (df_daily["sea_level_max"] > df_daily["flood_threshold"]).astype(
    int
)

df_hourly.to_pickle("df_hourly.pkl")
df_daily.to_pickle("df_daily.pkl")

# Filter data to only include training stations and
# exclude testing windows
training_stations = [
    "Annapolis",
    "Atlantic_City",
    "Charleston",
    "Washington",
    "Wilmington",
    "Eastport",
    "Portland",
    "Sewells_Point",
    "Sandy_Hook",
]


test_windows = [
    ("1962-03-06", "1962-03-26"),
    ("2013-07-21", "2013-08-10"),
    ("2011-05-13", "2011-06-02"),
    ("1995-12-21", "1996-01-10"),
    ("1995-09-05", "1995-09-25"),
    ("2009-12-31", "2010-01-20"),
    ("2020-09-16", "2020-10-06"),
    ("2013-10-07", "2013-10-27"),
    ("1958-04-03", "1958-04-23"),
    ("1988-04-08", "1988-04-28"),
    ("1996-12-04", "1996-12-24"),
    ("2003-04-14", "2003-05-04"),
    ("1979-01-25", "1979-02-14"),
    ("2015-03-18", "2015-04-07"),
]

df_daily_training = df_daily.query("station_name in @training_stations")

for s, e in test_windows:
    mask = (df_daily_training["time"] < s) | (df_daily_training["time"] > e)
    df_daily_training = df_daily_training.loc[mask]

# ============================================
# Build 7-day → 14-day Training Windows
# ============================================

FEATURES = ["sea_level"]
HIST_DAYS = 7
FUTURE_DAYS = 14

X_train, y_train = [], []
# Group hourly dataset by date
hourly_grouped = df_hourly.groupby(
    [df_hourly["time"].dt.date, df_hourly["station_name"]]
)

# Use daily dataset to construct 7 day windows from hourly
for stn, grp in df_daily_training.groupby("station_name"):
    grp = grp.sort_values("time").reset_index(drop=True)
    for i in range(0, len(grp) - HIST_DAYS - FUTURE_DAYS, HIST_DAYS):
        hist_days = grp.loc[i : i + HIST_DAYS - 1, "time"].dt.date.values
        hist_frames = []
        for day in hist_days:
            hist_frames.append(hourly_grouped.get_group((day, stn))[FEATURES])
        hist = pd.concat(hist_frames)
        if hist.shape != (168, 1):
            print(hist.shape, stn, hist_frames)
        future = grp.loc[
            i + HIST_DAYS : i + HIST_DAYS + FUTURE_DAYS - 1, "flood"
        ].values
        X_train.append(hist)
        y_train.append(future)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Use linear interpolation to fill missing values

imputer = InterpolationImputer()
X_train_imputed = imputer.fit_transform(X_train.squeeze())

# Create Gramian Angular Summation Fields from 7 day training windows
gaf = GramianAngularField()
X_train_GAF = gaf.fit_transform(X_train_imputed)


# np.save('X_train_GAF.npy', X_train_GAF)
# np.save('y_train.npy', y_train)


def get_x(i):
    # Scale GAF from [-1, 1] to [0, 255]
    gaf_scaled = ((X_train_GAF[i] + 1) * 127.5).astype(np.uint8)
    # Create grayscale PIL image
    img = Image.fromarray(gaf_scaled, mode="L")
    # Convert to RGB for pretrained models
    return PILImage.create(img).convert("RGB")


def get_y(i):
    return tensor(y_train[i]).float()


# X_train_GAF = numpy.load('X_train_GAF.npy')
# y_train = numpy.load('y_train.npy')

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=list(range(14)))),
    splitter=RandomSplitter(),
    get_x=get_x,
    get_y=get_y,
)

dls = dblock.dataloaders(range(len(X_train_GAF)), bs=32, num_workers=0)
learn = vision_learner(dls, resnet18, loss_func=BCEWithLogitsLossFlat())

# print(learn.lr_find())
# SuggestedLRs(valley=0.001737800776027143)1:28 2.3867]

# Save the best model automatically during training
cbs = [
    SaveModelCallback(monitor="valid_loss", fname="best_model"),
    EarlyStoppingCallback(monitor="valid_loss", patience=3, min_delta=0.01),
]
learn.fine_tune(12, base_lr=0.001, wd=0.01, cbs=cbs)

# Model training results
# epoch     train_loss  valid_loss  time
# 0         0.189654    0.119670    02:19
# Better model found at epoch 0 with valid_loss value: 0.1196695938706398.
# epoch     train_loss  valid_loss  time
# 0         0.128689    0.089312    04:35
# Better model found at epoch 0 with valid_loss value: 0.08931196480989456.
# 1         0.110778    0.085626    04:32
# Better model found at epoch 1 with valid_loss value: 0.08562576770782471.
# 2         0.104811    0.083499    04:48
# Better model found at epoch 2 with valid_loss value: 0.08349855244159698.
# 3         0.095273    0.080458    04:50
# Better model found at epoch 3 with valid_loss value: 0.08045750856399536.
# No improvement since epoch 0: early stopping

# Save the model
learn.save("flood_model")
learn.export("flood_model.pkl", pickle_module=dill)


test_stations = ["Lewes", "Fernandina_Beach", "The_Battery"]

# df_hourly = pd.read_pickle('df_hourly.pkl')
# df_daily_test = pd.read_pickle('df_daily.pkl').query('station_name in @test_stations')

df_daily_test = df_daily.query("station_name in @test_stations")

start_dates = [
    "3-6-1962",
    "7-21-2013",
    "5-13-2011",
    "12-21-1995",
    "9-5-1995",
    "12-31-2009",
    "9-16-2020",
    "10-7-2013",
    "4-3-1958",
    "4-8-1988",
    "12-4-1996",
    "4-14-2003",
    "1-25-1979",
    "3-18-2015",
]

end_dates = [
    "3-12-1962",
    "7-27-2013",
    "5-19-2011",
    "12-27-1995",
    "9-11-1995",
    "1-6-2010",
    "9-22-2020",
    "10-13-2013",
    "4-9-1958",
    "4-14-1988",
    "12-10-1996",
    "4-20-2003",
    "1-31-1979",
    "3-24-2015",
]


# ======================
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid:
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()


FEATURES = ["sea_level"]
X_test = []
y_true = []

hourly_grouped = df_hourly.groupby(
    [df_hourly["time"].dt.date, df_hourly["station_name"]]
)

for i in range(len(start_dates)):
    hist_start = pd.to_datetime(start_dates[i])
    hist_end = pd.to_datetime(end_dates[i])

    # Forecast period
    test_start = hist_end + pd.Timedelta(days=1)
    test_end = test_start + pd.Timedelta(days=13)

    print(f"Historical window: {hist_start.date()} → {hist_end.date()}")
    print(f"Forecast window:   {test_start.date()} → {test_end.date()}")

    # Build X_test for Selected Window
    for stn, grp in df_daily_test.groupby("station_name"):
        mask = (grp["time"] >= hist_start) & (grp["time"] <= hist_end)
        hist_days = grp.loc[mask, "time"].dt.date.values
        if len(hist_days) == 7:  # Ensure full 7-day block
            hist_frames = []
            for day in hist_days:
                hist_frames.append(hourly_grouped.get_group((day, stn))[FEATURES])
            hist = pd.concat(hist_frames)
            if hist.shape == (168, 1):  # 7 days * 24 hours
                X_test.append(hist.values.flatten())
        # Collect Ground Truth
        mask = (grp["time"] >= test_start) & (grp["time"] <= test_end)
        vals = grp.loc[mask, "flood"].values
        if len(vals) == 14:
            y_true.append(vals)

X_test = np.array(X_test)
y_true = np.array(y_true)

# ============================================
# Transform test windows to GAF images
# ============================================
imputer = InterpolationImputer()
X_test_imputed = imputer.fit_transform(X_test)

gaf = GramianAngularField()
X_test_GAF = gaf.fit_transform(X_test_imputed)


# ============================================
# Create test dataloader
# ============================================
# learn = load_learner('flood_model.pkl',pickle_module=dill)
def get_x_test(i):
    X_test_GAF_scaled = ((X_test_GAF + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(X_test_GAF_scaled[i], mode="L")
    return PILImage.create(img).convert("RGB")


test_imgs = [get_x_test(i) for i in range(len(X_test_GAF))]

# Create test dataloader
test_dl = learn.dls.test_dl(test_imgs)

# ============================================
# Forecast 14 Days Ahead
# ============================================
preds, _ = learn.get_preds(dl=test_dl)
y_pred = preds.numpy()
y_pred_bin = (y_pred > 0.5).astype(int)


# ============================================
# Evaluation
# ============================================
y_true_flat = y_true.flatten()
y_pred_flat = y_pred_bin.flatten()
# print(y_true_flat, y_pred_flat)
tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()
acc = accuracy_score(y_true_flat, y_pred_flat)
f1 = f1_score(y_true_flat, y_pred_flat)
mcc = matthews_corrcoef(y_true_flat, y_pred_flat)

print("=== Confusion Matrix ===")
print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
print("\n=== Metrics ===")
print(f"Accuracy: {acc:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"MCC: {mcc:.3f}")

# === Confusion Matrix ===
# TP: 404 | FP: 23 | TN: 42 | FN: 119
#
# === Metrics ===
# Accuracy: 0.759
# F1 Score: 0.851
# MCC: 0.294
