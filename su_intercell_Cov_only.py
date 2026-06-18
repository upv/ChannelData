"""
Single-user uplink with inter-cell interference, calibrated to a defined IoT.

Scenario:
  * One SERVING BS (the only receiver) at the origin.
  * One SERVED UE inside the serving cell  -> desired uplink signal.
  * Several INTERFERING UEs located in neighbouring cells (around the first
    tier of hex neighbours). Their uplink transmissions leak into the serving
    BS as inter-cell interference.

The aggregate interference is calibrated to a DEFINED IoT (interference-over-
thermal), IoT = (I + N) / N, by scaling the common interferer transmit power.

This script BATCH-GENERATES the channel for every model
  * channel model : UMa, UMi, CDL-B, CDL-C
and for each model saves TWO pictures:
  * one localization window (BS, served UE, interferers), and
  * one bundle (grid) of received-covariance windows, one per case over
        IoT [dB]     : 0, 10, 20
        received SNR : -20, 0, 20 dB
    each window annotated with its channel / IoT / SNR case.
The per-model channel tensors are stored in .pkl files with matching names.

Requires: sionna>=2.0 (PyTorch backend), torch, numpy, matplotlib.
"""

import glob
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from sionna.phy import config
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMa, UMi, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# ------------------------------- parameters ------------------------------- #
CARRIER_FREQUENCY  = 3.5e9
SUBCARRIER_SPACING = 30e3
FFT_SIZE           = 76
NUM_OFDM_SYMBOLS   = 14
NUM_BS_ANT         = 64        # serving-BS antennas (4x8 dual-pol panel)
NUM_UT_ANT         = 4         # antennas per user (1x2 dual-pol)

ISD               = 500.0       # inter-site distance [m] (UMa macro)
NUM_INTERFERERS   = 6           # one per first-tier neighbour cell (<= 6)
UT_HEIGHT         = 1.5

# Combinations to generate.
CHANNELS = ["UMa", "UMi", "CDL-B", "CDL-C"]
IOT_LIST = [0.0, 10.0, 20.0]    # interference-over-thermal targets [dB]
SNR_LIST = [-20.0, 0.0, 20.0]   # served-user received SNR vs thermal noise [dB]

# CDL-specific knobs (CDL is a link-level model with NO geometric path loss, so
# the per-link distance attenuation is applied manually using a log-distance law).
DELAY_SPREAD       = 100e-9     # rms delay spread for CDL-B / CDL-C [s]
PATH_LOSS_EXPONENT = 3.5        # path-loss exponent for the CDL geometry scaling

# A fresh random seed is drawn from OS entropy on every launch, so the served
# user and interferers are placed at NEW random positions each run. This single
# layout is then reused for every channel / IoT / SNR scenario below.
SEED = int(np.random.SeedSequence().generate_state(1)[0])
config.seed = SEED
rng = np.random.default_rng(SEED)
print(f"Random layout seed (new each launch) = {SEED}")

NUM_UT   = 1 + NUM_INTERFERERS   # UT 0 = served user
R_CELL   = ISD / np.sqrt(3.0)    # hex centre-to-vertex radius

# ---------------------------- geometry (top-down) ------------------------- #
# One shared random layout is used for every channel / IoT / SNR combination so
# the localization panel is consistent across all generated pictures.
def drop_in_cell(center, R, min_d=15.0):
    while True:
        ang = rng.uniform(0, 2*np.pi)
        r   = R * np.sqrt(rng.uniform(0, 0.81))     # area-uniform within 0.9 R
        if r > min_d:
            return np.array([center[0] + r*np.cos(ang),
                             center[1] + r*np.sin(ang)])

serving_bs = np.array([0.0, 0.0])
neigh_ang  = np.deg2rad(60.0 * np.arange(6))
neigh_bs   = np.stack([ISD*np.cos(neigh_ang), ISD*np.sin(neigh_ang)], axis=1)

served_ue  = drop_in_cell(serving_bs, R_CELL)
interferers = np.stack([drop_in_cell(neigh_bs[j], R_CELL)
                        for j in range(NUM_INTERFERERS)], axis=0)

ut_xy = np.vstack([served_ue, interferers])          # [NUM_UT, 2]
dist  = np.linalg.norm(ut_xy - serving_bs, axis=1)   # 2-D distance to serving BS

# ------------------------------ antenna arrays ---------------------------- #
ut_array = AntennaArray(num_rows=1, num_cols=NUM_UT_ANT // 2,
                        polarization="dual", polarization_type="cross",
                        antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY)
assert ut_array.num_ant == NUM_UT_ANT
BS_ROWS, BS_COLS = 4, 8                              # 4 x 8 x 2 (dual-pol) = 64
bs_array = AntennaArray(num_rows=BS_ROWS, num_cols=BS_COLS,
                        polarization="dual", polarization_type="cross",
                        antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY)
assert bs_array.num_ant == NUM_BS_ANT

# --------------------- channel model (serving BS = RX) -------------------- #
def col3(xy, z):                                     # -> [1, n, 3] float32
    arr = np.concatenate([xy, np.full((xy.shape[0], 1), z)], axis=1)
    return torch.tensor(arr[None], dtype=torch.float32)

frequencies = subcarrier_frequencies(FFT_SIZE, SUBCARRIER_SPACING)
fs = SUBCARRIER_SPACING * FFT_SIZE


def make_system_channel(model_cls, bs_height):
    """UMa / UMi: geometry-driven, multi-UT topology with built-in path loss."""
    model = model_cls(carrier_frequency=CARRIER_FREQUENCY, o2i_model="low",
                      ut_array=ut_array, bs_array=bs_array, direction="uplink",
                      enable_pathloss=True, enable_shadow_fading=True)
    model.set_topology(
        ut_loc          = col3(ut_xy, UT_HEIGHT),
        bs_loc          = col3(serving_bs[None], bs_height),
        ut_orientations = torch.zeros([1, NUM_UT, 3], dtype=torch.float32),
        bs_orientations = torch.zeros([1, 1, 3], dtype=torch.float32),
        ut_velocities   = torch.zeros([1, NUM_UT, 3], dtype=torch.float32),
        in_state        = torch.zeros([1, NUM_UT], dtype=torch.bool))  # outdoor
    return model(NUM_OFDM_SYMBOLS, fs)


def make_cdl_channel(model_letter):
    """CDL-B / CDL-C: one independent link per UT (batch = NUM_UT). CDL has no
    geometric path loss, so a log-distance attenuation is applied per link."""
    model = CDL(model_letter, delay_spread=DELAY_SPREAD,
                carrier_frequency=CARRIER_FREQUENCY,
                ut_array=ut_array, bs_array=bs_array, direction="uplink")
    a_cdl, tau_cdl = model(NUM_UT, NUM_OFDM_SYMBOLS, fs)
    # a_cdl: [NUM_UT, 1, Nbs, 1, Nut, paths, time], tau_cdl: [NUM_UT, 1, 1, paths]
    pl_lin = (dist / dist[0]) ** (-PATH_LOSS_EXPONENT)          # relative gain
    a_cdl  = a_cdl * torch.tensor(np.sqrt(pl_lin), dtype=a_cdl.dtype
                                  ).view(NUM_UT, 1, 1, 1, 1, 1, 1)
    # re-arrange to the multi-UT convention: [1, 1, Nbs, NUM_UT, Nut, paths, time]
    a_out   = a_cdl.squeeze(3).permute(1, 2, 0, 3, 4, 5).unsqueeze(0)
    tau_out = tau_cdl[:, :, 0, :].permute(1, 0, 2).unsqueeze(0)  # [1,1,NUM_UT,paths]
    return a_out, tau_out


def generate_channel(model_name):
    """Return the OFDM channel tensor h for the requested model and shared layout."""
    if model_name.startswith("CDL"):
        a, tau = make_cdl_channel(model_name.split("-")[1])
    else:
        bs_height = 10.0 if model_name == "UMi" else 25.0   # UMi micro vs UMa macro
        a, tau = make_system_channel(UMi if model_name == "UMi" else UMa, bs_height)
    return cir_to_ofdm_channel(frequencies, a, tau,
                               normalize=False).detach().cpu().numpy()


def cov(Hu):
    """BS spatial covariance E[h h^H] over (UT antenna, symbol, subcarrier)."""
    M = Hu.reshape(NUM_BS_ANT, -1)
    return (M @ M.conj().T) / M.shape[1]


def hexagon(cx, cy, R):
    ang = np.deg2rad(30 + 60*np.arange(6))
    return np.stack([cx + R*np.cos(ang), cy + R*np.sin(ang)], axis=1)


# --------------------------- localization window -------------------------- #
def make_location_figure(model_name):
    """One window with the mutual location of BS, served UE and interferers."""
    fig, axm = plt.subplots(figsize=(9, 8))
    fig.suptitle(f"Localization  -  {model_name}", fontsize=14, fontweight="bold")
    for c in np.vstack([serving_bs, neigh_bs]):
        axm.add_patch(Polygon(hexagon(c[0], c[1], R_CELL), closed=True,
                              fill=False, ec="0.7", lw=1.0))
    axm.scatter(neigh_bs[:, 0], neigh_bs[:, 1], marker="^", s=90,
                c="0.5", label="Neighbour BS")
    axm.scatter(*serving_bs, marker="^", s=260, c="k", label="Serving BS")
    axm.scatter(*served_ue, s=160, c="tab:green", ec="k", zorder=5, label="Served UE")
    axm.scatter(interferers[:, 0], interferers[:, 1], s=110, c="tab:red",
                ec="k", zorder=5, label="Interferers")
    axm.plot([serving_bs[0], served_ue[0]], [serving_bs[1], served_ue[1]],
             c="tab:green", lw=2.0, zorder=3)
    for j in range(NUM_INTERFERERS):
        axm.plot([serving_bs[0], interferers[j, 0]],
                 [serving_bs[1], interferers[j, 1]],
                 c="tab:red", lw=1.0, ls="--", alpha=0.6, zorder=2)
        axm.annotate(f"{dist[j+1]:.0f} m", interferers[j],
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    axm.annotate(f"{dist[0]:.0f} m", served_ue, fontsize=9, fontweight="bold",
                 xytext=(4, 4), textcoords="offset points")
    axm.set_title("serving BS, served UE, inter-cell interferers")
    axm.set_xlabel("x [m]"); axm.set_ylabel("y [m]")
    axm.set_aspect("equal"); axm.grid(alpha=0.3); axm.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = f"su_intercell_location_{model_name}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ----------------- bundle of covariance windows (IoT x SNR) --------------- #
def make_cov_bundle(model_name, R_ut, R_int_unit, gain, g_int):
    """Grid of received-covariance windows, one per (IoT, SNR) case."""
    nrow, ncol = len(IOT_LIST), len(SNR_LIST)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow),
                             squeeze=False)
    fig.suptitle(f"BS received covariance |R$_{{rx}}$| = signal + interf + noise"
                 f"   -   {model_name}", fontsize=15, fontweight="bold")

    for r, iot_db in enumerate(IOT_LIST):
        p_int = (10 ** (iot_db / 10.0) - 1.0) / np.sum(g_int)
        R_in  = p_int * R_int_unit + EYE                     # interference + noise
        iot_ach = 10 * np.log10(np.trace(R_in).real / NUM_BS_ANT)
        for c, snr_db in enumerate(SNR_LIST):
            R_sig = (10 ** (snr_db / 10.0) / gain[0]) * R_ut[0]   # desired signal
            R_rx  = R_sig + R_in
            ax = axes[r, c]
            im = ax.imshow(np.abs(R_rx), cmap="magma")
            ax.set_title(f"{model_name}  |  IoT = {iot_db:.0f} dB  (≈{iot_ach:.1f}),  "
                         f"SNR = {snr_db:.0f} dB", fontsize=10)
            ax.set_xlabel("BS antenna idx"); ax.set_ylabel("BS antenna idx")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|R|")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = f"su_intercell_cov_{model_name}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ------------------------------ main batch loop --------------------------- #
# Clear all existing .png files before generating the new pictures.
old_pngs = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.png"))
for f in old_pngs:
    os.remove(f)
print(f"Cleared {len(old_pngs)} existing .png file(s).")

EYE = np.eye(NUM_BS_ANT)

for model_name in CHANNELS:
    h    = generate_channel(model_name)
    # h: [1, num_rx=1, NUM_BS_ANT, NUM_UT, NUM_UT_ANT, NUM_OFDM_SYMBOLS, FFT_SIZE]
    H    = [h[0, 0, :, u, :, :, :] for u in range(NUM_UT)]
    gain = np.array([np.mean(np.abs(Hu)**2) for Hu in H])   # linear power gain
    g_int      = gain[1:]
    R_ut       = [cov(Hu) for Hu in H]
    R_int_unit = sum(R_ut[j] for j in range(1, NUM_UT))     # unit-Tx interferer cov
    H_mag_db   = 20 * np.log10(np.mean(np.abs(H[0]), axis=(1, 2)) + 1e-12)

    # Save the generated channel (model-dependent, independent of IoT/SNR).
    chan_file = f"su_intercell_channel_{model_name}.pkl"
    with open(chan_file, "wb") as fp:
        pickle.dump({"channel_model": model_name, "seed": SEED, "h": h,
                     "ut_xy": ut_xy, "serving_bs": serving_bs,
                     "neigh_bs": neigh_bs, "dist": dist, "gain": gain,
                     "H_mag_db": H_mag_db}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{model_name}] h shape {h.shape}  ->  saved channel to {chan_file}")

    loc_png = make_location_figure(model_name)
    cov_png = make_cov_bundle(model_name, R_ut, R_int_unit, gain, g_int)
    print(f"    location -> {loc_png}")
    print(f"    covariance bundle ({len(IOT_LIST)}x{len(SNR_LIST)} cases) -> {cov_png}")

n_fig = len(CHANNELS) * 2
print(f"\nDone: {n_fig} figures and {len(CHANNELS)} channel files saved.")
