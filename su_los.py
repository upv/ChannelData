"""
Single-user MIMO-OFDM channel with a MOVING user (CDL), modern Sionna 2.0.

Setup:
  * One UE with  4 antennas (1x2 dual-polarised panel).
  * One BS with 64 antennas (4x8 dual-polarised panel)  -> massive-MIMO array.
  * OFDM grid: FFT_SIZE subcarriers, NUM_OFDM_SYMBOLS symbols.
  * 3GPP TR 38.901 CDL clustered-delay-line model, uplink (UE -> BS).
  * The user is MOVING (constant speed, random heading) -> time-varying channel.

This script BATCH-GENERATES the channel for every CDL profile B, C, D, E using a
single random UE drop (new each launch, shared across profiles), and for each
profile saves THREE pictures:
  * one localization window (BS, moving UE, velocity vector),
  * one bundle (grid) of received-covariance windows  |R_rx| = signal + noise,
    one per received-SNR case, and
  * one bundle (grid) of mixed-channel magnitude windows |target + noise|,
    one per received-SNR case.
The per-profile channel tensors are stored in .npy files with matching names
(numpy-version-independent; loadable with numpy==1.26 via np.load).

Requires: sionna>=2.0 (PyTorch backend), torch, numpy, matplotlib.
"""

import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from sionna.phy import config
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# ------------------------------- parameters ------------------------------- #
CARRIER_FREQUENCY  = 3.5e9          # [Hz]
SUBCARRIER_SPACING = 30e3           # [Hz]
FFT_SIZE           = 128            # number of subcarriers
NUM_OFDM_SYMBOLS   = 512            # number of OFDM symbols (time steps)
NUM_UT_ANT         = 4              # UE antennas  (1x2 dual-pol)
NUM_BS_ANT         = 64             # BS antennas  (4x8 dual-pol)

DELAY_SPREAD       = 100e-9         # RMS delay spread of the profile [s]
DIRECTION          = "uplink"       # UE -> BS, so the BS (64) is the receiver
SPEED              = 30.0           # UE speed [m/s] (~108 km/h) -> Doppler
BS_HEIGHT          = 25.0
UT_HEIGHT          = 1.5

C0 = 299_792_458.0                  # speed of light [m/s]

# Profiles to generate (CDL-A/B/C are NLOS; CDL-D/E carry a dedicated LOS ray).
CDL_PROFILES = ["B", "C", "D", "E"]
LOS_MODELS   = {"D", "E"}
SNR_LIST     = [-20.0, 0.0, 20.0]   # received SNR vs thermal noise [dB]

# A fresh random seed is drawn from OS entropy on every launch, so the UE is
# dropped at a NEW random location/heading each run. This single drop is then
# reused for every CDL profile below.
SEED = int(np.random.SeedSequence().generate_state(1)[0])
config.seed = SEED
rng = np.random.default_rng(SEED)
print(f"Random UE drop seed (new each launch) = {SEED}")

# ------------------------ random user location ---------------------------- #
ut_dist = rng.uniform(50.0, 250.0)                      # [m]
ut_az   = rng.uniform(0.0, 2.0 * np.pi)                 # [rad]
ut_xy   = np.array([ut_dist * np.cos(ut_az), ut_dist * np.sin(ut_az)])

# UE yaw points back towards the BS (so its panel faces the serving site).
yaw_to_bs      = np.arctan2(-ut_xy[1], -ut_xy[0])
ut_orientation = torch.tensor([yaw_to_bs, 0.0, 0.0], dtype=torch.float32)

# Random heading for the motion -> velocity vector (constant speed).
heading     = rng.uniform(0.0, 2.0 * np.pi)
velocity    = np.array([SPEED * np.cos(heading), SPEED * np.sin(heading), 0.0])
ut_velocity = torch.tensor(velocity, dtype=torch.float32)

max_doppler = SPEED * CARRIER_FREQUENCY / C0            # [Hz]

# ------------------------------ antenna arrays ---------------------------- #
ut_array = AntennaArray(num_rows=1, num_cols=NUM_UT_ANT // 2,
                        polarization="dual", polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQUENCY)
bs_array = AntennaArray(num_rows=4, num_cols=NUM_BS_ANT // (2 * 4),
                        polarization="dual", polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQUENCY)
assert ut_array.num_ant == NUM_UT_ANT and bs_array.num_ant == NUM_BS_ANT

frequencies = subcarrier_frequencies(FFT_SIZE, SUBCARRIER_SPACING)


# ----------------------------- channel helpers ---------------------------- #
def generate_channel(model_letter):
    """Return the OFDM channel H [NUM_BS_ANT, NUM_UT_ANT, sym, sc] for a profile."""
    cdl = CDL(model=model_letter, delay_spread=DELAY_SPREAD,
              carrier_frequency=CARRIER_FREQUENCY,
              ut_array=ut_array, bs_array=bs_array, direction=DIRECTION,
              ut_orientation=ut_orientation, ut_velocity=ut_velocity)
    a, tau = cdl(batch_size=1, num_time_steps=NUM_OFDM_SYMBOLS,
                 sampling_frequency=SUBCARRIER_SPACING)
    h = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    # h: [batch, num_rx=1, NUM_BS_ANT, num_tx=1, NUM_UT_ANT, sym, sc]
    return h[0, 0, :, 0, :, :, :].detach().cpu().numpy()   # [64, 4, sym, sc]


def cov(H):
    """BS spatial covariance E[h h^H] over (UE antenna, symbol, subcarrier)."""
    M = H.reshape(NUM_BS_ANT, -1)
    return (M @ M.conj().T) / M.shape[1]


# --------------------------- localization window -------------------------- #
def make_location_figure(model_name):
    """One window with the BS, the moving UE and its velocity vector."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f"Localization  -  {model_name}", fontsize=14, fontweight="bold")
    ax.scatter(0, 0, marker="^", s=300, c="k", zorder=5, label="BS (64 ant)")
    ax.scatter(*ut_xy, s=180, c="tab:green", ec="k", zorder=5, label="UE (4 ant)")
    ax.plot([0, ut_xy[0]], [0, ut_xy[1]], c="0.6", ls="--", zorder=2)
    vscale = ut_dist * 0.35 / SPEED
    ax.quiver(ut_xy[0], ut_xy[1], velocity[0] * vscale, velocity[1] * vscale,
              angles="xy", scale_units="xy", scale=1, color="tab:red",
              width=0.012, zorder=6, label=f"velocity ({SPEED:.0f} m/s)")
    lim = ut_dist * 1.4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"Random UE drop  (d = {ut_dist:.0f} m, "
                 f"az = {np.rad2deg(ut_az):.0f} deg,  f_D = {max_doppler:.0f} Hz)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = f"su_los_location_{model_name}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ------------------ bundle of covariance windows (per SNR) ---------------- #
def make_cov_bundle(model_name, R_sig_unit, gain):
    """Grid of received-covariance windows |R_rx| = signal + noise, per SNR."""
    ncol = len(SNR_LIST)
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.6), squeeze=False)
    fig.suptitle(f"BS received covariance |R$_{{rx}}$| = signal + noise"
                 f"   -   {model_name}", fontsize=15, fontweight="bold")
    for c, snr_db in enumerate(SNR_LIST):
        p_sig = 10 ** (snr_db / 10.0) / gain
        R_rx  = p_sig * R_sig_unit + EYE                 # signal + noise (N = 1)
        ax = axes[0, c]
        im = ax.imshow(np.abs(R_rx), cmap="magma")
        ax.set_title(f"{model_name}  |  SNR = {snr_db:.0f} dB", fontsize=10)
        ax.set_xlabel("BS antenna idx"); ax.set_ylabel("BS antenna idx")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|R|")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = f"su_los_cov_{model_name}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ------------- bundle of mixed-channel magnitude windows (per SNR) -------- #
def make_mag_bundle(model_name, H, gain):
    """Grid of mixed received-magnitude windows (target + noise), per SNR.
    Per-(BS antenna, subcarrier) received power = P_sig*|H|^2 + N,  N = 1,
    averaged over UE antennas and OFDM symbols."""
    pwr = np.mean(np.abs(H) ** 2, axis=(1, 2))           # [NUM_BS_ANT, FFT_SIZE]
    ncol = len(SNR_LIST)
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.4), squeeze=False)
    fig.suptitle("Mixed channel magnitude  |target + noise|"
                 f"   -   {model_name}", fontsize=15, fontweight="bold")
    for c, snr_db in enumerate(SNR_LIST):
        p_sig = 10 ** (snr_db / 10.0) / gain
        mixed = p_sig * pwr + 1.0                         # received power, N = 1
        mag_db = 10 * np.log10(mixed + 1e-12)
        ax = axes[0, c]
        im = ax.imshow(mag_db, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, FFT_SIZE, 0, NUM_BS_ANT])
        ax.set_title(f"{model_name}  |  SNR = {snr_db:.0f} dB", fontsize=10)
        ax.set_xlabel("subcarrier idx"); ax.set_ylabel("BS antenna idx")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|mix| [dB]")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = f"su_los_mag_{model_name}.png"
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

for letter in CDL_PROFILES:
    model_name = f"CDL-{letter}"
    H    = generate_channel(letter)            # [NUM_BS_ANT, NUM_UT_ANT, sym, sc]
    gain = float(np.mean(np.abs(H) ** 2))      # linear power gain
    R    = cov(H)                              # BS spatial covariance

    # Save the generated channel tensor (profile-dependent, independent of SNR)
    # as a .npy file. The .npy format is numpy-version-independent, so it loads
    # with any numpy (including numpy==1.26) via np.load(chan_file).
    chan_file = f"su_los_channel_{model_name}.npy"
    np.save(chan_file, H.astype(np.complex64))
    print(f"[{model_name}] H shape {H.shape}  ->  saved channel to {chan_file}")

    loc_png = make_location_figure(model_name)
    cov_png = make_cov_bundle(model_name, R, gain)
    mag_png = make_mag_bundle(model_name, H, gain)
    print(f"    location -> {loc_png}")
    print(f"    covariance bundle ({len(SNR_LIST)} SNR cases) -> {cov_png}")
    print(f"    mixed-magnitude bundle ({len(SNR_LIST)} SNR cases) -> {mag_png}")

n_fig = len(CDL_PROFILES) * 3
print(f"\nDone: {n_fig} figures and {len(CDL_PROFILES)} channel files saved.")
