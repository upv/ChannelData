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
Path loss is enabled, so distant / NLOS interferers contribute less and the
per-interferer INR follows directly from the geometry.

This step only GENERATES THE CHANNEL and VISUALISES the mutual location of the
BS, the served user, and the interferers. (No detection / BER yet.)

Requires: sionna>=2.0 (PyTorch backend), torch, numpy, matplotlib.
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from sionna.phy import config
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMa, UMi, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# ------------------------------- parameters ------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--channel", default="UMa",
                    choices=["UMa", "UMi", "CDL-B", "CDL-C"],
                    help="channel model to generate")
parser.add_argument("--seed", type=int, default=None,
                    help="RNG seed; omit for a new random layout each run")
args, _ = parser.parse_known_args()
CHANNEL_MODEL = args.channel       # one of: UMa, UMi, CDL-B, CDL-C

CARRIER_FREQUENCY  = 3.5e9
SUBCARRIER_SPACING = 30e3
FFT_SIZE           = 76
NUM_OFDM_SYMBOLS   = 14
NUM_BS_ANT         = 64        # serving-BS antennas (4x8 dual-pol panel)
NUM_UT_ANT         = 4         # antennas per user (1x2 dual-pol)

ISD               = 500.0       # inter-site distance [m] (UMa macro)
NUM_INTERFERERS   = 6           # one per first-tier neighbour cell (<= 6)
IOT_DB            = 20.0         # DEFINED interference-over-thermal target [dB]
SNR_DB            = 10.0        # served-user received SNR vs thermal noise [dB]
UT_HEIGHT         = 1.5

# Random layout: use the given seed for reproducibility, otherwise draw a fresh
# one so the served user and interferers are placed at new random positions.
SEED = args.seed if args.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])

# Sweep ranges for the operating-point study (post-combining SINR vs IoT / SNR).
IOT_SWEEP_DB = np.arange(0.0, 30.1, 2.5)    # interference-over-thermal grid [dB]
SNR_SWEEP_DB = np.arange(-10.0, 20.1, 2.5)  # received SNR grid [dB]

# CDL-specific knobs (CDL is a link-level model with NO geometric path loss, so
# the per-link distance attenuation is applied manually using a log-distance law).
DELAY_SPREAD       = 100e-9     # rms delay spread for CDL-B / CDL-C [s]
PATH_LOSS_EXPONENT = 3.5        # path-loss exponent for the CDL geometry scaling

IS_CDL    = CHANNEL_MODEL.startswith("CDL")
BS_HEIGHT = 10.0 if CHANNEL_MODEL == "UMi" else 25.0   # UMi micro vs UMa macro

config.seed = SEED
rng = np.random.default_rng(SEED)
print(f"[{CHANNEL_MODEL}] random layout seed = {SEED}")
NUM_UT   = 1 + NUM_INTERFERERS   # UT 0 = served user
R_CELL   = ISD / np.sqrt(3.0)    # hex centre-to-vertex radius

# ---------------------------- geometry (top-down) ------------------------- #
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


def make_system_channel(model_cls):
    """UMa / UMi: geometry-driven, multi-UT topology with built-in path loss."""
    model = model_cls(carrier_frequency=CARRIER_FREQUENCY, o2i_model="low",
                      ut_array=ut_array, bs_array=bs_array, direction="uplink",
                      enable_pathloss=True, enable_shadow_fading=True)
    model.set_topology(
        ut_loc          = col3(ut_xy, UT_HEIGHT),
        bs_loc          = col3(serving_bs[None], BS_HEIGHT),
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


if IS_CDL:
    a, tau = make_cdl_channel(CHANNEL_MODEL.split("-")[1])
else:
    a, tau = make_system_channel(UMi if CHANNEL_MODEL == "UMi" else UMa)

h = cir_to_ofdm_channel(frequencies, a, tau, normalize=False).detach().cpu().numpy()
# h: [1, num_rx=1, NUM_BS_ANT, NUM_UT, 1, NUM_OFDM_SYMBOLS, FFT_SIZE]
H = [h[0, 0, :, u, :, :, :] for u in range(NUM_UT)]   # per-UT [Nbs, Nut, sym, sc]
gain = np.array([np.mean(np.abs(Hu)**2) for Hu in H]) # linear power gain (incl. PL)
print(f"[{CHANNEL_MODEL}] generated channel tensor h with shape:", h.shape)

# ----------------- calibrate interferer power to defined IoT -------------- #
# Normalise thermal noise to N = 1. With equal interferer Tx power P:
#   I = P * sum(gain_interferers),   IoT = (I + N)/N  ->  pick P for target IoT.
g_int  = gain[1:]
I_lin  = 10 ** (IOT_DB / 10.0) - 1.0
P_int  = I_lin / np.sum(g_int)                 # common interferer Tx power (rel.)
INR    = P_int * g_int                         # per-interferer I_i / N (linear)
IoT_achieved = 10 * np.log10(1.0 + np.sum(INR))

print(f"Distances to serving BS [m]: {np.round(dist, 1)}")
print(f"Per-interferer INR [dB]:     {np.round(10*np.log10(INR), 1)}")
print(f"Target IoT = {IOT_DB:.1f} dB  ->  achieved IoT = {IoT_achieved:.2f} dB")

# ------------------ spatial covariance + magnitude + eigenmodes ----------- #
# Per-UT BS spatial covariance  R_u = E[h h^H]  over (symbol, subcarrier).
def cov(Hu):
    M = Hu.reshape(NUM_BS_ANT, -1)
    return (M @ M.conj().T) / M.shape[1]

R_ut       = [cov(Hu) for Hu in H]
R_int_unit = sum(R_ut[j] for j in range(1, NUM_UT))   # unit-Tx interferer cov sum
EYE        = np.eye(NUM_BS_ANT)

def cov_signal(snr_db):
    """Desired-signal covariance scaled to the wanted RECEIVED SNR (N = 1)."""
    return (10 ** (snr_db / 10.0) / gain[0]) * R_ut[0]

def cov_intf_noise(iot_db):
    """Interference-plus-noise covariance scaled to a target IoT (N = 1)."""
    p_int = (10 ** (iot_db / 10.0) - 1.0) / np.sum(g_int)
    return p_int * R_int_unit + EYE

R_sig = cov_signal(SNR_DB)                        # desired-signal covariance
R_in  = cov_intf_noise(IOT_DB)                    # interference + noise covariance
# Total received spatial covariance the BS array observes: signal + interf + noise.
R_rx  = R_sig + R_in

# Served-user channel magnitude |H| over (BS antenna, subcarrier), averaged
# across UT antennas and OFDM symbols  ->  [NUM_BS_ANT, FFT_SIZE].
H_mag = np.mean(np.abs(H[0]), axis=(1, 2))
H_mag_db = 20 * np.log10(H_mag + 1e-12)

# Eigenmodes (eigenvalue spectra) of the BS spatial covariance matrices.
ev_rx  = np.clip(np.linalg.eigvalsh(R_rx)[::-1],  0.0, None)   # descending
ev_sig = np.clip(np.linalg.eigvalsh(R_sig)[::-1], 0.0, None)
ev_in  = np.clip(np.linalg.eigvalsh(R_in)[::-1],  0.0, None)

# ----------------- operating-point sweep over IoT and SNR ----------------- #
# Optimal (MMSE / max-SINR) post-combining output SINR for the served user is
# the largest generalized eigenvalue of (R_sig, R_in):  max eig of R_in^-1 R_sig.
def mmse_out_sinr_db(snr_db, iot_db):
    Rs = cov_signal(snr_db)
    Ri = cov_intf_noise(iot_db)
    ev = np.linalg.eigvals(np.linalg.solve(Ri, Rs)).real
    return 10.0 * np.log10(max(ev.max(), 1e-12))

sinr_grid = np.array([[mmse_out_sinr_db(snr, iot) for snr in SNR_SWEEP_DB]
                      for iot in IOT_SWEEP_DB])    # [len(IoT), len(SNR)]

print("\nPost-combining output SINR [dB] over the IoT/SNR grid:")
print("  IoT\\SNR  " + "  ".join(f"{s:6.1f}" for s in SNR_SWEEP_DB))
for i, iot in enumerate(IOT_SWEEP_DB):
    print(f"  {iot:6.1f}  " + "  ".join(f"{v:6.1f}" for v in sinr_grid[i]))

np.savez("su_intercell_channel.npz", channel_model=CHANNEL_MODEL, seed=SEED,
         h=h, ut_xy=ut_xy, serving_bs=serving_bs, neigh_bs=neigh_bs, gain=gain,
         INR=INR, R_sig=R_sig, R_in=R_in, R_rx=R_rx, H_mag=H_mag,
         ev_rx=ev_rx, ev_sig=ev_sig, ev_in=ev_in,
         iot_sweep_db=IOT_SWEEP_DB, snr_sweep_db=SNR_SWEEP_DB, sinr_grid=sinr_grid)

# -------------------------------- visualise ------------------------------- #
def hexagon(cx, cy, R):
    ang = np.deg2rad(30 + 60*np.arange(6))
    return np.stack([cx + R*np.cos(ang), cy + R*np.sin(ang)], axis=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axm, axg, axc, axe = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
fig.suptitle(f"Single-user uplink with inter-cell interference  -  {CHANNEL_MODEL}",
             fontsize=14, fontweight="bold")

# --- localization: mutual location of BS, served user, interferers ---
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

axm.set_title("Mutual location: serving BS, served UE, inter-cell interferers")
axm.set_xlabel("x [m]"); axm.set_ylabel("y [m]")
axm.set_aspect("equal"); axm.grid(alpha=0.3); axm.legend(loc="upper right")

# --- magnitude: served-user channel |H| over BS antenna x subcarrier ---
img = axg.imshow(H_mag_db, aspect="auto", origin="lower", cmap="viridis",
                 extent=[0, FFT_SIZE, 0, NUM_BS_ANT])
axg.set_title("Served-user channel magnitude |H|\n"
              "(averaged over UT antennas and OFDM symbols)")
axg.set_xlabel("subcarrier index"); axg.set_ylabel("BS antenna index")
fig.colorbar(img, ax=axg, label="|H|  [dB]")

# --- total received spatial covariance at the serving BS ---
im = axc.imshow(np.abs(R_rx), cmap="magma")
vmax = np.abs(R_rx).max()
if NUM_BS_ANT <= 8:                                  # annotate values only if small
    for i in range(NUM_BS_ANT):
        for k in range(NUM_BS_ANT):
            val = np.abs(R_rx[i, k])
            axc.text(k, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                     color="w" if val < 0.6 * vmax else "k")
axc.set_title("BS received covariance "
              f"|R$_{{rx}}$| = signal + interf + noise  ({NUM_BS_ANT}×{NUM_BS_ANT})\n"
              f"SNR = {SNR_DB:.0f} dB,  achieved IoT = {IoT_achieved:.2f} dB,  N=1")
axc.set_xlabel("BS antenna index"); axc.set_ylabel("BS antenna index")
fig.colorbar(im, ax=axc, label="|R|  (linear, N=1)")

# --- eigenmodes: eigenvalue spectra of the BS spatial covariance matrices ---
idx = np.arange(1, NUM_BS_ANT + 1)
axe.plot(idx, 10 * np.log10(ev_rx + 1e-12), "o-", ms=3, lw=1.0,
         color="tab:purple", label="R$_{rx}$ (signal+interf+noise)")
axe.plot(idx, 10 * np.log10(ev_sig + 1e-12), "s-", ms=3, lw=1.0,
         color="tab:green", label="R$_{sig}$ (desired signal)")
axe.plot(idx, 10 * np.log10(ev_in + 1e-12), "^-", ms=3, lw=1.0,
         color="tab:red", label="R$_{in}$ (interf+noise)")
axe.set_title("BS covariance eigenmodes\n"
              f"SNR = {SNR_DB:.0f} dB,  achieved IoT = {IoT_achieved:.2f} dB,  N=1")
axe.set_xlabel("eigenmode index (sorted)"); axe.set_ylabel("eigenvalue [dB]")
axe.grid(alpha=0.3); axe.legend(fontsize=8, loc="upper right")

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = f"su_intercell_layout_{CHANNEL_MODEL}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved figure to {out_png}")

# ------------------- operating-point sweep visualisation ------------------ #
fig2, (ax_hm, ax_ln) = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle(f"Post-combining output SINR vs IoT / SNR  -  {CHANNEL_MODEL}",
              fontsize=14, fontweight="bold")

# --- heatmap of optimal output SINR over the (SNR, IoT) grid ---
im2 = ax_hm.imshow(sinr_grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[SNR_SWEEP_DB[0], SNR_SWEEP_DB[-1],
                           IOT_SWEEP_DB[0], IOT_SWEEP_DB[-1]])
ax_hm.scatter([SNR_DB], [IOT_DB], marker="*", s=220, c="red", ec="k", zorder=5,
              label=f"operating point\n(SNR={SNR_DB:.0f}, IoT={IOT_DB:.0f} dB)")
ax_hm.set_title("Optimal MMSE output SINR [dB]")
ax_hm.set_xlabel("received SNR [dB]"); ax_hm.set_ylabel("IoT [dB]")
ax_hm.legend(fontsize=8, loc="lower right")
fig2.colorbar(im2, ax=ax_hm, label="output SINR [dB]")

# --- line cuts: output SINR vs SNR for a few representative IoT values ---
iot_cuts = IOT_SWEEP_DB[::max(1, len(IOT_SWEEP_DB) // 5)]
for iot in iot_cuts:
    sinr_line = [mmse_out_sinr_db(snr, iot) for snr in SNR_SWEEP_DB]
    ax_ln.plot(SNR_SWEEP_DB, sinr_line, "o-", ms=3, lw=1.2, label=f"IoT = {iot:.1f} dB")
ax_ln.set_title("Output SINR vs SNR for varying IoT")
ax_ln.set_xlabel("received SNR [dB]"); ax_ln.set_ylabel("output SINR [dB]")
ax_ln.grid(alpha=0.3); ax_ln.legend(fontsize=8, loc="upper left", title="interference")

fig2.tight_layout(rect=[0, 0, 1, 0.96])
out_png2 = f"su_intercell_sinr_sweep_{CHANNEL_MODEL}.png"
fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
print(f"Saved figure to {out_png2}")
plt.show()
