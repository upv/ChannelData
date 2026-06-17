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

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from sionna.phy import config
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMa
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
IOT_DB            = 20.0         # DEFINED interference-over-thermal target [dB]
SNR_DB            = 10.0        # served-user received SNR vs thermal noise [dB]
BS_HEIGHT         = 25.0
UT_HEIGHT         = 1.5
SEED              = 42

config.seed = SEED
rng = np.random.default_rng(SEED)
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

# ------------------- UMa with one (serving) BS as receiver ---------------- #
uma = UMa(carrier_frequency=CARRIER_FREQUENCY, o2i_model="low",
          ut_array=ut_array, bs_array=bs_array, direction="uplink",
          enable_pathloss=True, enable_shadow_fading=True)

def col3(xy, z):                                     # -> [1, n, 3] float32
    arr = np.concatenate([xy, np.full((xy.shape[0], 1), z)], axis=1)
    return torch.tensor(arr[None], dtype=torch.float32)

uma.set_topology(
    ut_loc          = col3(ut_xy, UT_HEIGHT),
    bs_loc          = col3(serving_bs[None], BS_HEIGHT),
    ut_orientations = torch.zeros([1, NUM_UT, 3], dtype=torch.float32),
    bs_orientations = torch.zeros([1, 1, 3], dtype=torch.float32),
    ut_velocities   = torch.zeros([1, NUM_UT, 3], dtype=torch.float32),
    in_state        = torch.zeros([1, NUM_UT], dtype=torch.bool))   # all outdoor

# --------------------------- generate the channel ------------------------- #
frequencies = subcarrier_frequencies(FFT_SIZE, SUBCARRIER_SPACING)
fs = SUBCARRIER_SPACING * FFT_SIZE
a, tau = uma(NUM_OFDM_SYMBOLS, fs)
h = cir_to_ofdm_channel(frequencies, a, tau, normalize=False).detach().cpu().numpy()
# h: [1, num_rx=1, NUM_BS_ANT, NUM_UT, 1, NUM_OFDM_SYMBOLS, FFT_SIZE]
H = [h[0, 0, :, u, :, :, :] for u in range(NUM_UT)]   # per-UT [Nbs, Nut, sym, sc]
gain = np.array([np.mean(np.abs(Hu)**2) for Hu in H]) # linear power gain (incl. PL)
print("Generated channel tensor h with shape:", h.shape)

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

# --------------- spatial covariance + power-delay profile ----------------- #
# Per-UT BS spatial covariance  R_u = E[h h^H]  over (symbol, subcarrier).
def cov(Hu):
    M = Hu.reshape(NUM_BS_ANT, -1)
    return (M @ M.conj().T) / M.shape[1]

R_ut  = [cov(Hu) for Hu in H]
# Served-user transmit power set so the desired RECEIVED SNR = SNR_DB (N = 1).
P_sig = 10 ** (SNR_DB / 10.0) / gain[0]
R_sig = P_sig * R_ut[0]                           # desired-signal covariance (scaled)
# Interference-plus-noise covariance at the serving BS (thermal noise N = 1),
# using the IoT-calibrated common interferer power P_int.
R_in  = sum(P_int * R_ut[j] for j in range(1, NUM_UT)) + np.eye(NUM_BS_ANT)
# Total received spatial covariance the BS array observes: signal + interf + noise.
R_rx  = R_sig + R_in

# Served-user power-delay profile: true UMa clusters + OFDM-sampled CIR.
a_np   = a.detach().cpu().numpy()                 # [1,1,Nant,NUM_UT,1,paths,time]
tau_np = tau.detach().cpu().numpy()              # [1,1,NUM_UT,paths]
pdp_cl = np.mean(np.abs(a_np[0, 0, :, 0, :, :, :]) ** 2, axis=(0, 1, 3))  # [paths]
tau_cl = tau_np[0, 0, 0, :]
m      = tau_cl >= 0.0
tau_cl, pdp_cl = tau_cl[m], pdp_cl[m]
o      = np.argsort(tau_cl)
tau_cl, pdp_cl = tau_cl[o], pdp_cl[o]
# OFDM-resolvable CIR: IDFT across the FFT_SIZE subcarriers -> FFT_SIZE bins.
cir_s      = np.fft.ifft(H[0], axis=-1)          # [Nbs, Nut, sym, FFT_SIZE]
pdp_samp   = np.mean(np.abs(cir_s) ** 2,
                     axis=tuple(range(cir_s.ndim - 1)))   # [FFT_SIZE]
pdp_samp_n = pdp_samp / pdp_samp.max()
delay_res  = 1.0 / (FFT_SIZE * SUBCARRIER_SPACING)      # seconds per bin
delay_bins = np.arange(FFT_SIZE) * delay_res

np.savez("su_intercell_channel.npz", h=h, ut_xy=ut_xy, serving_bs=serving_bs,
         neigh_bs=neigh_bs, gain=gain, INR=INR, R_sig=R_sig, R_in=R_in, R_rx=R_rx,
         tau_cl=tau_cl, pdp_cl=pdp_cl, pdp_sampled=pdp_samp, delay_bins=delay_bins)

# -------------------------------- visualise ------------------------------- #
def hexagon(cx, cy, R):
    ang = np.deg2rad(30 + 60*np.arange(6))
    return np.stack([cx + R*np.cos(ang), cy + R*np.sin(ang)], axis=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axm, axb, axc, axp = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# --- map: mutual location of BS, served user, interferers ---
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

# --- per-interferer INR bar (ties geometry to the defined IoT) ---
order = np.argsort(dist[1:])
axb.bar(range(NUM_INTERFERERS), 10*np.log10(INR[order]), color="tab:red", alpha=0.8)
axb.set_xticks(range(NUM_INTERFERERS))
axb.set_xticklabels([f"{dist[1:][k]:.0f} m" for k in order], rotation=45, fontsize=8)
axb.set_title(f"Per-interferer INR\n(defined IoT = {IOT_DB:.1f} dB, "
              f"achieved = {IoT_achieved:.2f} dB)")
axb.set_xlabel("interferer (by distance to serving BS)")
axb.set_ylabel("INR = I$_i$/N  [dB]")
axb.grid(alpha=0.3, axis="y")

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

# --- served-user power-delay profile (FFT_SIZE delay bins + clusters) ---
axp.plot(delay_bins * 1e6, 10 * np.log10(pdp_samp_n + 1e-12), "o-", ms=3, lw=1.0,
         color="tab:green",
         label=f"sampled CIR ({FFT_SIZE} bins, Δτ={delay_res*1e9:.0f} ns)")
axp.scatter(tau_cl * 1e6, 10 * np.log10(pdp_cl / pdp_cl.max() + 1e-12),
            marker="x", s=40, color="tab:blue", zorder=5,
            label=f"UMa clusters ({len(tau_cl)})")
axp.set_ylim(-60, 3)
axp.set_title("Served-user power-delay profile")
axp.set_xlabel("delay [µs]"); axp.set_ylabel("relative power [dB]")
axp.grid(alpha=0.3); axp.legend(fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig("su_intercell_layout.png", dpi=150, bbox_inches="tight")
print("Saved figure to su_intercell_layout.png")
plt.show()
