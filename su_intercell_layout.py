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

Requires: sionna>=1.0, tensorflow, numpy, matplotlib.
"""

import numpy as np
import tensorflow as tf
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
NUM_BS_ANT         = 4

ISD               = 500.0       # inter-site distance [m] (UMa macro)
NUM_INTERFERERS   = 6           # one per first-tier neighbour cell (<= 6)
IOT_DB            = 6.0         # DEFINED interference-over-thermal target [dB]
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
ut_array = Antenna(polarization="single", polarization_type="V",
                   antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY)
bs_array = AntennaArray(num_rows=1, num_cols=NUM_BS_ANT // 2,
                        polarization="dual", polarization_type="cross",
                        antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY)

# ------------------- UMa with one (serving) BS as receiver ---------------- #
uma = UMa(carrier_frequency=CARRIER_FREQUENCY, o2i_model="low",
          ut_array=ut_array, bs_array=bs_array, direction="uplink",
          enable_pathloss=True, enable_shadow_fading=True)

def col3(xy, z):                                     # -> [1, n, 3] float32
    arr = np.concatenate([xy, np.full((xy.shape[0], 1), z)], axis=1)
    return tf.constant(arr[None], tf.float32)

uma.set_topology(
    ut_loc          = col3(ut_xy, UT_HEIGHT),
    bs_loc          = col3(serving_bs[None], BS_HEIGHT),
    ut_orientations = tf.zeros([1, NUM_UT, 3], tf.float32),
    bs_orientations = tf.zeros([1, 1, 3], tf.float32),
    ut_velocities   = tf.zeros([1, NUM_UT, 3], tf.float32),
    in_state        = tf.zeros([1, NUM_UT], tf.bool))     # all outdoor

# --------------------------- generate the channel ------------------------- #
frequencies = subcarrier_frequencies(FFT_SIZE, SUBCARRIER_SPACING)
fs = SUBCARRIER_SPACING * FFT_SIZE
a, tau = uma(NUM_OFDM_SYMBOLS, fs)
h = cir_to_ofdm_channel(frequencies, a, tau, normalize=False).numpy()
# h: [1, num_rx=1, NUM_BS_ANT, NUM_UT, 1, NUM_OFDM_SYMBOLS, FFT_SIZE]
H = [h[0, 0, :, u, 0, :, :] for u in range(NUM_UT)]   # per-UT [Nant, sym, sc]
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

np.savez("su_intercell_channel.npz", h=h, ut_xy=ut_xy, serving_bs=serving_bs,
         neigh_bs=neigh_bs, gain=gain, INR=INR)

# -------------------------------- visualise ------------------------------- #
def hexagon(cx, cy, R):
    ang = np.deg2rad(30 + 60*np.arange(6))
    return np.stack([cx + R*np.cos(ang), cy + R*np.sin(ang)], axis=1)

fig, (axm, axb) = plt.subplots(1, 2, figsize=(15, 7),
                               gridspec_kw={"width_ratios": [1.5, 1]})

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

fig.tight_layout()
fig.savefig("su_intercell_layout.png", dpi=150, bbox_inches="tight")
print("Saved figure to su_intercell_layout.png")
plt.show()
