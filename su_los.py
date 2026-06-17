"""
Single-user MIMO-OFDM channel with a MOVING user (CDL-B), modern Sionna 2.0.

Setup:
  * One UE with  4 antennas (1x2 dual-polarised panel).
  * One BS with 64 antennas (4x8 dual-polarised panel)  -> massive-MIMO array.
  * OFDM grid: FFT_SIZE = 128 subcarriers, NUM_OFDM_SYMBOLS = 512 symbols.
  * 3GPP TR 38.901 CDL-B clustered-delay-line model, uplink (UE -> BS).
  * The user is MOVING (constant speed, random heading) -> time-varying channel
    with a non-trivial Doppler spectrum.

The user is dropped at a RANDOM location around the BS. From the realised
channel we ESTIMATE:
  * the channel character     (power-delay profile, RMS delay spread,
                               coherence bandwidth, max Doppler, coherence time);
  * the 64x64 BS spatial COVARIANCE matrix R = E[h h^H] and its eigen-structure
                              (effective rank / spatial correlation).

The CDL profile is selectable on the command line, e.g.:
    python su_los.py --model B      # CDL-B (NLOS, default)
    python su_los.py --model C      # CDL-C (NLOS)
    python su_los.py --model D      # CDL-D (LOS, dominant direct ray)
CDL-A/B/C are NLOS profiles; CDL-D/E carry a dedicated LOS ray.

Requires: sionna>=2.0 (PyTorch backend), torch, numpy, matplotlib.
"""

import argparse

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

CDL_MODEL          = "E"            # default profile (override with --model)
DELAY_SPREAD       = 100e-9         # RMS delay spread of the profile [s]
DIRECTION          = "uplink"       # UE -> BS, so the BS (64) is the receiver
SPEED              = 30.0           # UE speed [m/s] (~108 km/h) -> Doppler
BS_HEIGHT          = 25.0
UT_HEIGHT          = 1.5
SEED               = 42

C0 = 299_792_458.0                  # speed of light [m/s]

# CDL-A/B/C are NLOS profiles; CDL-D/E contain a dedicated LOS ray.
LOS_MODELS = {"D", "E"}

# ----------------------------- CLI overrides ------------------------------ #
parser = argparse.ArgumentParser(
    description="Single-user moving CDL MIMO-OFDM channel (Sionna 2.0).")
parser.add_argument("--model", choices=["A", "B", "C", "D", "E"],
                    default=CDL_MODEL,
                    help="CDL profile to use, e.g. B, C, D (default: %(default)s).")
parser.add_argument("--speed", type=float, default=SPEED,
                    help="UE speed in m/s (default: %(default)s).")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: random each run; "
                         "pass an int to reproduce a specific UE drop).")
args, _ = parser.parse_known_args()
CDL_MODEL, SPEED = args.model, args.speed
# Random placement every run unless a seed is given. The drawn seed is printed
# so any interesting drop can be reproduced with --seed.
SEED = args.seed if args.seed is not None \
       else int(np.random.default_rng().integers(0, 2**31 - 1))
IS_LOS = CDL_MODEL in LOS_MODELS
print(f"Using CDL-{CDL_MODEL} "
      f"({'LOS' if IS_LOS else 'NLOS'}), speed = {SPEED:.1f} m/s, seed = {SEED}")

config.seed = SEED
rng = np.random.default_rng(SEED)

# ------------------------ random user location ---------------------------- #
# BS at the origin; drop the UE at a random distance / azimuth around it.
ut_dist = rng.uniform(50.0, 250.0)                      # [m]
ut_az   = rng.uniform(0.0, 2.0 * np.pi)                 # [rad]
ut_xy   = np.array([ut_dist * np.cos(ut_az),
                    ut_dist * np.sin(ut_az)])

# UE yaw points back towards the BS (so its panel faces the serving site).
yaw_to_bs = np.arctan2(-ut_xy[1], -ut_xy[0])
ut_orientation = torch.tensor([yaw_to_bs, 0.0, 0.0], dtype=torch.float32)

# Random heading for the motion -> velocity vector (constant speed).
heading  = rng.uniform(0.0, 2.0 * np.pi)
velocity = np.array([SPEED * np.cos(heading), SPEED * np.sin(heading), 0.0])
ut_velocity = torch.tensor(velocity, dtype=torch.float32)

max_doppler    = SPEED * CARRIER_FREQUENCY / C0          # [Hz]
ofdm_symbol_T  = 1.0 / SUBCARRIER_SPACING               # ~ symbol duration [s]
frame_duration = NUM_OFDM_SYMBOLS * ofdm_symbol_T       # [s]

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

# ------------------------------- CDL-B model ------------------------------ #
cdl = CDL(model=CDL_MODEL, delay_spread=DELAY_SPREAD,
          carrier_frequency=CARRIER_FREQUENCY,
          ut_array=ut_array, bs_array=bs_array, direction=DIRECTION,
          ut_orientation=ut_orientation, ut_velocity=ut_velocity)

# ----------------------------- generate channel --------------------------- #
frequencies = subcarrier_frequencies(FFT_SIZE, SUBCARRIER_SPACING)
# One time step per OFDM symbol -> sampling frequency = symbol rate = SCS.
a, tau = cdl(batch_size=1, num_time_steps=NUM_OFDM_SYMBOLS,
             sampling_frequency=SUBCARRIER_SPACING)
h = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
# h: [batch, num_rx=1, NUM_BS_ANT, num_tx=1, NUM_UT_ANT, NUM_OFDM_SYMBOLS, FFT_SIZE]
H = h[0, 0, :, 0, :, :, :].detach().cpu().numpy()       # [64, 4, 512, 128]
print("Generated channel tensor h with shape:", tuple(h.shape))

# --------------------- power-delay profile (channel type) ----------------- #
a_np   = a.detach().cpu().numpy()                       # [...,num_paths,time]
pdp    = np.mean(np.abs(a_np) ** 2, axis=(0, 1, 2, 3, 4, 6))   # per path
tau_s  = tau.detach().cpu().numpy()[0, 0, 0, :]         # [num_paths] in seconds
order  = np.argsort(tau_s)
tau_s, pdp = tau_s[order], pdp[order]
pdp_n  = pdp / pdp.sum()

mean_delay = np.sum(pdp_n * tau_s)
rms_ds     = np.sqrt(np.sum(pdp_n * (tau_s - mean_delay) ** 2))
coh_bw     = 1.0 / (2.0 * np.pi * rms_ds)               # coherence bandwidth [Hz]
coh_time   = 0.423 / max_doppler if max_doppler > 0 else np.inf
total_bw   = FFT_SIZE * SUBCARRIER_SPACING

# Sampled power-delay profile as seen through the OFDM grid: the IDFT across the
# FFT_SIZE subcarriers yields FFT_SIZE delay bins (the resolvable taps). Delay
# resolution = 1/(N*SCS); total span = 1/SCS (the OFDM symbol duration).
cir        = np.fft.ifft(H, axis=-1)                    # [64,4,512,FFT_SIZE]
pdp_samp   = np.mean(np.abs(cir) ** 2, axis=(0, 1, 2))  # [FFT_SIZE] delay bins
pdp_samp_n = pdp_samp / pdp_samp.max()
delay_res  = 1.0 / (FFT_SIZE * SUBCARRIER_SPACING)      # seconds per bin
delay_bins = np.arange(FFT_SIZE) * delay_res            # [FFT_SIZE] seconds

# --------------------- BS spatial covariance (64 x 64) -------------------- #
# Stack every (UE-antenna, symbol, subcarrier) realisation as a 64-dim sample.
Hm   = H.transpose(0, 1, 2, 3).reshape(NUM_BS_ANT, -1)  # [64, N]
Nsmp = Hm.shape[1]
R    = (Hm @ Hm.conj().T) / Nsmp                        # [64, 64] Hermitian
R    = R / np.trace(R).real * NUM_BS_ANT                # normalise: trace = Nant

eigvals = np.linalg.eigvalsh(R)[::-1].real              # descending
eigvals = np.clip(eigvals, 1e-12, None)
p_eig   = eigvals / eigvals.sum()
eff_rank = float(np.exp(-np.sum(p_eig * np.log(p_eig))))   # entropy-based rank
rank90   = int(np.searchsorted(np.cumsum(p_eig), 0.90) + 1)
cond_db  = 10.0 * np.log10(eigvals[0] / eigvals[-1])

# ------------------------------- estimate / classify ---------------------- #
freq_sel = "frequency-selective" if coh_bw < total_bw else "frequency-flat"
time_sel = "fast-fading (time-selective)" if coh_time < frame_duration \
           else "slow-fading"
spatial  = ("highly correlated" if eff_rank < NUM_BS_ANT / 4 else
            "moderately correlated" if eff_rank < NUM_BS_ANT / 2 else
            "weakly correlated")

print("\n================ user / channel estimate ================")
print(f"UE location           : ({ut_xy[0]:+.1f}, {ut_xy[1]:+.1f}) m   "
      f"(d = {ut_dist:.1f} m, az = {np.rad2deg(ut_az):.1f} deg)")
print(f"UE speed / heading    : {SPEED:.1f} m/s @ {np.rad2deg(heading):.1f} deg")
print(f"Model                 : CDL-{CDL_MODEL} "
      f"({'LOS' if IS_LOS else 'NLOS'}, {DIRECTION}), "
      f"DS = {DELAY_SPREAD*1e9:.0f} ns")
print(f"RMS delay spread      : {rms_ds*1e9:.1f} ns")
print(f"Coherence bandwidth   : {coh_bw/1e3:.1f} kHz  (total BW = "
      f"{total_bw/1e6:.2f} MHz) -> {freq_sel}")
print(f"Max Doppler           : {max_doppler:.1f} Hz")
print(f"Coherence time        : {coh_time*1e3:.2f} ms (frame = "
      f"{frame_duration*1e3:.2f} ms) -> {time_sel}")
print(f"Cov. effective rank   : {eff_rank:.1f} / {NUM_BS_ANT}  "
      f"(90% energy in {rank90} eigenmodes) -> {spatial}")
print(f"Cov. condition number : {cond_db:.1f} dB")
print("=========================================================")

npz_name = f"su_los_channel_{CDL_MODEL}.npz"
np.savez(npz_name,
         h_slice=H[:, :, 0, :], R=R, eigvals=eigvals,
         tau_s=tau_s, pdp=pdp, pdp_sampled=pdp_samp, delay_bins=delay_bins,
         ut_xy=ut_xy, velocity=velocity,
         max_doppler=max_doppler, rms_ds=rms_ds, coh_bw=coh_bw)

# -------------------------------- visualise ------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax = axes.ravel()

# (1) random user location + motion ---------------------------------------
ax[0].scatter(0, 0, marker="^", s=300, c="k", zorder=5, label="BS (64 ant)")
ax[0].scatter(*ut_xy, s=180, c="tab:green", ec="k", zorder=5,
              label="UE (4 ant)")
ax[0].plot([0, ut_xy[0]], [0, ut_xy[1]], c="0.6", ls="--", zorder=2)
vscale = ut_dist * 0.35 / SPEED
ax[0].quiver(ut_xy[0], ut_xy[1], velocity[0] * vscale, velocity[1] * vscale,
             angles="xy", scale_units="xy", scale=1, color="tab:red",
             width=0.012, zorder=6, label=f"velocity ({SPEED:.0f} m/s)")
lim = ut_dist * 1.4
ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3)
ax[0].set_title(f"Random UE drop  (d = {ut_dist:.0f} m, "
                f"az = {np.rad2deg(ut_az):.0f} deg)")
ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]"); ax[0].legend(loc="upper right")

# (2) sampled power-delay profile (FFT_SIZE delay bins from the OFDM grid) --
ax[1].plot(delay_bins * 1e6, 10 * np.log10(pdp_samp_n + 1e-12),
           "o-", ms=3, lw=1.0, color="tab:blue",
           label=f"sampled CIR ({FFT_SIZE} bins, Δτ={delay_res*1e9:.0f} ns)")
ax[1].scatter(tau_s * 1e6, 10 * np.log10(pdp / pdp.max() + 1e-12),
              marker="x", s=40, color="tab:red", zorder=5,
              label=f"CDL-{CDL_MODEL} clusters ({len(tau_s)})")
ax[1].set_ylim(-60, 3)
ax[1].set_title(f"Power-delay profile  ({FFT_SIZE} delay bins)\n"
                f"RMS DS = {rms_ds*1e9:.1f} ns,  Bc ≈ {coh_bw/1e3:.0f} kHz")
ax[1].set_xlabel("delay [µs]"); ax[1].set_ylabel("relative power [dB]")
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="upper right")

# (3) time-frequency channel magnitude (one antenna pair) ------------------
Htf = 20 * np.log10(np.abs(H[0, 0, :, :]) + 1e-12)      # [symbols, subcarriers]
im2 = ax[2].imshow(Htf, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, FFT_SIZE, 0, NUM_OFDM_SYMBOLS])
ax[2].set_title("|H| over time-frequency  (BS ant 0, UE ant 0)")
ax[2].set_xlabel("subcarrier"); ax[2].set_ylabel("OFDM symbol (time)")
fig.colorbar(im2, ax=ax[2], label="|H| [dB]")

# (4) BS spatial covariance magnitude --------------------------------------
im3 = ax[3].imshow(np.abs(R), cmap="magma")
ax[3].set_title("BS spatial covariance |R|  (64 x 64)")
ax[3].set_xlabel("antenna index"); ax[3].set_ylabel("antenna index")
fig.colorbar(im3, ax=ax[3], label="|R|")

# (5) covariance eigenvalue spectrum ---------------------------------------
ax[4].plot(10 * np.log10(eigvals), "o-", ms=4, c="tab:purple")
ax[4].axvline(eff_rank - 1, color="tab:red", ls="--",
              label=f"eff. rank ≈ {eff_rank:.1f}")
ax[4].set_title("Covariance eigenvalue spectrum")
ax[4].set_xlabel("eigenmode index"); ax[4].set_ylabel("eigenvalue [dB]")
ax[4].grid(alpha=0.3); ax[4].legend()

# (6) Doppler spectrum (one antenna pair, one subcarrier) ------------------
x   = H[0, 0, :, FFT_SIZE // 2]                          # time series [512]
psd = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
psd = psd / psd.max()
fdax = np.fft.fftshift(np.fft.fftfreq(NUM_OFDM_SYMBOLS, d=ofdm_symbol_T))
ax[5].plot(fdax, 10 * np.log10(psd + 1e-12), c="tab:orange")
ax[5].axvline(+max_doppler, color="k", ls="--", alpha=0.7,
              label=f"±f_D = {max_doppler:.0f} Hz")
ax[5].axvline(-max_doppler, color="k", ls="--", alpha=0.7)
ax[5].set_xlim(-3 * max_doppler, 3 * max_doppler)
ax[5].set_title("Doppler spectrum  (BS ant 0, UE ant 0)")
ax[5].set_xlabel("Doppler frequency [Hz]"); ax[5].set_ylabel("PSD [dB]")
ax[5].grid(alpha=0.3); ax[5].legend()

fig.suptitle(f"Single-user CDL-{CDL_MODEL} channel  |  "
             f"{freq_sel}, {time_sel}, {spatial} array "
             f"(eff. rank {eff_rank:.1f}/{NUM_BS_ANT})",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
png_name = f"su_los_channel_{CDL_MODEL}.png"
fig.savefig(png_name, dpi=150, bbox_inches="tight")
print(f"Saved figure to {png_name}  (data: {npz_name})")
plt.show()
