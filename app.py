import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import io

# =========================
# LANCZOS INTERPOLATION
# =========================

def lanczos_kernel(x, a=3):
    x = np.array(x, dtype=float)

    out = np.sinc(x) * np.sinc(x / a)

    out[np.abs(x) >= a] = 0

    return out


def lanczos_resample(sample_pos, sampled, full_pos, a=3):

    reconstructed = np.zeros(len(full_pos))

    for i, t in enumerate(full_pos):

        x = (t - sample_pos) / np.mean(np.diff(sample_pos))

        weights = lanczos_kernel(x, a)

        s = np.sum(weights)

        if abs(s) > 1e-12:
            reconstructed[i] = np.sum(sampled * weights) / s

    return reconstructed


# =========================
# WINDOWED SINC
# =========================

def sinc_resample(sample_pos, sampled, full_pos, window=8):

    reconstructed = np.zeros(len(full_pos))

    Ts = np.mean(np.diff(sample_pos))

    for i, t in enumerate(full_pos):

        x = (t - sample_pos) / Ts

        mask = np.abs(x) <= window

        x_local = x[mask]

        samples_local = sampled[mask]

        weights = np.sinc(x_local)

        hamming = (
            0.54
            + 0.46 * np.cos(np.pi * x_local / window)
        )

        weights *= hamming

        s = np.sum(weights)

        if abs(s) > 1e-12:
            reconstructed[i] = (
                np.sum(samples_local * weights) / s
            )

    return reconstructed

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="DSP Lab", layout="wide")

# =========================
# MATTE HIGH-CONTRAST DESIGN SYSTEM
# =========================
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: #f3f1ea;
    color: #1e1e1e;
    font-family: Georgia, serif;
}

/* ===== GLOBAL TEXT ===== */
h1, h2, h3, p, label {
    text-align: center;
    color: #1e1e1e !important;
}

/* ===== TITLE ===== */
h1 {
    font-size: 40px;
    margin-bottom: 0px;
}

/* ===== EXPLANATION BOX ===== */
.explain {
    background: #ffffff;
    border: 1px solid #c9b48a;
    border-radius: 14px;
    padding: 18px;
    margin: 20px auto;
    width: 85%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    line-height: 1.6;
    color: #1e1e1e;
}

/* ===== CONTROL BOX ===== */
.control-box {
    background: #ffffff;
    border: 1px solid #d6c7a8;
    border-radius: 14px;
    padding: 16px;
    margin: 20px auto;
    width: 85%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #c9b48a;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetricLabel"] {
    color: #2b2b2b !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #082567 !important;
    font-weight: 700;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #d6c7a8;
    border-radius: 12px;
    padding: 10px;
}

/* ===== SLIDER / INPUTS ===== */
[data-testid="stSlider"],
[data-testid="stSelectbox"],
[data-testid="stNumberInput"] {
    background: #ffffff;
    border: 1px solid #d6c7a8;
    border-radius: 10px;
    padding: 6px;
}

/* ===== AUDIO ===== */
audio {
    width: 100%;
    margin-top: 10px;
}

/* ===== PLOTS FIX (IMPORTANT) ===== */
.js-plotly-plot {
    margin: 25px auto;
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

/* Prevent global centering breaking plots */
div[data-testid="stPlotlyChart"] {
    padding: 10px 0px;
}

/* ===== SPACING ===== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1>Digital Signal Sampling & Reconstruction</h1>", unsafe_allow_html=True)


# =========================
# FILE UPLOAD
# =========================
st.markdown("### Upload Audio Signal")
uploaded_file = st.file_uploader(
    "",
    type=["wav"],
    label_visibility="collapsed"
)

def load_audio(file_obj):
    fs, audio = wavfile.read(file_obj)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(float)
    audio = audio / np.max(np.abs(audio))
    return fs, audio

if uploaded_file:
    fs, audio = load_audio(io.BytesIO(uploaded_file.read()))
else:
    fs, audio = load_audio("101_1b1_Pr_sc_Meditron.wav")

# =========================
# AUDIO PLAYBACK
# =========================

st.markdown("### Original Signal")
st.audio((audio * 32767).astype(np.int16), sample_rate=fs)


# =========================
# FFT ANALYSIS
# =========================
fft_vals = np.abs(np.fft.rfft(audio))
fft_freqs = np.fft.rfftfreq(len(audio), 1/fs)

dominant_frequency = float(fft_freqs[np.argmax(fft_vals[1:]) + 1])
threshold = 0.05 * np.max(fft_vals)
max_frequency = float(np.max(fft_freqs[fft_vals > threshold]))
nyquist = max(1, int(2 * max_frequency))

# =========================
# CONTROLS
# =========================
st.markdown("<div class='control-box'>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:

    rate = int(
        st.number_input(
            "Sampling Rate (Hz)",
            min_value=1,
            value=max(1000, nyquist),
            step=100
        )
    )

with col2:
    method = st.selectbox(
        "Reconstruction Method",
        ["Nearest Neighbor", "Linear", "Cubic", "Lanczos", "Sinc"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# SAMPLING
# =========================
step = max(1, int(fs / rate))

sampled = audio[::step]
sample_pos = np.arange(0, len(sampled) * step, step)
full_pos = np.arange(len(audio))

# =========================
# RECONSTRUCTION
# =========================

if method == "Nearest Neighbor":

    reconstructed = interp1d(
        sample_pos,
        sampled,
        kind="nearest",
        fill_value="extrapolate"
    )(full_pos)

elif method == "Linear":

    reconstructed = interp1d(
        sample_pos,
        sampled,
        kind="linear",
        fill_value="extrapolate"
    )(full_pos)

elif method == "Cubic":

    reconstructed = interp1d(
        sample_pos,
        sampled,
        kind="cubic",
        fill_value="extrapolate"
    )(full_pos)

elif method == "Lanczos":

    reconstructed = lanczos_resample(
        sample_pos,
        sampled,
        full_pos
    )

elif method == "Sinc":

    reconstructed = sinc_resample(
        sample_pos,
        sampled,
        full_pos
    )
# =========================
# RECONSTRUCTED PLAYBACK
# =========================

reconstructed_audio = reconstructed.copy()

max_amp = np.max(np.abs(reconstructed_audio))

if max_amp > 0:
    reconstructed_audio = reconstructed_audio / max_amp

st.markdown("### Reconstructed Signal")

st.audio(
    (reconstructed_audio * 32767).astype(np.int16),
    sample_rate=fs
)
# =========================
# METRICS
# =========================
error = audio - reconstructed
accuracy = max(0, min(100, 100 - np.mean(np.abs(error)) * 100))

st.markdown("## Signal Metrics")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Dominant Frequency", f"{dominant_frequency:.2f} Hz")
m2.metric("Nyquist Rate", f"{nyquist} Hz")
m3.metric("Sampling Rate", f"{rate} Hz")
m4.metric("Accuracy", f"{accuracy:.2f}%")

# =========================
# INFO BOX
# =========================
st.markdown(f"""
<div class="explain">

<b>Nyquist Condition:</b> Fs ≥ 2Fmax<br><br>

Sampling Rate: {rate} Hz<br>
Max Frequency: {max_frequency:.2f} Hz<br>
Nyquist Rate: {nyquist} Hz<br>

</div>
""", unsafe_allow_html=True)


# =========================
# CONCLUSION BOX
# =========================

if rate < nyquist:

    status = "ALIASING DETECTED"

    conclusion = """
    Sampling frequency is below the Nyquist rate.
    High-frequency components overlap and cause aliasing,
    resulting in loss of information and distortion.
    """

    colour = "#8B0000"

elif rate < 1.5 * nyquist:

    status = "ACCEPTABLE RECONSTRUCTION"

    conclusion = """
    Sampling frequency satisfies the Nyquist criterion.
    The signal can be reconstructed with reasonable
    accuracy, though small errors may remain.
    """

    colour = "#B8860B"

else:

    status = "HIGH-FIDELITY RECONSTRUCTION"

    conclusion = """
    Sampling frequency is well above the Nyquist rate.
    Aliasing is avoided and the reconstructed signal
    closely matches the original signal.
    """

    colour = "#1E6B3A"

st.markdown(
f"""
<div class="explain">

<h3 style="color:{colour}; text-align:center;">
{status}
</h3>

<p style="text-align:center;">
{conclusion}
</p>

</div>
""",
unsafe_allow_html=True
)
# =========================
# PLOTS (RESTORED + FIXED)
# =========================
display = min(4000, len(audio))
x = np.arange(display)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=audio[:display],
        name="Original",
        line=dict(color="#0B2545", width=2.5)
    )
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=reconstructed[:display],
        name="Reconstructed",
        line=dict(color="#8B5E34", width=2.5)
    )
)
fig.update_layout(
    title="Original vs Reconstructed Signal",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#1e1e1e"),
    legend=dict(
        font=dict(
            color="#1A1A1A",
            size=14
        )
    ),
    height=450,
    margin=dict(l=40, r=40, t=50, b=40)
)

st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=x,
        y=error[:display],
        name="Error",
        line=dict(color="#7A1F1F", width=2.5)
    )
)
fig2.update_layout(
    title="Reconstruction Error",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#1e1e1e"),
    legend=dict(
        font=dict(
            color="#1A1A1A",
            size=14
        )
    ),
    height=400,
    margin=dict(l=40, r=40, t=50, b=40)
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# ADVANCED INFO
# =========================
with st.expander("Advanced Signal Information"):
    st.write({
        "Sample Rate": fs,
        "Duration (s)": round(len(audio)/fs, 2),
        "Total Samples": len(audio),
        "Max Frequency": round(max_frequency, 2),
        "Method": method
    })
