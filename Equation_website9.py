import streamlit as st
import scipy
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io


st.title("Maxwell-Boltzmann Speed Distribution Calculator")
st.write("Calculate and visualize the RMS, Mean, and Most Probable speeds of different gases at different temperatures using the Maxwell-Boltzmann equation.")
st.latex(r"f(v) = 4\pi \left( \frac{m}{2\pi k_B T} \right)^{3/2} v^2 \exp\left( -\frac{m v^2}{2 k_B T} \right)")
st.header("RMS Speed")
st.write(f"The RMS speed is calculated by multiplying the Maxwell-Boltzmann equation by x squared")


gas_data = {
    "neon": 0.02018, "argon": 0.039948, "krypton": 0.083789,
    "helium": 0.0040026, "xenon": 0.131293, "radon": 0.222,
    "hydrogen": 0.002016, "nitrogen": 0.028014, "oxygen": 0.031998,
    "fluorine": 0.037996, "chlorine": 0.0709, "bromine": 0.159808,
    "iodine": 0.253808, "sulfur": 0.25652
}


selected_gas = st.selectbox("Select a gas:", list(gas_data.keys()), index=6, format_func=lambda x: x.capitalize()) 
temp = st.slider("Temperature (K):", min_value=1, max_value=3000, value=298)
mass = gas_data[selected_gas]


def mp(mass, temp):
    silent_output = io.StringIO()
    with contextlib.redirect_stdout(silent_output):
        mostprob = scipy.optimize.fmin(lambda speed: -4 * np.pi * (((mass/(2 * 8.314 * temp))/ np.pi)**(3/2) ) * speed**2 * np.exp(-(mass/(2 * 8.314 * temp)) * speed**2), 0)[0]
    return mostprob

rms_noint = lambda speed, mass, temp: 4 * np.pi * (((mass/(2 * 8.314 * temp))/ np.pi)**(3/2) ) * speed**2 * np.exp(-(mass/(2 * 8.314 * temp)) * speed**2)* speed**2
mean_noint = lambda speed, mass, temp: 4 * np.pi * (((mass/(2 * 8.314 * temp))/ np.pi)**(3/2) ) * speed**2 * np.exp(-(mass/(2 * 8.314 * temp)) * speed**2)* speed
speed_func = lambda speed, mass, temp: 4 * np.pi * (((mass/(2*8.314*temp))/np.pi)**(3/2)) * speed**2 * np.exp(-(mass/(2*8.314*temp)) * speed**2)

if st.button("Calculate & Plot"):
    
    rms_int = scipy.integrate.quad(rms_noint, 0, np.inf, args=(mass, temp))
    rms = np.sqrt(rms_int[0])
    mean_int = scipy.integrate.quad(mean_noint, 0, np.inf, args=(mass, temp))[0]
    p = mp(mass, temp)

    st.subheader(f"Results for {selected_gas.capitalize()} at {temp} K")
    col1, col2, col3 = st.columns(3)
    col1.metric("Most Probable Speed", f"{p:.2f} m/s")
    col2.metric("Mean Speed", f"{mean_int:.2f} m/s")
    col3.metric("RMS Speed", f"{rms:.2f} m/s")

    x_axis = np.linspace(0, 5000, 3000) 
    
    n = np.array(speed_func(x_axis, mass, temp))
    y_rms = speed_func(rms, mass, temp)
    y_mean = speed_func(mean_int, mass, temp)
    y_p = speed_func(p, mass, temp)

    fig, ax = plt.subplots()
    ax.plot(x_axis, n, color="black")
    
    ax.plot(p, y_p, marker=".", label="Most Probable", color="black", markersize=12, linestyle="None")
    ax.plot(mean_int, y_mean, marker="*", label="Mean", color="black", markersize=10, linestyle="None")
    ax.plot(rms, y_rms, marker="x", label="RMS", color="black", markersize=10, linestyle="None")

    offset = 400  
    ax.annotate(f"{p:.0f}", (p, y_p), xytext=(p + offset, y_p), fontsize=9, color="black",
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color="black"))
    ax.annotate(f"{mean_int:.0f}", (mean_int, y_mean), xytext=(mean_int + offset, y_mean), fontsize=9, color="black",
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color="black"))
    ax.annotate(f"{rms:.0f}", (rms, y_rms), xytext=(rms + offset, y_rms), fontsize=9, color="black",
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color="black"))

    ax.set_ylabel("Probability Density")
    ax.set_xlabel(f"{selected_gas.capitalize()} Particle Speed at {temp} K (m/s)")
    ax.legend()


    st.pyplot(fig)
