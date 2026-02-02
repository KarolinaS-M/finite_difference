import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Finite Difference Schemes for x'(t)=λx(t)",
    layout="wide"
)

# ======================================================
# Title and problem description
# ======================================================

st.title("Finite Difference Schemes for a Linear ODE")

st.markdown(
    "We compare three classical finite difference discretizations for the linear "
    "ordinary differential equation and illustrate their numerical stability properties."
)

st.latex(r"""
x'(t)=\lambda x(t), \qquad x(0)=x_0
""")

st.info(
    "Backward difference is unconditionally stable for λ < 0, "
    "forward difference may become unstable for coarse grids, "
    "while central difference - although higher order - often produces oscillations."
)

# ======================================================
# Sidebar: parameters
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input("λ (negative for stability test)", value=-1.0, step=0.1, format="%.3f")
    T = st.number_input("Final time T", value=5.0, step=0.5, format="%.2f")
    x0 = st.number_input("Initial value x(0)", value=1.0, step=0.1, format="%.2f")

    st.markdown("---")
    st.header("Grid resolution")

    N = st.slider("Number of grid points N", min_value=5, max_value=200, value=20, step=5)

# ======================================================
# Time grid
# ======================================================

h = T / N
t = np.linspace(0, T, N + 1)

# ======================================================
# Exact solution
# ======================================================

x_exact = x0 * np.exp(lam * t)

# ======================================================
# Backward difference (stable)
# ======================================================

x_backward = np.zeros(N + 1)
x_backward[0] = x0

for i in range(1, N + 1):
    x_backward[i] = x_backward[i - 1] / (1 - h * lam)

# ======================================================
# Forward difference (possibly unstable)
# ======================================================

x_forward = np.zeros(N + 1)
x_forward[0] = x0

for i in range(N):
    x_forward[i + 1] = x_forward[i] * (1 + h * lam)

# ======================================================
# Central difference (oscillatory)
# ======================================================

x_central = np.zeros(N + 1)
x_central[0] = x0
x_central[1] = x_exact[1]

for i in range(1, N):
    x_central[i + 1] = x_central[i - 1] + 2 * h * lam * x_central[i]

# ======================================================
# Numerical summary
# ======================================================

st.subheader("Discretization summary")

st.markdown(
    f"""
**Step size:** $h={h:.4f}$  

• backward difference - stable decay  
• central difference - oscillatory artifacts  
• forward difference - possible divergence
"""
)

# ======================================================
# Plot
# ======================================================

st.subheader("Comparison of finite difference schemes")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

# Backward - GREEN
axes[0].plot(t, x_exact, color="black", linewidth=2, label="Exact")
axes[0].plot(t, x_backward, "o--", color="green", label="Backward FD")
axes[0].set_title("Backward difference")
axes[0].grid(True)
axes[0].legend()

# Central - ORANGE
axes[1].plot(t, x_exact, color="black", linewidth=2, label="Exact")
axes[1].plot(t, x_central, "o--", color="orange", label="Central FD")
axes[1].set_title("Central difference")
axes[1].grid(True)
axes[1].legend()

# Forward - RED
axes[2].plot(t, x_exact, color="black", linewidth=2, label="Exact")
axes[2].plot(t, x_forward, "o--", color="red", label="Forward FD")
axes[2].set_title("Forward difference")
axes[2].grid(True)
axes[2].legend()

for ax in axes:
    ax.set_xlabel("Time t")

axes[0].set_ylabel("x(t)")

plt.tight_layout()
st.pyplot(fig)

# ======================================================
# Interpretation
# ======================================================

st.info(
    "For small step sizes all methods approximate the exact solution well. "
    "As the grid becomes coarser, the forward scheme may diverge, "
    "the central scheme develops spurious oscillations, "
    "while the backward scheme remains stable."
)