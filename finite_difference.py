import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Finite Difference Schemes: Linear BVP Illustration",
    layout="wide"
)

# ======================================================
# Title and problem description
# ======================================================

st.title("Finite Difference Schemes: Linear Illustration for a Boundary Value Problem")

st.markdown(
    "We compare three finite difference discretizations for the linear ODE "
    "and highlight their qualitative behavior on a coarse grid."
)

st.latex(r"""
x'(t)=\lambda x(t), \qquad t\in[0,T]
""")

st.markdown(
    "For reference, the exact solution satisfying the initial value $x(0)=x_0$ is "
    "$x(t)=x_0 e^{\lambda t}$. In this illustration we focus on how the **discretization stencil** "
    "affects the numerical trajectory, especially when the grid is coarse."
)

# ======================================================
# Sidebar: parameters
# ======================================================

with st.sidebar:
    st.header("Parameters")

    lam = st.number_input("λ", value=-1.0, step=0.1, format="%.4f")
    T = st.number_input("Terminal time T", value=5.0, step=0.5, format="%.4f")
    x0 = st.number_input("Initial value x(0)=x₀", value=1.0, step=0.1, format="%.4f")

    st.markdown("---")
    st.header("Grid")

    N = st.slider(
        "Number of subintervals N (coarse grids reveal artifacts)",
        min_value=5,
        max_value=200,
        value=5,
        step=1
    )

# ======================================================
# Time grid and exact solution
# ======================================================

h = T / N
t = np.linspace(0, T, N + 1)

x_exact = x0 * np.exp(lam * t)

# ======================================================
# Finite difference "schemes" (same as your good .ipynb)
# ======================================================

# Backward difference (implicit recurrence for IVP form)
x_backward = np.zeros(N + 1)
x_backward[0] = x0
for i in range(1, N + 1):
    denom = (1 - h * lam)
    # Guard against division by zero (rare, but possible if user sets lam = 1/h)
    if abs(denom) < 1e-14:
        x_backward[i] = np.nan
    else:
        x_backward[i] = x_backward[i - 1] / denom

# Forward difference (explicit recurrence for IVP form)
x_forward = np.zeros(N + 1)
x_forward[0] = x0
for i in range(N):
    x_forward[i + 1] = x_forward[i] * (1 + h * lam)

# Central difference (two-step recursion; seeded for illustration)
x_central = np.zeros(N + 1)
x_central[0] = x0

# Seed x_1 with the exact solution (purely to start the two-step recursion)
if N >= 1:
    x_central[1] = x_exact[1]

for i in range(1, N):
    x_central[i + 1] = x_central[i - 1] + 2 * h * lam * x_central[i]

# ======================================================
# Console-style summary for the reader
# ======================================================

st.subheader("Finite difference method summary")

st.markdown(
    f"""
- **Equation:** $x'(t)=\\lambda x(t)$  
- **Parameters:** $\\lambda={lam:.4f}$, $T={T:.4f}$, $x_0={x0:.4f}$  
- **Grid:** $N={N}$, step size $h=T/N={h:.4f}$  

**Notes (what you should expect to see):**
- **Backward difference:** stable decay for $\\lambda<0$ and often good qualitative behavior on coarse grids.
- **Central difference:** may show **alternating (oscillatory) artifacts** because the symmetric stencil admits spurious alternating modes in a first-order setting.
- **Forward difference:** **stable in this linear example**, but on coarse grids it can be visibly less accurate than backward.
"""
)

# ======================================================
# Plot (three panels)
# ======================================================

st.subheader("Comparison of schemes (three panels)")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

# Backward - green
axes[0].plot(t, x_exact, color="black", linewidth=2, label="Exact")
axes[0].plot(t, x_backward, "o--", color="green", label="Backward FD")
axes[0].set_title("Backward difference")
axes[0].grid(True)
axes[0].legend()

# Central - orange
axes[1].plot(t, x_exact, color="black", linewidth=2, label="Exact")
axes[1].plot(t, x_central, "o--", color="orange", label="Central FD")
axes[1].set_title("Central difference")
axes[1].grid(True)
axes[1].legend()

# Forward - red
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
# Interpretation box (short + consistent with what is visible)
# ======================================================

st.info(
    "Try a coarse grid (e.g., N=5–15) to see how the stencil matters: "
    "the central difference may exhibit alternating artifacts, while backward and forward "
    "remain monotone for λ<0. As N increases, all three curves approach the exact solution."
)