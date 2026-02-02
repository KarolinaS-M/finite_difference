import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Finite Difference Methods for a Linear Boundary Value Problem",
    layout="wide"
)

# ======================================================
# Title and description
# ======================================================

st.title("Finite Difference Schemes for a Linear Boundary Value Problem")

st.markdown(
    "This application compares three classical finite difference discretizations "
    "for the linear boundary value problem"
)

st.latex(r"""
x'(t)=\lambda x(t), \qquad x(0)=x_0,\quad x(T)=x_T
""")

st.markdown(
    "Although all three schemes approximate the same differential equation, "
    "their qualitative behavior differs markedly depending on how the derivative "
    "is discretized."
)

# ======================================================
# Sidebar: parameters
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input(
        r"Growth/decay rate $\lambda$",
        value=-1.0,
        step=0.1,
        format="%.3f"
    )

    T = st.number_input(
        "Terminal time T",
        value=5.0,
        step=0.5,
        format="%.2f"
    )

    x0 = st.number_input(
        "Initial value x(0)",
        value=1.0,
        step=0.1,
        format="%.2f"
    )

    st.markdown("---")
    st.header("Grid resolution")

    N = st.slider(
        "Number of subintervals N",
        min_value=5,
        max_value=200,
        value=20,
        step=5
    )

# ======================================================
# Grid and step size
# ======================================================

h = T / N
t = np.linspace(0, T, N + 1)

# ======================================================
# Exact solution
# ======================================================

x_exact = x0 * np.exp(lam * t)

# ======================================================
# Backward difference scheme
# ======================================================

x_backward = np.zeros(N + 1)
x_backward[0] = x0

for i in range(1, N + 1):
    x_backward[i] = x_backward[i - 1] / (1 - h * lam)

# ======================================================
# Forward difference scheme
# ======================================================

x_forward = np.zeros(N + 1)
x_forward[0] = x0

for i in range(N):
    x_forward[i + 1] = x_forward[i] * (1 + h * lam)

# ======================================================
# Central difference scheme
# ======================================================

x_central = np.zeros(N + 1)
x_central[0] = x0
x_central[1] = x_exact[1]  # starting value from exact solution

for i in range(1, N):
    x_central[i + 1] = x_central[i - 1] + 2 * h * lam * x_central[i]

# ======================================================
# Stability diagnostics (BONUS)
# ======================================================

st.subheader("Discretization diagnostics")

st.markdown(
    rf"""
**Step size:** $h = {h:.4f}$  

• backward difference – stable decay for $\lambda<0$  
• central difference – oscillatory artifacts (alternating mode)  
• forward difference – conditionally stable
"""
)

forward_stable = abs(1 + h * lam) < 1

if forward_stable:
    st.success(
        r"Forward difference stability condition holds: "
        r"$|1+h\lambda|<1$."
    )
else:
    st.warning(
        r"Forward difference stability condition fails: "
        r"$|1+h\lambda|\ge 1$. "
        r"Oscillations or divergence may occur."
    )

# ======================================================
# Plot
# ======================================================

st.subheader("Comparison of finite difference schemes")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# Backward – GREEN
axes[0].plot(t, x_exact, color="black", linewidth=2, label="Exact solution")
axes[0].plot(t, x_backward, "o--", color="green", label="Backward FD")
axes[0].set_title("Backward difference")
axes[0].grid(True)
axes[0].legend()

# Central – ORANGE
axes[1].plot(t, x_exact, color="black", linewidth=2, label="Exact solution")
axes[1].plot(t, x_central, "o--", color="orange", label="Central FD")
axes[1].set_title("Central difference")
axes[1].grid(True)
axes[1].legend()

# Forward – RED
axes[2].plot(t, x_exact, color="black", linewidth=2, label="Exact solution")
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
    "All three schemes solve the same algebraic boundary value problem, "
    "but their qualitative behavior differs. The backward scheme preserves "
    "the monotone decay of the continuous solution. The central scheme introduces "
    "small oscillations because a symmetric stencil is applied to a first–order "
    "equation. The forward scheme is conditionally stable: for sufficiently fine "
    "grids it performs well, but coarse discretizations may lead to oscillatory "
    "or unstable behavior."
)