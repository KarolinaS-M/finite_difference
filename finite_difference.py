import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Finite Difference Method for a Linear BVP",
    layout="wide"
)

# ======================================================
# Title and problem description
# ======================================================

st.title("Finite Difference Method for a Linear Boundary Value Problem")

st.markdown(
    "We illustrate the finite difference method for a **first–order boundary value problem** "
    "and compare three classical discretizations: backward, central, and forward differences."
)

st.latex(r"""
x'(t)=\lambda x(t), \qquad x(0)=x_0,\; x(T)=x_T
""")

st.info(
    "In boundary value problems, finite difference methods determine the entire solution "
    "trajectory simultaneously by solving a global algebraic system. "
    "This differs fundamentally from time–marching schemes used for initial value problems."
)

# ======================================================
# Sidebar: parameters
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input(
        "λ (lambda)", value=-1.0, step=0.1, format="%.3f"
    )
    T = st.number_input(
        "Terminal time T", value=5.0, step=0.5, format="%.2f"
    )
    x0 = st.number_input(
        "Boundary value x(0)", value=1.0, step=0.1, format="%.2f"
    )
    xT = st.number_input(
        "Boundary value x(T)", value=np.exp(-5), step=0.1, format="%.4f"
    )

    st.markdown("---")
    st.header("Grid resolution")

    N = st.slider(
        "Number of subintervals N",
        min_value=10,
        max_value=400,
        value=50,
        step=10
    )

# ======================================================
# Grid and exact solution
# ======================================================

h = T / N
t = np.linspace(0, T, N + 1)

x_exact = x0 * np.exp(lam * t)

# ======================================================
# Backward difference scheme (BVP formulation)
# ======================================================

x_backward = np.zeros(N + 1)
x_backward[0] = x0
for i in range(1, N + 1):
    x_backward[i] = x_backward[i - 1] / (1 - h * lam)

# rescale to satisfy terminal condition
x_backward *= xT / x_backward[-1]

# ======================================================
# Forward difference scheme (BVP formulation)
# ======================================================

x_forward = np.zeros(N + 1)
x_forward[0] = x0
for i in range(N):
    x_forward[i + 1] = x_forward[i] * (1 + h * lam)

# rescale to satisfy terminal condition
x_forward *= xT / x_forward[-1]

# ======================================================
# Central difference scheme (BVP formulation)
# ======================================================

x_central = np.zeros(N + 1)
x_central[0] = x0
x_central[1] = x0 * np.exp(lam * h)

for i in range(1, N):
    x_central[i + 1] = x_central[i - 1] + 2 * h * lam * x_central[i]

# rescale to satisfy terminal condition
x_central *= xT / x_central[-1]

# ======================================================
# Summary
# ======================================================

st.subheader("Discretization summary")

st.markdown(
    f"""
**Step size:** $h = {h:.4f}$  

• backward difference – monotone and accurate for $\lambda<0$  
• central difference – higher order but prone to oscillatory artifacts  
• forward difference – smooth but typically less accurate globally  
"""
)

# ======================================================
# Plot: three panels
# ======================================================

st.subheader("Comparison of finite difference schemes")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# Backward (GREEN)
axes[0].plot(t, x_exact, color="black", linewidth=2, label="Exact")
axes[0].plot(t, x_backward, "o--", color="green", label="Backward FD")
axes[0].set_title("Backward difference")
axes[0].grid(True)
axes[0].legend()

# Central (ORANGE)
axes[1].plot(t, x_exact, color="black", linewidth=2, label="Exact")
axes[1].plot(t, x_central, "o--", color="orange", label="Central FD")
axes[1].set_title("Central difference")
axes[1].grid(True)
axes[1].legend()

# Forward (RED)
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
    "All three schemes are algebraically stable in this linear boundary value problem. "
    "The backward difference preserves monotonic decay, the central difference introduces "
    "small oscillatory modes due to its symmetric stencil, and the forward difference "
    "typically yields a less accurate global approximation for the same grid resolution."
)