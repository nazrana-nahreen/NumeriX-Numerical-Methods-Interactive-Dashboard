import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sympy as sp
from sympy import symbols, sympify, lambdify, diff, latex
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NumeriX — Numerical Methods Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark sidebar */
section[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] * {
    color: #c8ccd8 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem;
    letter-spacing: 0.03em;
}

/* Main background */
.stApp {
    background: #0a0c10 !important;
}

/* Main content color */
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 { color: #e8eaf0 !important; }
p, label, .stMarkdown { color: #9aa0b4 !important; }

/* Glowing header bar */
.hero-bar {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-bar::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(99,102,241,0.12) 0%, transparent 70%);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-sub {
    color: #6b7280 !important;
    font-size: 0.9rem;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Method cards / metric boxes */
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace;
}
.metric-good { color: #34d399 !important; }
.metric-warn { color: #fbbf24 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 10px;
    gap: 0.3rem;
    padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2433 !important;
    color: #a5b4fc !important;
}

/* Input fields */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* Result box */
.result-box {
    background: #0d1117;
    border: 1px solid #374151;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #d1d5db;
    white-space: pre-wrap;
}

/* Section labels */
.section-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.6rem;
}

/* Matplotlib dark style helper */
div[data-testid="stPlotlyChart"], .stpyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
DARK = {
    'bg':     '#0a0c10',
    'panel':  '#111827',
    'border': '#1f2937',
    'text':   '#9aa0b4',
    'grid':   '#1a2035',
    'colors': ['#818cf8','#34d399','#fbbf24','#f87171','#60a5fa','#a78bfa'],
}

def dark_fig(w=10, h=5, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h))
    axes = [ax] if (nrows == 1 and ncols == 1) else ax.flatten() if hasattr(ax,'flatten') else ax
    fig.patch.set_facecolor(DARK['bg'])
    for a in (axes if isinstance(axes, (list, np.ndarray)) else [axes]):
        a.set_facecolor(DARK['panel'])
        a.tick_params(colors=DARK['text'], labelsize=9)
        a.spines[['top','right']].set_visible(False)
        a.spines[['left','bottom']].set_color(DARK['border'])
        a.yaxis.label.set_color(DARK['text'])
        a.xaxis.label.set_color(DARK['text'])
        a.title.set_color('#e8eaf0')
        a.grid(True, color=DARK['grid'], linewidth=0.6, linestyle='--')
    return fig, ax


def parse_func(expr_str):
    x = symbols('x')
    expr = sympify(expr_str, locals={'x': x, 'e': sp.E, 'pi': sp.pi,
                                     'sin': sp.sin, 'cos': sp.cos,
                                     'tan': sp.tan, 'exp': sp.exp,
                                     'log': sp.log, 'sqrt': sp.sqrt,
                                     'abs': sp.Abs})
    f = lambdify(x, expr, modules=['numpy'])
    return expr, f


# ─────────────────────────────────────────────
#  ROOT FINDING
# ─────────────────────────────────────────────
def bisection(f, a, b, tol=1e-6, max_iter=100):
    history = []
    if f(a) * f(b) > 0:
        return None, history, "f(a) and f(b) must have opposite signs"
    for i in range(max_iter):
        c = (a + b) / 2
        history.append({'iter': i+1, 'a': a, 'b': b, 'c': c, 'fc': f(c), 'error': abs(b-a)/2})
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, history, "Converged"
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, history, "Max iterations reached"

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    history = []
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            return x, history, "Derivative near zero"
        x_new = x - fx / dfx
        err = abs(x_new - x)
        history.append({'iter': i+1, 'x': x_new, 'fx': f(x_new), 'error': err})
        x = x_new
        if err < tol:
            return x, history, "Converged"
    return x, history, "Max iterations reached"

def false_position(f, a, b, tol=1e-6, max_iter=100):
    history = []
    if f(a) * f(b) > 0:
        return None, history, "f(a) and f(b) must have opposite signs"
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        err = abs(fc)
        history.append({'iter': i+1, 'c': c, 'fc': fc, 'error': err})
        if err < tol:
            return c, history, "Converged"
        if fa * fc < 0:
            b = c
        else:
            a = c
    return c, history, "Max iterations reached"

def secant(f, x0, x1, tol=1e-6, max_iter=100):
    history = []
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-12:
            return x1, history, "Division near zero"
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        err = abs(x2 - x1)
        history.append({'iter': i+1, 'x': x2, 'fx': f(x2), 'error': err})
        x0, x1 = x1, x2
        if err < tol:
            return x2, history, "Converged"
    return x2, history, "Max iterations reached"


def render_root_finding():
    st.markdown('<div class="section-tag">Module 01</div>', unsafe_allow_html=True)
    st.markdown("## Root Finding Methods")
    st.markdown('<p>Compare Bisection, Newton-Raphson, False Position & Secant side-by-side.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        eq_input = st.text_input("f(x) =", value="x**3 - 6*x + 4",
                                  help="Use Python syntax: x**3, sin(x), exp(x), log(x)")
    with col2:
        tol = st.number_input("Tolerance", value=1e-6, format="%.2e", min_value=1e-12, max_value=1e-1)

    col3, col4, col5, col6 = st.columns(4)
    with col3: a = st.number_input("Interval a", value=-3.0)
    with col4: b = st.number_input("Interval b", value=-1.0)
    with col5: x0_nr = st.number_input("NR Initial x₀", value=-2.0)
    with col6: x1_sec = st.number_input("Secant x₁", value=-1.5)

    if st.button("⚡  Run All Root-Finding Methods", key="run_root"):
        try:
            expr, f = parse_func(eq_input)
            x = symbols('x')
            df_expr = diff(expr, x)
            df = lambdify(x, df_expr, modules=['numpy'])

            r_bis, h_bis, s_bis = bisection(f, a, b, tol)
            r_nr,  h_nr,  s_nr  = newton_raphson(f, df, x0_nr, tol)
            r_fp,  h_fp,  s_fp  = false_position(f, a, b, tol)
            r_sec, h_sec, s_sec = secant(f, x0_nr, x1_sec, tol)

            # ── Results row ──
            cols = st.columns(4)
            method_results = [
                ("Bisection",       r_bis, len(h_bis), s_bis, 0),
                ("Newton-Raphson",  r_nr,  len(h_nr),  s_nr,  1),
                ("False Position",  r_fp,  len(h_fp),  s_fp,  2),
                ("Secant",          r_sec, len(h_sec), s_sec, 3),
            ]
            clr = DARK['colors']
            for col, (name, root, iters, status, idx) in zip(cols, method_results):
                with col:
                    root_str = f"{root:.8f}" if root is not None else "N/A"
                    color_cls = "metric-good" if "Converged" in status else "metric-warn"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value" style="color:{clr[idx]}">{root_str}</div>
                        <div style="font-size:0.75rem;color:#6b7280;font-family:'JetBrains Mono',monospace;">
                            {iters} iterations · {status}
                        </div>
                    </div>""", unsafe_allow_html=True)

            # ── Plots ──
            tab1, tab2 = st.tabs(["📉  Convergence Comparison", "🔵  Function Plot"])

            with tab1:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                fig.patch.set_facecolor(DARK['bg'])
                for ax in axes:
                    ax.set_facecolor(DARK['panel'])
                    ax.tick_params(colors=DARK['text'], labelsize=9)
                    ax.spines[['top','right']].set_visible(False)
                    ax.spines[['left','bottom']].set_color(DARK['border'])
                    ax.yaxis.label.set_color(DARK['text'])
                    ax.xaxis.label.set_color(DARK['text'])
                    ax.title.set_color('#e8eaf0')
                    ax.grid(True, color=DARK['grid'], linewidth=0.6, linestyle='--')

                method_histories = [
                    ("Bisection",      h_bis, 0),
                    ("Newton-Raphson", h_nr,  1),
                    ("False Position", h_fp,  2),
                    ("Secant",         h_sec, 3),
                ]

                # Error vs iteration
                for name, hist, idx in method_histories:
                    if hist:
                        errors = [h['error'] for h in hist]
                        axes[0].semilogy(range(1, len(errors)+1), errors,
                                         color=clr[idx], linewidth=2, marker='o',
                                         markersize=4, label=name)
                axes[0].set_xlabel("Iteration", fontsize=10)
                axes[0].set_ylabel("Error (log scale)", fontsize=10)
                axes[0].set_title("Error Convergence", fontsize=12, fontweight='bold')
                axes[0].legend(framealpha=0.2, labelcolor=DARK['text'],
                                facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)

                # Iteration count bar
                names_list = [m[0] for m in method_results]
                iter_counts = [m[2] for m in method_results]
                bars = axes[1].bar(names_list, iter_counts,
                                   color=[clr[i] for i in range(4)],
                                   edgecolor=DARK['border'], linewidth=0.8,
                                   alpha=0.85, width=0.55)
                for bar, cnt in zip(bars, iter_counts):
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                                 str(cnt), ha='center', va='bottom',
                                 color=DARK['text'], fontsize=10, fontfamily='monospace')
                axes[1].set_ylabel("Iterations to converge", fontsize=10)
                axes[1].set_title("Speed Comparison", fontsize=12, fontweight='bold')
                axes[1].tick_params(axis='x', labelsize=9)

                plt.tight_layout(pad=2)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with tab2:
                x_range = np.linspace(a - 1, b + 1, 400)
                try:
                    y_vals = f(x_range)
                    fig2, ax2 = dark_fig(10, 4.5)
                    ax2.plot(x_range, y_vals, color=clr[4], linewidth=2.5, label=f"f(x) = {eq_input}")
                    ax2.axhline(0, color=DARK['border'], linewidth=1)
                    ax2.axvline(0, color=DARK['border'], linewidth=1)
                    for name, root, iters, status, idx in method_results:
                        if root is not None:
                            ax2.scatter([root], [0], color=clr[idx], s=80, zorder=5,
                                        label=f"{name}: {root:.5f}", edgecolors='white', linewidths=0.8)
                    ax2.set_ylim(np.nanpercentile(y_vals, 2), np.nanpercentile(y_vals, 98))
                    ax2.set_xlabel("x", fontsize=10)
                    ax2.set_ylabel("f(x)", fontsize=10)
                    ax2.set_title(f"f(x) = {eq_input}", fontsize=12, fontweight='bold')
                    ax2.legend(framealpha=0.2, labelcolor=DARK['text'],
                                facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()
                except Exception:
                    st.warning("Could not plot function in given range.")

        except Exception as e:
            st.error(f"Error parsing function: {e}")


# ─────────────────────────────────────────────
#  INTERPOLATION
# ─────────────────────────────────────────────
def newton_forward(x_data, y_data, x_val):
    n = len(x_data)
    h = x_data[1] - x_data[0]
    # Build difference table
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_data
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
    s = (x_val - x_data[0]) / h
    result = diff_table[0][0]
    s_term = 1
    factorial = 1
    for k in range(1, n):
        s_term *= (s - (k-1))
        factorial *= k
        result += (s_term / factorial) * diff_table[0][k]
    return result, diff_table

def newton_backward(x_data, y_data, x_val):
    import math
    n = len(x_data)
    h = x_data[1] - x_data[0]
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_data
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
    s = (x_val - x_data[-1]) / h
    result = diff_table[n-1][0]
    for k in range(1, n):
        s_term_arr = 1
        for m in range(k):
            s_term_arr *= (s + m)
        result += (s_term_arr / math.factorial(k)) * diff_table[n-1-k][k]
    return result

def lagrange(x_data, y_data, x_val):
    n = len(x_data)
    result = 0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if j != i:
                term *= (x_val - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

def newton_divided_diff(x_data, y_data, x_val):
    n = len(x_data)
    coef = np.zeros((n, n))
    coef[:, 0] = y_data
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_data[i+j] - x_data[i])
    result = coef[0][0]
    prod = 1.0
    for k in range(1, n):
        prod *= (x_val - x_data[k-1])
        result += prod * coef[0][k]
    return result, coef

def render_interpolation():
    st.markdown('<div class="section-tag">Module 02</div>', unsafe_allow_html=True)
    st.markdown("## Interpolation Methods")
    st.markdown('<p>Newton Forward/Backward, Lagrange, Divided Difference — compare all four.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_input = st.text_input("x values (comma separated)", "1, 2, 3, 4, 5")
    with col2:
        y_input = st.text_input("y = f(x) values (comma separated)", "1, 8, 27, 64, 125")
    x_find = st.number_input("Estimate f(x) at x =", value=2.5)

    if st.button("⚡  Interpolate", key="run_interp"):
        try:
            x_data = np.array([float(v.strip()) for v in x_input.split(',')])
            y_data = np.array([float(v.strip()) for v in y_input.split(',')])

            r_fwd, diff_table = newton_forward(x_data, y_data, x_find)
            r_bwd = newton_backward(x_data, y_data, x_find)
            r_lag = lagrange(x_data, y_data, x_find)
            r_div, div_table = newton_divided_diff(x_data, y_data, x_find)

            true_val = np.interp(x_find, x_data, y_data) if x_find >= x_data[0] and x_find <= x_data[-1] else None

            cols = st.columns(4)
            results = [
                ("Newton Forward",   r_fwd, 0),
                ("Newton Backward",  r_bwd, 1),
                ("Lagrange",         r_lag, 2),
                ("Divided Diff",     r_div, 3),
            ]
            for col, (name, val, idx) in zip(cols, results):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value" style="color:{DARK['colors'][idx]}">{val:.6f}</div>
                        <div style="font-size:0.75rem;color:#6b7280;font-family:'JetBrains Mono',monospace;">at x = {x_find}</div>
                    </div>""", unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["📈  Interpolation Plot", "🗂️  Difference Table"])

            with tab1:
                x_dense = np.linspace(x_data[0]-0.5, x_data[-1]+0.5, 400)
                y_fwd = [newton_forward(x_data, y_data, xi)[0] for xi in x_dense]
                y_lag = [lagrange(x_data, y_data, xi) for xi in x_dense]
                y_div = [newton_divided_diff(x_data, y_data, xi)[0] for xi in x_dense]

                fig, ax = dark_fig(10, 5)
                ax.scatter(x_data, y_data, color='white', s=70, zorder=6, label='Data points',
                           edgecolors=DARK['colors'][0], linewidths=1.5)
                ax.plot(x_dense, y_fwd, color=DARK['colors'][0], linewidth=2, label='Newton Forward', linestyle='-')
                ax.plot(x_dense, y_lag, color=DARK['colors'][1], linewidth=2, label='Lagrange', linestyle='--')
                ax.plot(x_dense, y_div, color=DARK['colors'][3], linewidth=1.5, label='Divided Diff', linestyle=':')
                ax.axvline(x_find, color='#fbbf24', linewidth=1, linestyle='--', alpha=0.6)
                ax.scatter([x_find]*4, [r_fwd, r_bwd, r_lag, r_div],
                           color=DARK['colors'][:4], s=60, zorder=7, marker='D')
                ax.set_xlabel("x"); ax.set_ylabel("f(x)")
                ax.set_title(f"Interpolated value at x = {x_find}", fontsize=12, fontweight='bold')
                ax.set_ylim(min(y_data)*0.5, max(y_data)*1.2)
                ax.legend(framealpha=0.2, labelcolor=DARK['text'],
                           facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with tab2:
                st.markdown("**Forward Difference Table:**")
                n = len(x_data)
                header = ["x", "y"] + [f"Δ{i+1}y" for i in range(n-1)]
                rows = []
                for i in range(n):
                    row = [f"{x_data[i]:.2f}", f"{diff_table[i][0]:.4f}"]
                    for j in range(1, n - i):
                        row.append(f"{diff_table[i][j]:.4f}")
                    rows.append(row)
                table_md = "| " + " | ".join(header) + " |\n"
                table_md += "| " + " | ".join(["---"]*len(header)) + " |\n"
                for row in rows:
                    padded = row + [""] * (len(header) - len(row))
                    table_md += "| " + " | ".join(padded) + " |\n"
                st.markdown(table_md)

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
#  LINEAR SYSTEMS
# ─────────────────────────────────────────────
def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1,1).astype(float)])
    steps = []
    for i in range(n):
        pivot = Ab[i, i]
        if abs(pivot) < 1e-12:
            return None, steps, "Zero pivot encountered"
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] -= factor * Ab[i]
            steps.append(f"R{j+1} ← R{j+1} - ({factor:.3f}) × R{i+1}")
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    return x, steps, "Solved"

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    history = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        err = np.linalg.norm(x - x_old)
        history.append({'iter': k+1, 'x': x.copy(), 'error': err})
        if err < tol:
            return x, history, "Converged"
    return x, history, "Max iterations"

def jacobi(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    history = []
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i,j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        err = np.linalg.norm(x_new - x)
        history.append({'iter': k+1, 'x': x_new.copy(), 'error': err})
        x = x_new
        if err < tol:
            return x, history, "Converged"
    return x, history, "Max iterations"

def render_linear_systems():
    st.markdown('<div class="section-tag">Module 03</div>', unsafe_allow_html=True)
    st.markdown("## Linear Systems Solver")
    st.markdown('<p>Gaussian Elimination, Gauss-Jordan, Jacobi, Gauss-Seidel with convergence tracking.</p>', unsafe_allow_html=True)

    st.markdown("**Enter coefficient matrix A and RHS vector b:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        A_input = st.text_area("Matrix A (rows separated by `;`, values by `,`)",
                               "10, 1, 1; 2, 10, 1; 2, 2, 10", height=100)
    with col2:
        b_input = st.text_area("Vector b", "12; 13; 14", height=100)

    if st.button("⚡  Solve System", key="run_linsys"):
        try:
            A = np.array([[float(v) for v in row.split(',')]
                           for row in A_input.split(';')])
            b = np.array([float(v.strip()) for v in b_input.split(';')])

            x_gauss, steps_g, status_g = gaussian_elimination(A, b)
            x_np = np.linalg.solve(A, b)
            x_jac, hist_jac, status_jac = jacobi(A, b)
            x_sei, hist_sei, status_sei = gauss_seidel(A, b)

            # Results
            n = len(b)
            var_names = [f"x{i+1}" for i in range(n)]

            st.markdown("### Solution")
            method_cols = st.columns(3)
            method_data = [
                ("Gaussian Elim.", x_gauss, status_g, 0),
                ("Jacobi", x_jac, status_jac, 1),
                ("Gauss-Seidel", x_sei, status_sei, 2),
            ]
            for col, (name, sol, status, idx) in zip(method_cols, method_data):
                with col:
                    vals_str = "<br>".join([f"{var_names[i]} = {sol[i]:.6f}" for i in range(n)]) if sol is not None else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                             color:{DARK['colors'][idx]};line-height:1.8;">{vals_str}</div>
                        <div style="font-size:0.72rem;color:#6b7280;margin-top:0.4rem;">{status}</div>
                    </div>""", unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["📉  Iterative Convergence", "🔢  Elimination Steps"])

            with tab1:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                fig.patch.set_facecolor(DARK['bg'])
                for ax in axes:
                    ax.set_facecolor(DARK['panel'])
                    ax.tick_params(colors=DARK['text'], labelsize=9)
                    ax.spines[['top','right']].set_visible(False)
                    ax.spines[['left','bottom']].set_color(DARK['border'])
                    ax.yaxis.label.set_color(DARK['text'])
                    ax.xaxis.label.set_color(DARK['text'])
                    ax.title.set_color('#e8eaf0')
                    ax.grid(True, color=DARK['grid'], linewidth=0.6, linestyle='--')

                if hist_jac:
                    axes[0].semilogy([h['iter'] for h in hist_jac],
                                     [h['error'] for h in hist_jac],
                                     color=DARK['colors'][1], linewidth=2,
                                     marker='o', markersize=4, label='Jacobi')
                if hist_sei:
                    axes[0].semilogy([h['iter'] for h in hist_sei],
                                     [h['error'] for h in hist_sei],
                                     color=DARK['colors'][2], linewidth=2,
                                     marker='s', markersize=4, label='Gauss-Seidel')
                axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Error (log)")
                axes[0].set_title("Iterative Convergence", fontsize=12, fontweight='bold')
                axes[0].legend(framealpha=0.2, labelcolor=DARK['text'],
                                facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)

                # Variable evolution
                if hist_sei:
                    for i in range(n):
                        vals = [h['x'][i] for h in hist_sei]
                        axes[1].plot(range(1, len(vals)+1), vals,
                                     color=DARK['colors'][i % 6], linewidth=2,
                                     label=f"{var_names[i]} (GS)", marker='.')
                    axes[1].axhline(0, color=DARK['border'], linewidth=0.5)
                    true_lines = [x_np[i] for i in range(n)]
                    for i, tl in enumerate(true_lines):
                        axes[1].axhline(tl, color=DARK['colors'][i%6],
                                        linewidth=0.8, linestyle=':', alpha=0.6)
                axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Variable value")
                axes[1].set_title("Variable Convergence (Gauss-Seidel)", fontsize=12, fontweight='bold')
                axes[1].legend(framealpha=0.2, labelcolor=DARK['text'],
                                facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)

                plt.tight_layout(pad=2)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with tab2:
                if steps_g:
                    st.markdown("**Gaussian Elimination Row Operations:**")
                    steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps_g)])
                    st.markdown(f'<div class="result-box">{steps_text}</div>', unsafe_allow_html=True)
                st.markdown(f"**NumPy reference solution:** {x_np}")

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
#  NUMERICAL INTEGRATION
# ─────────────────────────────────────────────
def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2), x, y

def simpsons_13(f, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
    return result, x, y

def simpsons_38(f, a, b, n):
    if n % 3 != 0: n = n + (3 - n%3)
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = 3*h/8 * (y[0] + 3*np.sum(y[1:-1][0::3]) + 3*np.sum(y[1:-1][1::3]) +
                       2*np.sum(y[1:-1][2::3]) + y[-1])
    return result, x, y

def render_integration():
    st.markdown('<div class="section-tag">Module 04</div>', unsafe_allow_html=True)
    st.markdown("## Numerical Integration")
    st.markdown('<p>Trapezoidal, Simpson\'s 1/3 & 3/8 — accuracy vs. subintervals analysis.</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: func_str = st.text_input("f(x) =", "log10(x)", key="int_func")
    with col2: a_int = st.number_input("Lower limit a", value=1.0, key="int_a")
    with col3: b_int = st.number_input("Upper limit b", value=5.0, key="int_b")
    with col4: n_int = st.number_input("Subintervals n", value=100, min_value=2, max_value=10000, step=10, key="int_n")

    if st.button("⚡  Integrate", key="run_int"):
        try:
            func_str_safe = func_str.replace('log10', 'log(_, 10)').replace('log(x, 10)', 'log(x)/log(10)')
            expr, f = parse_func(func_str)
            # Use scipy for "true" value
            from scipy import integrate as sci_int
            true_val, _ = sci_int.quad(f, a_int, b_int)

            r_trap, x_trap, y_trap = trapezoidal(f, a_int, b_int, n_int)
            r_s13,  x_s13,  y_s13  = simpsons_13(f, a_int, b_int, n_int)
            r_s38,  x_s38,  y_s38  = simpsons_38(f, a_int, b_int, n_int)

            results = [
                ("Trapezoidal",    r_trap, abs(r_trap - true_val), 0),
                ("Simpson's 1/3",  r_s13,  abs(r_s13 - true_val),  1),
                ("Simpson's 3/8",  r_s38,  abs(r_s38 - true_val),  2),
                ("True Value",     true_val, 0.0,                   4),
            ]

            cols = st.columns(4)
            for col, (name, val, err, idx) in zip(cols, results):
                with col:
                    err_str = f"Error: {err:.2e}" if err > 0 else "Reference"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value" style="color:{DARK['colors'][idx]}">{val:.8f}</div>
                        <div style="font-size:0.72rem;color:#6b7280;font-family:'JetBrains Mono',monospace;">{err_str}</div>
                    </div>""", unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["🟣  Area Under Curve", "📊  Error vs n"])

            with tab1:
                x_fine = np.linspace(a_int, b_int, 500)
                y_fine = f(x_fine)
                fig, ax = dark_fig(10, 5)
                ax.plot(x_fine, y_fine, color=DARK['colors'][4], linewidth=2.5, label=f"f(x) = {func_str}")
                ax.fill_between(x_fine, 0, y_fine, alpha=0.15, color=DARK['colors'][0])
                # Trapezoidal bars (coarser)
                n_show = min(n_int, 20)
                x_show = np.linspace(a_int, b_int, n_show+1)
                y_show = f(x_show)
                ax.bar(x_show[:-1], y_show[:-1], width=(b_int-a_int)/n_show,
                        align='edge', alpha=0.2, color=DARK['colors'][0],
                        edgecolor=DARK['colors'][0], linewidth=0.5, label='Trapezoids (visual)')
                ax.set_xlabel("x"); ax.set_ylabel("f(x)")
                ax.set_title(f"∫ {func_str} dx from {a_int} to {b_int}", fontsize=12, fontweight='bold')
                ax.legend(framealpha=0.2, labelcolor=DARK['text'],
                           facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with tab2:
                n_vals = [4, 8, 16, 32, 64, 128, 256, 512, 1000]
                e_trap, e_s13, e_s38 = [], [], []
                for nv in n_vals:
                    e_trap.append(abs(trapezoidal(f, a_int, b_int, nv)[0] - true_val))
                    e_s13.append(abs(simpsons_13(f, a_int, b_int, nv)[0] - true_val))
                    e_s38.append(abs(simpsons_38(f, a_int, b_int, nv)[0] - true_val))

                fig2, ax2 = dark_fig(10, 5)
                ax2.loglog(n_vals, e_trap, color=DARK['colors'][0], linewidth=2,
                            marker='o', markersize=5, label='Trapezoidal')
                ax2.loglog(n_vals, e_s13,  color=DARK['colors'][1], linewidth=2,
                            marker='s', markersize=5, label="Simpson's 1/3")
                ax2.loglog(n_vals, e_s38,  color=DARK['colors'][2], linewidth=2,
                            marker='^', markersize=5, label="Simpson's 3/8")
                ax2.set_xlabel("Number of subintervals n (log scale)")
                ax2.set_ylabel("Absolute Error (log scale)")
                ax2.set_title("Error Analysis: All Methods", fontsize=12, fontweight='bold')
                ax2.legend(framealpha=0.2, labelcolor=DARK['text'],
                            facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
#  ODE SOLVER
# ─────────────────────────────────────────────
def euler_method(f, x0, y0, h, x_end):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    while x < x_end - 1e-10:
        y = y + h * f(x, y)
        x = round(x + h, 10)
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

def rk4_method(f, x0, y0, h, x_end):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    while x < x_end - 1e-10:
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x = round(x + h, 10)
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

def render_ode():
    st.markdown('<div class="section-tag">Module 05</div>', unsafe_allow_html=True)
    st.markdown("## ODE Solver: Euler vs Runge-Kutta 4")
    st.markdown('<p>Solve dy/dx = f(x, y) and compare accuracy visually.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        ode_str = st.text_input("dy/dx = f(x, y) =", "x + y", key="ode_func",
                                 help="Use x and y as variables. Example: x**3 + y, sin(x) - y")
    with col2:
        exact_str = st.text_input("Exact solution y(x) = (optional, for error plot)",
                                   "2*exp(x) - x - 1", key="ode_exact",
                                   help="Leave blank if unknown. Example for dy/dx=x+y, y(0)=1: 2*exp(x)-x-1")

    col3, col4, col5, col6 = st.columns(4)
    with col3: x0_ode = st.number_input("x₀", value=0.0, key="ode_x0")
    with col4: y0_ode = st.number_input("y₀", value=1.0, key="ode_y0")
    with col5: h_ode  = st.number_input("Step size h", value=0.1, min_value=0.001, max_value=1.0, step=0.01, key="ode_h")
    with col6: x_end  = st.number_input("Solve until x =", value=2.0, key="ode_xend")

    if st.button("⚡  Solve ODE", key="run_ode"):
        try:
            x_sym, y_sym = symbols('x y')
            ode_expr = sympify(ode_str, locals={'x': x_sym, 'y': y_sym,
                                                 'sin': sp.sin, 'cos': sp.cos,
                                                 'exp': sp.exp, 'log': sp.log})
            ode_f = lambdify([x_sym, y_sym], ode_expr, modules=['numpy'])

            xs_eu, ys_eu = euler_method(ode_f, x0_ode, y0_ode, h_ode, x_end)
            xs_rk, ys_rk = rk4_method(ode_f, x0_ode, y0_ode, h_ode, x_end)

            # Exact if provided
            has_exact = False
            if exact_str.strip():
                try:
                    exact_expr = sympify(exact_str, locals={'x': x_sym, 'exp': sp.exp,
                                                             'sin': sp.sin, 'cos': sp.cos})
                    exact_f = lambdify(x_sym, exact_expr, modules=['numpy'])
                    x_exact = np.linspace(x0_ode, x_end, 300)
                    y_exact = exact_f(x_exact)
                    has_exact = True
                except: pass

            # Show final values
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Euler — y({x_end:.2f})</div>
                    <div class="metric-value" style="color:{DARK['colors'][0]}">{ys_eu[-1]:.8f}</div>
                    <div style="font-size:0.72rem;color:#6b7280;font-family:'JetBrains Mono',monospace;">h = {h_ode} · {len(xs_eu)-1} steps</div>
                </div>""", unsafe_allow_html=True)
            with col_r2:
                rk_final = ys_rk[-1]
                exact_final = exact_f(x_end) if has_exact else None
                err_str = f"Error vs exact: {abs(rk_final - exact_final):.2e}" if has_exact else "No exact provided"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">RK4 — y({x_end:.2f})</div>
                    <div class="metric-value" style="color:{DARK['colors'][1]}">{rk_final:.8f}</div>
                    <div style="font-size:0.72rem;color:#6b7280;font-family:'JetBrains Mono',monospace;">{err_str}</div>
                </div>""", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["📈  Solution Curves", "📉  Error vs Step Size", "🗂️  Step Table"])

            with tab1:
                fig, ax = dark_fig(10, 5)
                if has_exact:
                    ax.plot(x_exact, y_exact, color='white', linewidth=2, linestyle='-',
                             label='Exact solution', zorder=5)
                ax.plot(xs_eu, ys_eu, color=DARK['colors'][0], linewidth=2,
                         marker='o', markersize=3, label='Euler', linestyle='--')
                ax.plot(xs_rk, ys_rk, color=DARK['colors'][1], linewidth=2,
                         marker='s', markersize=3, label='RK4', linestyle='-.')
                ax.set_xlabel("x"); ax.set_ylabel("y(x)")
                ax.set_title(f"ODE: dy/dx = {ode_str}", fontsize=12, fontweight='bold')
                ax.legend(framealpha=0.2, labelcolor=DARK['text'],
                           facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with tab2:
                if has_exact:
                    h_vals = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
                    e_eu_list, e_rk_list = [], []
                    for hv in h_vals:
                        _, ys_e = euler_method(ode_f, x0_ode, y0_ode, hv, x_end)
                        _, ys_r = rk4_method(ode_f, x0_ode, y0_ode, hv, x_end)
                        e_eu_list.append(abs(ys_e[-1] - exact_f(x_end)))
                        e_rk_list.append(abs(ys_r[-1] - exact_f(x_end)))

                    fig2, ax2 = dark_fig(10, 5)
                    ax2.loglog(h_vals, e_eu_list, color=DARK['colors'][0], linewidth=2,
                                marker='o', markersize=6, label='Euler (O(h))')
                    ax2.loglog(h_vals, e_rk_list, color=DARK['colors'][1], linewidth=2,
                                marker='s', markersize=6, label='RK4 (O(h⁴))')
                    ax2.set_xlabel("Step size h (log scale)")
                    ax2.set_ylabel("Global error at x_end (log scale)")
                    ax2.set_title("Global Error vs Step Size", fontsize=12, fontweight='bold')
                    ax2.legend(framealpha=0.2, labelcolor=DARK['text'],
                                facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()
                else:
                    st.info("Provide an exact solution above to generate error analysis.")

            with tab3:
                n_show = min(len(xs_rk), 20)
                idxs = np.linspace(0, len(xs_rk)-1, n_show, dtype=int)
                rows_data = []
                for i in idxs:
                    row = {"x": f"{xs_rk[i]:.4f}",
                            "Euler y": f"{ys_eu[i]:.6f}" if i < len(ys_eu) else "—",
                            "RK4 y": f"{ys_rk[i]:.6f}"}
                    if has_exact:
                        yt = float(exact_f(xs_rk[i]))
                        row["Exact"] = f"{yt:.6f}"
                        row["RK4 Error"] = f"{abs(ys_rk[i]-yt):.2e}"
                    rows_data.append(row)
                import pandas as pd
                df = pd.DataFrame(rows_data)
                st.dataframe(df, use_container_width=True,
                             column_config={c: st.column_config.TextColumn(c) for c in df.columns})

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
#  CURVE FITTING
# ─────────────────────────────────────────────
def render_curve_fitting():
    st.markdown('<div class="section-tag">Module 06</div>', unsafe_allow_html=True)
    st.markdown("## Curve Fitting & Least Squares")
    st.markdown('<p>Fit linear, parabolic, and exponential models — compare R² values.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x_cf = st.text_input("x values", "-2, -1, 0, 1, 2", key="cf_x")
    with col2:
        y_cf = st.text_input("y values", "1, 2, 3, 4, 5", key="cf_y")

    if st.button("⚡  Fit Curves", key="run_cf"):
        try:
            x_data = np.array([float(v.strip()) for v in x_cf.split(',')])
            y_data = np.array([float(v.strip()) for v in y_cf.split(',')])

            # Linear
            coeffs1 = np.polyfit(x_data, y_data, 1)
            y_lin = np.polyval(coeffs1, x_data)
            ss_res = np.sum((y_data - y_lin)**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2_lin = 1 - ss_res/ss_tot if ss_tot != 0 else 1.0

            # Parabolic
            coeffs2 = np.polyfit(x_data, y_data, 2)
            y_par = np.polyval(coeffs2, x_data)
            ss_res2 = np.sum((y_data - y_par)**2)
            r2_par = 1 - ss_res2/ss_tot if ss_tot != 0 else 1.0

            # Exponential (y = a*e^(bx))
            has_exp = all(y > 0 for y in y_data)
            if has_exp:
                log_y = np.log(y_data)
                coeffs_exp = np.polyfit(x_data, log_y, 1)
                b_exp = coeffs_exp[0]
                a_exp = np.exp(coeffs_exp[1])
                y_exp = a_exp * np.exp(b_exp * x_data)
                ss_exp = np.sum((y_data - y_exp)**2)
                r2_exp = 1 - ss_exp/ss_tot if ss_tot != 0 else 1.0

            # Metric cards
            cols = st.columns(3 if has_exp else 2)
            fit_results = [
                ("Linear y = a + bx",
                 f"a={coeffs1[1]:.4f}, b={coeffs1[0]:.4f}", r2_lin, 0),
                ("Parabola y = a + bx + cx²",
                 f"a={coeffs2[2]:.4f}, b={coeffs2[1]:.4f}, c={coeffs2[0]:.4f}", r2_par, 1),
            ]
            if has_exp:
                fit_results.append(
                    ("Exponential y = ae^(bx)",
                     f"a={a_exp:.4f}, b={b_exp:.4f}", r2_exp, 2))

            for col, (name, params, r2, idx) in zip(cols, fit_results):
                with col:
                    r2_color = DARK['colors'][1] if r2 > 0.95 else DARK['colors'][3]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                             color:{DARK['colors'][idx]};line-height:1.6;">{params}</div>
                        <div style="font-size:1rem;font-weight:600;font-family:'JetBrains Mono',monospace;
                             color:{r2_color};margin-top:0.3rem;">R² = {r2:.6f}</div>
                    </div>""", unsafe_allow_html=True)

            # Plot
            x_dense = np.linspace(x_data[0]-0.5, x_data[-1]+0.5, 300)
            y_lin_d = np.polyval(coeffs1, x_dense)
            y_par_d = np.polyval(coeffs2, x_dense)

            fig, ax = dark_fig(10, 5)
            ax.scatter(x_data, y_data, color='white', s=80, zorder=6,
                        edgecolors=DARK['colors'][4], linewidths=1.5, label='Data')
            ax.plot(x_dense, y_lin_d, color=DARK['colors'][0], linewidth=2,
                     label=f"Linear (R²={r2_lin:.4f})")
            ax.plot(x_dense, y_par_d, color=DARK['colors'][1], linewidth=2,
                     linestyle='--', label=f"Parabola (R²={r2_par:.4f})")
            if has_exp:
                y_exp_d = a_exp * np.exp(b_exp * x_dense)
                ax.plot(x_dense, y_exp_d, color=DARK['colors'][2], linewidth=2,
                         linestyle=':', label=f"Exponential (R²={r2_exp:.4f})")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title("Curve Fitting Comparison", fontsize=12, fontweight='bold')
            ax.legend(framealpha=0.2, labelcolor=DARK['text'],
                       facecolor=DARK['panel'], edgecolor=DARK['border'], fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0.5rem 1rem;">
        <div style="font-size:1.3rem;font-weight:700;
             background:linear-gradient(90deg,#818cf8,#60a5fa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             letter-spacing:-0.02em;">NumeriX</div>
        <div style="font-size:0.7rem;color:#4b5563;font-family:'JetBrains Mono',monospace;
             margin-top:0.2rem;">CSE-4746 · Numerical Methods Lab</div>
    </div>
    <hr style="border-color:#1f2937;margin:0 0 1rem;">
    """, unsafe_allow_html=True)

    module = st.radio(
        "Select Module",
        options=[
            "🏠  Overview",
            "⚙️  Root Finding",
            "📐  Interpolation",
            "🔲  Linear Systems",
            "∫   Numerical Integration",
            "📈  ODE Solver",
            "〰️  Curve Fitting",
        ],
        index=0
    )

    st.markdown("""
    <hr style="border-color:#1f2937;margin:1rem 0;">
    <div style="font-size:0.7rem;color:#374151;font-family:'JetBrains Mono',monospace;
         padding:0 0.5rem;line-height:1.8;">
        IIUC · 7th Semester<br>
        CSE-4746 Project<br>
        All CLOs covered
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  OVERVIEW PAGE
# ─────────────────────────────────────────────
def render_overview():
    st.markdown("""
    <div class="hero-bar">
        <p class="hero-title">NumeriX Dashboard</p>
        <p class="hero-sub">// numerical methods · interactive · CSE-4746</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    overview_cards = [
        ("5 Modules", "Root finding, interpolation, linear systems, integration, ODE", DARK['colors'][0]),
        ("4 Root Methods", "Bisection · Newton-Raphson · False Position · Secant", DARK['colors'][1]),
        ("Full CLO Coverage", "CLO1 · CLO2 · CLO3 — all outcomes demonstrated", DARK['colors'][2]),
    ]
    for col, (title, desc, color) in zip([col1, col2, col3], overview_cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid {color};">
                <div class="metric-value" style="color:{color};font-size:1.1rem;">{title}</div>
                <div style="font-size:0.8rem;color:#6b7280;margin-top:0.4rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### What's inside")

    modules_info = [
        ("⚙️", "Root Finding", "Compare Bisection, Newton-Raphson, False Position & Secant on any equation. Live convergence graphs and iteration count comparison."),
        ("📐", "Interpolation", "Newton Forward/Backward, Lagrange & Divided Difference. Tabulated difference tables + visual curve overlay."),
        ("🔲", "Linear Systems", "Gaussian Elimination step-by-step, plus Jacobi & Gauss-Seidel convergence tracking with variable evolution plots."),
        ("∫",  "Numerical Integration", "Trapezoidal, Simpson's 1/3 & 3/8. Error vs subintervals log-log analysis — see why Simpson beats Trapezoidal."),
        ("📈", "ODE Solver", "Euler vs RK4 on any dy/dx. If you supply the exact solution, it plots global error vs step size — showing O(h) vs O(h⁴)."),
        ("〰️", "Curve Fitting", "Least-square linear, parabolic & exponential fits with R² comparison. See which model best fits your data."),
    ]

    for icon, title, desc in modules_info:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;
             padding:1rem 1.3rem;margin-bottom:0.6rem;display:flex;gap:1rem;align-items:flex-start;">
            <div style="font-size:1.4rem;line-height:1;">{icon}</div>
            <div>
                <div style="font-weight:600;color:#e8eaf0;font-size:0.9rem;">{title}</div>
                <div style="font-size:0.8rem;color:#6b7280;margin-top:0.2rem;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);
         border-radius:10px;padding:1rem 1.3rem;margin-top:1rem;">
        <div style="color:#a5b4fc;font-size:0.85rem;font-weight:600;">How to use</div>
        <div style="color:#6b7280;font-size:0.8rem;margin-top:0.3rem;">
        Pick a module from the sidebar → enter your equation or data → click the ⚡ Run button.
        All computations run from scratch with your inputs. Compare results across methods instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROUTING
# ─────────────────────────────────────────────
if "Overview" in module:
    render_overview()
elif "Root Finding" in module:
    render_root_finding()
elif "Interpolation" in module:
    render_interpolation()
elif "Linear Systems" in module:
    render_linear_systems()
elif "Integration" in module:
    render_integration()
elif "ODE" in module:
    render_ode()
elif "Curve Fitting" in module:
    render_curve_fitting()
