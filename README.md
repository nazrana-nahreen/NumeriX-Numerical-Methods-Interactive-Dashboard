# ⚡ NumeriX — Numerical Methods Interactive Dashboard

An interactive dashboard built with **Python + Streamlit** for visualizing and comparing numerical methods. Developed as a project for **CSE-4746: Numerical Methods Lab** at IIUC (7th Semester).

---

## 📸 Modules

| Module | Methods Covered |
|--------|----------------|
| ⚙️ Root Finding | Bisection, Newton-Raphson, False Position, Secant |
| 📐 Interpolation | Newton Forward, Newton Backward, Lagrange, Divided Difference |
| 🔲 Linear Systems | Gaussian Elimination, Gauss-Jordan, Jacobi, Gauss-Seidel |
| ∫ Numerical Integration | Trapezoidal, Simpson's 1/3, Simpson's 3/8 |
| 📈 ODE Solver | Euler's Method, Runge-Kutta 4 (RK4) |
| 〰️ Curve Fitting | Least Square Linear, Parabolic, Exponential |

---

## ✨ Features

- **Type any equation** — uses SymPy to parse and evaluate symbolically
- **Side-by-side method comparison** — see which method converges fastest
- **Convergence graphs** — error vs iteration on log scale
- **Error analysis** — accuracy vs number of subintervals (log-log plots)
- **ODE error vs step size** — see O(h) Euler vs O(h⁴) RK4 difference visually
- **Difference tables** — Newton forward difference table rendered in-app
- **R² comparison** — for curve fitting models
- **Dark themed UI** — clean dashboard aesthetic

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/numerix-dashboard.git
cd numerix-dashboard

# Install dependencies
pip install streamlit matplotlib numpy sympy scipy pandas
```

### Run

```bash
python -m streamlit run numerical_dashboard.py
```

Then open **http://localhost:8501** in your browser.

---

## 🧪 Usage Examples

### Root Finding
Enter any equation like `x**3 - 6*x + 4` with an interval, and all four methods run simultaneously showing convergence speed comparison.

### Interpolation
Enter comma-separated x and y values, specify the point to estimate — all four interpolation methods compute and plot results together.

### Linear Systems
Enter matrix A and vector b — Gaussian Elimination shows step-by-step row operations, while Jacobi and Gauss-Seidel show iteration convergence graphs.

### Numerical Integration
Enter any function (e.g. `log10(x)`), limits, and subinterval count — compares all three rules with error analysis against the true value.

### ODE Solver
Enter `dy/dx = f(x, y)` — Euler and RK4 solutions are plotted together. Provide the exact solution to generate a global error vs step size plot.

### Curve Fitting
Enter data points — linear, parabolic, and exponential models are all fitted with R² values for comparison.

---

## 🧮 Supported Function Syntax

| Math | Python syntax |
|------|--------------|
| x³   | `x**3` |
| sin(x) | `sin(x)` |
| eˣ | `exp(x)` |
| ln(x) | `log(x)` |
| log₁₀(x) | `log(x, 10)` |
| √x | `sqrt(x)` |
| π | `pi` |

---

## 📁 Project Structure

```
numerix-dashboard/
│
├── numerical_dashboard.py   # Main Streamlit app (all modules)
└── README.md
```

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web UI framework
- [NumPy](https://numpy.org/) — numerical computation
- [SymPy](https://www.sympy.org/) — symbolic math / equation parsing
- [SciPy](https://scipy.org/) — reference integration values
- [Matplotlib](https://matplotlib.org/) — plotting and visualization
- [Pandas](https://pandas.pydata.org/) — data display

---

## 📚 Course Info

**Course:** CSE-4746 — Numerical Methods Lab  
**Semester:** 7th Semester, BSc in CSE  
**Institution:** International Islamic University Chittagong (IIUC)  

**CLOs Addressed:**
- CLO1 — Use modern tools to solve numerical computational problems
- CLO2 — Implement the solution of different numerical methods
- CLO3 — Analyze different solution approaches and decide which is appropriate

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
