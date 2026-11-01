# Fractional Burgers toolkit

A basic solver for the 1D Burgers equation with fractional orders in space and time (FDEs), and the integer-order PDE. It covers the **integer-order** case: $\alpha=1,\ \beta=2$ is the classical viscous Burgers equation $u_t+c\,u\,u_x=\nu u_{xx}$, and with $c=0$ the heat equation $u_t=\nu u_{xx}$.

$$\partial_t^\alpha u + c\,u\,\partial_x u = -\nu\,(-\Delta)^{\beta/2}u,\qquad x\in[0,L)\ \text{periodic},\quad 0<\alpha\le1,\ 0<\beta\le2$$

| type | $\alpha$ | $\beta$ | `--type` |
|---|---|---|---|
| classical (integer orders) | $=1$ | $=2$ | `classical` |
| space-fractional | $=1$ | $<2$ | `space` |
| time-fractional | $<1$ | $=2$ | `time` |
| space-time fractional | $<1$ | $<2$ | `space-time` |

## 1. Generate data

```bash
python main.py --type space --n-samples 250                  # default orders of the type
python main.py --type classical                              # integer orders: classical viscous Burgers (alpha=1, beta=2)
python main.py --alpha 1 --beta 2 --c 0                      # integer orders, linear: heat equation u_t = nu u_xx
python main.py --alpha 1 --beta 1.2 --nu 0.2                 # any (alpha, beta); low beta needs a larger nu to stay smooth
python main.py --alpha 0.6 --beta 1.5 --check                # only prints the smallest stable nt (nothing generated)
```

| argument | default | meaning |
|---|---|---|
| `--equation` | `burgers` | which PDE (see section 4) |
| `--type`, `--alpha`, `--beta` | `space` | orders (`--alpha/--beta` override the type) |
| `--nu`, `--c` | 0.05, 1 | viscosity; nonlinearity (`c=0`: linear problem) |
| `--n-samples`, `--n-test` | 250, 50 | trajectories; the last `n-test` are the test set |
| `--N`, `--N-out` | 256, 128 | solver grid, saved grid (solve fine, save coarse) |
| `--L`, `--T` | 2π, 1 | domain length, final time |
| `--nt` | 1600 | solver time steps, **fixed** (see below) |
| `--frames` | 21 | saved frames per trajectory, including $t=0$; `nt` must be a multiple of `frames-1` |
| `--p`, `--seed` | 2, 0 | initial-condition spectrum slope, seed |
| `--out` | `data/` | output folder |

`nt` is not chosen automatically so that all files are comparable. If a run blows up, the script stops: raise `--nt` (small $\alpha$ or $\beta$, or a large `N`, need more steps; `--check` finds the smallest stable one among 800 to 6400). A finite run is not necessarily accurate, the scheme is first order in time.

Output, one file per setting (the name contains the parameters):

```
data/<classical|space|time|space-time>/
    burgers_a<alpha>_b<beta>_nu<nu>.npz     u [n_samples, frames, N_out], t, x and the settings
    simulations/<name>.png                  image of the first sample
    equations/<name>.txt                    the simulated equation and its parameters, `name: value` per line
```

## 2. Use it

```python
from fpde_solver import load_dataset, add_noise
u_train, u_test, t = load_dataset('data/space/burgers_a1_b1.5_nu0.05.npz')   # [200, 21, 128], [50, 21, 128], [21]; frame 0 = initial condition
u_noisy = add_noise(u_train, sigma=0.05)                                       # noisy copy of the data
```

`add_noise` adds Gaussian noise with std $\sigma\times$ the std of the snapshots at the same time. Other files: `fpde_toolkit.ipynb` (demo of every option), `tests.ipynb` (validation of the solver and of the dataset tools).

## 3. How the data is generated

**Operators.** $(-\Delta)^{\beta/2}$ is the Fourier multiplier $|k|^\beta$ ($k$ = angular wavenumber), so for $\beta=2$ it is $-\partial_{xx}$. $\partial_t^\alpha$ is the Caputo derivative
$$\partial_t^\alpha u(t)=\frac1{\Gamma(1-\alpha)}\int_0^t (t-s)^{-\alpha}\,\partial_s u(s)\,ds,$$
which has memory of the whole past; for $\alpha=1$ it is $\partial_t$.

**Space.** Fourier pseudo-spectral on $N$ points: in Fourier space, mode $k$ obeys $\partial_t^\alpha \hat u_k = -\lambda_k\hat u_k + \hat N_k(u)$ with $\lambda_k=\nu|k|^\beta$ and $N=-c\,\partial_x(u^2/2)$ (computed on the grid, 2/3 dealiasing).

**Time.** Caputo L1 scheme with step $\tau=T/N_t$ and weights $b_j=\big((j+1)^{1-\alpha}-j^{1-\alpha}\big)/\Gamma(2-\alpha)$:
$$(b_0+\tau^\alpha\lambda_k)\,\hat u_k^{n}=b_0\hat u_k^{n-1}-\sum_{j=1}^{n-1}b_j\big(\hat u_k^{n-j}-\hat u_k^{n-j-1}\big)+\tau^\alpha\hat N_k(u^{n-1}).$$
The linear part is implicit, the nonlinear part explicit (first order). The memory sum makes the cost $O(N_t^2)$. The effective explicit step is $\tau^\alpha\Gamma(2-\alpha)$, so small $\alpha$ and small $\beta$ need more steps.

**Exact solution** (only $c=0$): $\hat u_k(t)=E_\alpha(-\lambda_k t^\alpha)\,\hat u_k(0)$, with $E_\alpha$ the Mittag-Leffler function; used in `tests.ipynb`.

**Initial conditions.** Random Fourier series with amplitudes $|k|^{-p}$ and random phases, zero mean, unit RMS; sample $i$ uses seed `seed + i`.

**Saving.** The equation is solved on `--N` points and saved on `--N-out` points by exact spectral truncation, in float32.

**Smoothness.** For $\beta<2$ the dissipation damps high frequencies less than the Laplacian and Burgers steepens fronts, so a coarse grid gives wiggles. A finer `--N`, a larger `--nu` or a larger `--p` smooths the fields.

## 4. Adding a PDE

Not implemented yet, but the code is ready for it. A PDE is an `Equation` in `fpde_solver.py`: the implicit linear symbol `lam(k, beta)`, the explicit Fourier-space term `nl(u, k, mask)`, its initial conditions `ic`, and `params`, `text`, `tag` for the saved files. Write a constructor like `burgers()`, register it in `EQUATIONS`, and add its options to `main.py`, named like the constructor arguments; then `--equation <name>` selects it. The solver, the dataset tools and the file layout do not change.
