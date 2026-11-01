"""1D periodic fractional PDEs (Fourier pseudo-spectral in x, Caputo L1 in t).

    d_t^alpha u = -lam(k) u_k + N_k(u),    x in [0, L) periodic,  0 < alpha <= 1,  0 < beta <= 2

  alpha = 1, beta < 2: space-fractional      alpha < 1, beta = 2: time-fractional      alpha < 1, beta < 2: space-time
Equation (choose with burgers()), `beta` is the fractional space order of the dissipation:
  burgers     d_t^a u + c u u_x = -nu (-Delta)^(beta/2) u       lam = nu |k|^beta
(-Delta)^(beta/2) is the Fourier multiplier |k|^beta (k = angular wavenumber).
"""
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from math import gamma
import mpmath as mp
from scipy.special import rgamma


# An equation is defined by its explicit/implicit split in Fourier space. To add a PDE: write a constructor like burgers() that returns an
# Equation, and register it in EQUATIONS (below). solve(), make_dataset() and main.py need no change.
@dataclass
class Equation:
    name: str
    lam: Callable          # lam(k, beta): linear symbol, implicit in the scheme
    nl: Callable           # nl(u, k, mask): Fourier transform of the explicit term (nonlinearity / reaction), already dealiased
    ic: Callable           # ic(B, N, p, seed): random initial conditions that suit the equation
    params: dict = field(default_factory=dict)   # saved in the dataset and in the equation .txt
    text: str = ''         # the equation, written in the .txt
    tag: str = ''          # parameters in the file name, e.g. 'nu0.05'


def burgers(nu=0.05, c=1.0):
    return Equation('burgers', lambda k, beta: nu * k ** beta, lambda u, k, m: -c * 1j * k * m * np.fft.rfft(0.5 * u * u, axis=-1), lambda B, N, p, seed: random_ic(B, N, p, seed=seed), dict(nu=nu, c=c),
                    'd_t^alpha u + c u u_x = -nu (-Delta)^(beta/2) u', f'nu{nu:g}')


EQUATIONS = {'burgers': burgers}      # name -> constructor; its arguments are filled from the main.py options of the same name


def wavenumbers(N, L=2 * np.pi):
    return 2 * np.pi / L * np.fft.rfftfreq(N, d=1.0 / N)


def l1_weights(alpha, Nt):
    """b_j = ((j+1)^(1-a) - j^(1-a)) / Gamma(2-a), j = 0..Nt-1  (alpha = 1: b_0 = 1, others 0 -> backward Euler)."""
    j = np.arange(Nt, dtype=float)
    return ((j + 1) ** (1 - alpha) - np.where(j > 0, j ** (1 - alpha), 0.0)) / gamma(2 - alpha)


def solve(u0, alpha, beta, T, Nt, eq, L=2 * np.pi, save_every=1):
    """u0: [B, N] real initial fields, eq: an Equation. Returns (t [Nt//save_every+1], u [B, Nt//save_every+1, N]), nan after a blow-up.
    L1 scheme with uniform step tau = T/Nt, per Fourier mode (linear part implicit, nonlinear/reaction part explicit):
        (b0 + tau^a lam) u^n = b0 u^{n-1} - sum_{j=1}^{n-1} b_j (u^{n-j} - u^{n-j-1}) + tau^a N(u^{n-1})."""
    u0 = np.atleast_2d(u0)
    B, N = u0.shape
    k = wavenumbers(N, L)
    mask = (np.arange(len(k)) <= N // 3)                                  # 2/3 dealiasing
    tau = T / Nt
    ta = tau ** alpha
    b = l1_weights(alpha, Nt)
    den = b[0] + ta * eq.lam(k, beta)
    U = np.zeros((Nt + 1, B, len(k)), complex)       # u^n in Fourier space
    D = np.zeros_like(U)                              # D[n] = u^n - u^{n-1}
    U[0] = np.fft.rfft(u0, axis=-1)
    for n in range(1, Nt + 1):
        rhs = b[0] * U[n - 1] + ta * eq.nl(np.fft.irfft(U[n - 1], n=N, axis=-1), k, mask)
        if n > 1:
            rhs = rhs - np.einsum('j,jbk->bk', b[1:n], D[n - 1:0:-1])
        U[n] = rhs / den
        D[n] = U[n] - U[n - 1]
        if not np.all(np.isfinite(U[n])) or np.abs(U[n]).max() > 1e8:      # blow-up: stop, pad with nan
            U[n:] = np.nan
            break
    idx = np.arange(0, Nt + 1, save_every)
    return idx * tau, np.fft.irfft(U[idx], n=N, axis=-1).transpose(1, 0, 2)


def mittag_leffler(alpha, z):
    """E_alpha(z) for complex z with Re z <= 0, 0 < alpha <= 1. Power series in high precision (the terms are huge and cancel)
    when it is cheap, otherwise the asymptotic expansion -sum_m z^-m / Gamma(1 - alpha m) (algebraic decay), truncated at its smallest term."""
    z = complex(z)
    if abs(z) < 1e-14:
        return 1.0
    if abs(z) ** (1 / alpha) < 400:
        mp.mp.dps = 40 + int(abs(z) ** (1 / alpha) / 2.3)
        a, zz, s, n, n0 = mp.mpf(alpha), mp.mpc(z), mp.mpf(0), 0, 3 * abs(z) ** (1 / alpha) + 10
        while True:
            term = zz ** n / mp.gamma(a * n + 1)
            s += term
            n += 1
            if n > n0 and abs(term) < mp.mpf(10) ** -20:
                return complex(s)
    s, prev = 0.0, np.inf
    for m in range(1, 60):
        term = z ** -m * rgamma(1 - alpha * m)
        if abs(term) > prev and m > 2:
            break
        s, prev = s - term, abs(term) if term != 0 else prev
    return complex(s)


def exact_linear(u0, alpha, beta, t, eq, L=2 * np.pi):
    """Exact solution when the explicit term is off (burgers with c=0): u_k(t) = E_alpha(-lam_k t^alpha) u_k(0).
    u0: [B, N]; returns [B, len(t), N]."""
    u0 = np.atleast_2d(u0)
    N = u0.shape[-1]
    lam = eq.lam(wavenumbers(N, L), beta)
    U0 = np.fft.rfft(u0, axis=-1)
    E = np.array([[mittag_leffler(alpha, -l * ti ** alpha) for l in lam] for ti in t])      # [len(t), K]
    return np.fft.irfft(U0[:, None, :] * E[None], n=N, axis=-1)


def burgers_ref(u0, nu, c, T, n_out, L=2 * np.pi, dt=2e-4):
    """Classical Burgers u_t + c u u_x = nu u_xx (alpha = 1, beta = 2): integrating-factor RK4. Returns [B, n_out+1, N]."""
    u0 = np.atleast_2d(u0)
    N = u0.shape[-1]
    k = wavenumbers(N, L)
    ik = 1j * k * (np.arange(len(k)) <= N // 3)
    sub = max(1, int(np.ceil(T / n_out / dt)))
    h = T / n_out / sub
    E, E2 = np.exp(-nu * k ** 2 * h), np.exp(-nu * k ** 2 * h / 2)
    f = lambda w: -c * ik * np.fft.rfft(0.5 * np.fft.irfft(w, n=N, axis=-1) ** 2, axis=-1)
    w, out = np.fft.rfft(u0, axis=-1), [u0]
    for _ in range(n_out):
        for _ in range(sub):
            a = h * f(w); b = h * f(E2 * (w + a / 2)); cc = h * f(E2 * w + b / 2); d = h * f(E * w + E2 * cc)
            w = E * w + (E * a + 2 * E2 * (b + cc) + d) / 6
        out.append(np.fft.irfft(w, n=N, axis=-1))
    return np.stack(out, axis=1)


def random_ic(B, N, p=2.0, rms=1.0, mean=0.0, seed=0):
    """Random Fourier series, amplitude ~ |k|^-p, random phases, zero mean (+ `mean`), normalised to RMS `rms` (before adding the mean)."""
    rng = np.random.default_rng(seed)
    K = N // 2 + 1
    amp = np.zeros(K)
    amp[1:] = np.arange(1, K) ** (-float(p))
    U = amp * np.exp(2j * np.pi * rng.random((B, K)))
    u = np.fft.irfft(U, n=N, axis=-1)
    return rms * u / np.sqrt((u ** 2).mean(-1, keepdims=True)) + mean


def rel_l2(a, b):
    """Relative L2 error (per time step if the last axis is x), in %."""
    return 100 * np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def min_stable_steps(eq, alpha, beta, T=1.0, steps=(200, 400, 800, 1600, 3200), N=128, L=2 * np.pi, p=2, seed=0):
    """Stability check: smallest Nt in `steps` for which 2 random initial conditions stay finite (None if none does). Finite != accurate."""
    u0 = eq.ic(2, N, p, seed)
    for n in steps:
        if np.isfinite(solve(u0, alpha, beta, T, n, eq, L, save_every=n)[1]).all():
            return n
    return None


def make_dataset(eq, alpha, beta, n_samples, N=128, L=2 * np.pi, T=1.0, Nt=800, save_every=40, p=2, seed=0, chunk=50, path=None, N_out=None, n_test=50):
    """Trajectories of equation `eq` for one (alpha, beta): u [n_samples, Nt//save_every+1, N_out] with the initial conditions eq.ic(., p, seed).
    Solved on N points, saved on N_out <= N points (exact spectral truncation: solve fine, save coarse); the last n_test samples are the test set.
    Raises if the run blows up at this Nt (see min_stable_steps). Saved as a compressed .npz (float32) if `path` is given."""
    u = np.concatenate([solve(eq.ic(min(chunk, n_samples - i), N, p, seed + i), alpha, beta, T, Nt, eq, L, save_every)[1] for i in range(0, n_samples, chunk)])
    if not np.isfinite(u).all():
        raise RuntimeError(f'{eq.name} (alpha={alpha}, beta={beta}) blew up at Nt={Nt}: raise Nt (see min_stable_steps) or change the parameters')
    N_out = N_out or N
    if N_out < N:
        u = np.fft.irfft(np.fft.rfft(u, axis=-1)[..., :N_out // 2 + 1] * (N_out / N), n=N_out, axis=-1)
    out = dict(u=u.astype(np.float32), t=np.arange(0, Nt + 1, save_every) * T / Nt, x=np.arange(N_out) * L / N_out, equation=eq.name, alpha=alpha, beta=beta, p=p, seed=seed, T=T, Nt=Nt, L=L, N_solve=N, n_test=n_test, **eq.params)
    if path:
        np.savez_compressed(path, **out)
    return out


def load_dataset(path):
    """(u_train, u_test, t) from a file of make_dataset: u [n, frames, N] (frame 0 = initial condition), t [frames]."""
    d = np.load(path)
    n_test = int(d['n_test'])
    return d['u'][:-n_test], d['u'][-n_test:], d['t']


def add_noise(u, sigma, seed=0):
    """Noisy copy of u [n, frames, N]: Gaussian, std = sigma * std of the
    snapshots at the same time (over samples and x), so a frame that has decayed gets proportionally less noise."""
    rng = np.random.default_rng(seed)
    return u + sigma * u.std(axis=(0, 2), keepdims=True) * rng.standard_normal(u.shape).astype(u.dtype)
