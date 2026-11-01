"""Generate fractional PDE datasets, periodic on [0, L); default equation (burgers):  d_t^alpha u + c u u_x = -nu (-Delta)^(beta/2) u.

    python main.py --type space --n-samples 250
    python main.py --alpha 0.8 --beta 1.7 --nt 1600 --n-samples 100 --out data
    python main.py --type classical                       # integer orders (alpha = 1, beta = 2): classical viscous Burgers
    python main.py --alpha 1 --beta 2 --c 0               # integer orders, linear: heat equation u_t = nu u_xx
    python main.py --alpha 1 --beta 1.2 --nu 0.2          # low beta needs a larger nu to stay smooth
    python main.py --alpha 0.6 --beta 1.5 --check         # only print the smallest stable nt

Layout: data/<kind>/ with
    <equation>_a<alpha>_b<beta>_<params>.npz     u [n_samples, frames, N_out] (frame 0 = initial condition, last n_test = test set), t, x, settings;
                                     read it with fpde_solver.load_dataset(path) -> u_train, u_test, t
    simulations/<name>.png           image of the first sample
    equations/<name>.txt             the simulated equation and its parameters, one `name: value` per line
kind = space | time | space-time | classical (from the final alpha, beta).
"""
import argparse
import inspect
import os

import matplotlib
matplotlib.use('Agg')                                    # no display needed
import matplotlib.pyplot as plt

from fpde_solver import EQUATIONS, make_dataset, min_stable_steps

# candidates for --check; finite does not mean accurate (first order in time), so compare a run with 2x more steps
STEPS = (800, 1600, 3200, 6400)
# default (alpha, beta) of each type
TYPES = {'classical': (1.0, 2.0), 'space': (1.0, 1.5), 'time': (0.6, 2.0), 'space-time': (0.6, 1.5)}

# ---------------------------------------------------------------- arguments
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--equation', choices=EQUATIONS, default='burgers', help='which PDE (see EQUATIONS in fpde_solver.py)')
ap.add_argument('--type', choices=TYPES, default='space', help='sets the default alpha, beta')
ap.add_argument('--alpha', type=float, help='Caputo time order in (0, 1] (overrides --type)')
ap.add_argument('--beta', type=float, help='space order of the dissipation in (0, 2] (overrides --type)')
ap.add_argument('--nu', type=float, default=0.05, help='burgers: viscosity')
ap.add_argument('--c', type=float, default=1.0, help='burgers: nonlinearity (0: linear problem)')
# a new PDE: add its options here, named like the arguments of its constructor
ap.add_argument('--n-samples', type=int, default=250, help='trajectories')
ap.add_argument('--n-test', type=int, default=50, help='the last n-test trajectories are the test set')
ap.add_argument('--N', type=int, default=256, help='solver grid points (beta < 2 needs a fine grid to be smooth)')
ap.add_argument('--N-out', type=int, default=128, help='saved grid points (<= N): solve fine, save coarse')
ap.add_argument('--L', type=float, default=6.283185307179586, help='domain length')
ap.add_argument('--T', type=float, default=1.0, help='final time')
ap.add_argument('--nt', type=int, default=1600, help='solver time steps, fixed (small alpha or beta, large N need more: see --check)')
ap.add_argument('--frames', type=int, default=21, help='saved frames per trajectory (incl. t=0); nt must be a multiple of frames-1')
ap.add_argument('--p', type=float, default=2.0, help='initial condition spectrum slope |k|^-p (small = rougher)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), help='output folder')
ap.add_argument('--check', action='store_true', help='only print the smallest stable nt, do not generate')
a = ap.parse_args()

# ---------------------------------------------------------------- setup
alpha, beta = (a.alpha or TYPES[a.type][0]), (a.beta or TYPES[a.type][1])
build = EQUATIONS[a.equation]
eq = build(**{k: getattr(a, k) for k in inspect.signature(build).parameters})      # constructor arguments come from the options of the same name
# --check: smallest stable nt among STEPS (nothing is generated)
if a.check:
    print(f'alpha={alpha}, beta={beta}, N={a.N}: smallest stable nt = {min_stable_steps(eq, alpha, beta, a.T, STEPS, a.N, a.L, a.p, a.seed)}')
    raise SystemExit

nt = a.nt
if nt % (a.frames - 1):
    ap.error(f'nt ({nt}) must be a multiple of --frames - 1 ({a.frames - 1})')

# output names: type from the final orders
kind = 'classical' if (alpha, beta) == (1, 2) else 'space' if alpha == 1 else 'time' if beta == 2 else 'space-time'
name = f'{eq.name}_a{alpha:g}_b{beta:g}_{eq.tag}'
root = os.path.join(a.out, kind)
for sub in ('simulations', 'equations'):
    os.makedirs(os.path.join(root, sub), exist_ok=True)

# ---------------------------------------------------------------- dataset
try:
    d = make_dataset(eq, alpha, beta, a.n_samples, a.N, a.L, a.T, nt, nt // (a.frames - 1), a.p, a.seed, path=os.path.join(root, name + '.npz'), N_out=a.N_out, n_test=a.n_test)
except RuntimeError as e:                                # a sample blew up: raise --nt (see --check)
    ap.exit(1, f'error: {e}\n')

# ---------------------------------------------------------------- image: first sample, u(x, t) and 3 profiles
u, t, x = d['u'][0], d['t'], d['x']
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={'width_ratios': [1.3, 1]})
im = ax.imshow(u, origin='lower', aspect='auto', extent=[0, a.L, 0, a.T], cmap='RdBu_r', interpolation='bilinear', vmin=-abs(u).max(), vmax=abs(u).max())
ax.set(xlabel='x', ylabel='t', title=f'{kind}: alpha={alpha:g}, beta={beta:g}')
fig.colorbar(im, ax=ax, label='u')

for k in (0, len(t) // 2, len(t) - 1):                   # t = 0, T/2, T
    ax2.plot(x, u[k], label=f't={t[k]:.2g}')
ax2.set(xlabel='x', ylabel='u', title='profiles')
ax2.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(root, 'simulations', name + '.png'), dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- equation file: one "name: value" per line
params = dict(equation=eq.text, type=kind, alpha=alpha, beta=beta, **eq.params, L=a.L, T=a.T, N_solve=a.N, N_saved=a.N_out, nt=nt, frames=a.frames, ic_p=a.p, seed=a.seed, n_train=a.n_samples - a.n_test, n_test=a.n_test, dataset=name + '.npz')
with open(os.path.join(root, 'equations', name + '.txt'), 'w') as f:
    f.write(''.join(f'{k}: {v:g}\n' if isinstance(v, float) else f'{k}: {v}\n' for k, v in params.items()))
print(f'{os.path.join(root, name)}: u {d["u"].shape}, t in [0, {d["t"][-1]:g}], nt = {nt}')
