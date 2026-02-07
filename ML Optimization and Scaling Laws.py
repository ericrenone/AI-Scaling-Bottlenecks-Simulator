import numpy as np
import matplotlib.pyplot as plt

# ================================
# Simulation Parameters
# ================================
dim = 50
steps = 15000
base_lr = 0.01
noise_scale = 1.0
eigvals = np.linspace(0.1, 5.0, dim)
H = np.diag(eigvals)

E_factor = 2
e_flop_real = 1e-9
f_bits_per_flop = 32
T = 300
kB = 1.380649e-23

epochs = 20
L_inf = 0.01
A = 1.0
alpha = 0.28
B = 1.0
beta = 0.34

N_vals = np.logspace(6, 11, 20)
D_vals = np.logspace(6, 11, 20)
N_grid, D_grid = np.meshgrid(N_vals, D_vals)

# ================================
# Loss Functions
# ================================
def loss(theta):
    return 0.5 * theta @ H @ theta

def grad(theta):
    return H @ theta

def loss_scaling(N, D):
    return L_inf + A / N**alpha + B / D**beta

Loss_grid = loss_scaling(N_grid, D_grid)
Compute_grid = N_grid * D_grid * epochs * E_factor
Energy_real = Compute_grid * e_flop_real
E_landauer = kB * T * np.log(2) * f_bits_per_flop * Compute_grid

# ================================
# SGD with Energy Tracking
# ================================
def run_sgd_energy(lr, batch_size, steps=steps):
    np.random.seed(42)
    theta = np.random.randn(dim)
    losses = []
    compute_cost = []
    energy_cost = []
    flops_per_step = batch_size * dim * E_factor
    for step in range(steps):
        g = grad(theta)
        noise = np.sqrt(2 * lr / batch_size) * noise_scale * np.random.randn(dim)
        theta -= lr * g + noise
        current_loss = loss(theta)
        losses.append(current_loss)
        compute_cost.append(flops_per_step)
        energy_cost.append(flops_per_step * e_flop_real)
        if step % 5000 == 0:
            print(f'Step {step}: Loss={current_loss:.4f}, Cum Compute={sum(compute_cost):.2e} FLOPs, Cum Energy={sum(energy_cost):.2e} J')
    print(f'Final Summary for Batch {batch_size}: Min Loss={min(losses):.4f}, Total Compute={sum(compute_cost):.2e} FLOPs, Total Energy={sum(energy_cost):.2e} J')
    return np.array(losses), np.array(compute_cost), np.array(energy_cost)

batch_sizes = [32, 128, 512]
sgd_results = {}
for B in batch_sizes:
    lr = base_lr * (B / batch_sizes[0])
    losses, comp, en = run_sgd_energy(lr, B)
    sgd_results[B] = (losses, np.cumsum(comp), np.cumsum(en))

# ================================
# Plotting
# ================================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, 1], wspace=0.3, hspace=0.4)

# ---- Top Row: SGD Convergence ----
ax1 = fig.add_subplot(gs[0, 0])
for B, (losses, _, _) in sgd_results.items():
    ax1.plot(losses, label=f"Batch {B} (LR scaled)")
ax1.set_yscale("log")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.set_title("SGD Convergence vs Steps (Linear LR Scaling)")
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(gs[0, 1])
for B, (losses, comp, _) in sgd_results.items():
    ax2.plot(comp, losses, label=f"Batch {B}")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Cumulative Compute (FLOPs)")
ax2.set_ylabel("Loss")
ax2.set_title("Loss vs Compute Cost")
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(gs[0, 2])
for B, (losses, _, en) in sgd_results.items():
    ax3.plot(en, losses, label=f"Batch {B}")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("Cumulative Energy (Joules)")
ax3.set_ylabel("Loss")
ax3.set_title("Loss vs Energy Cost")
ax3.legend()
ax3.grid(True)

# ---- Middle Row: Scaling Laws ----
ax4 = fig.add_subplot(gs[1, 0:2], projection='3d')
ax4.plot_surface(np.log10(N_grid), np.log10(D_grid), Loss_grid, cmap='viridis', edgecolor='none')
ax4.set_xlabel('log10(Model Params N)')
ax4.set_ylabel('log10(Data Size D)')
ax4.set_zlabel('Loss')
ax4.set_title('Loss Surface (Scaling Law)')

ax5 = fig.add_subplot(gs[1, 2], projection='3d')
ax5.plot_surface(np.log10(Compute_grid), np.log10(D_grid), Loss_grid, cmap='plasma', edgecolor='none')
ax5.set_xlabel('log10(Compute FLOPs)')
ax5.set_ylabel('log10(Data Size D)')
ax5.set_zlabel('Loss')
ax5.set_title('Loss vs Compute & Data')

# ---- Bottom Row: Energy Analysis ----
ax6 = fig.add_subplot(gs[2, 0])
im6 = ax6.contourf(np.log10(Energy_real), np.log10(D_grid), Loss_grid, levels=20, cmap='inferno')
fig.colorbar(im6, ax=ax6, label='Loss', shrink=0.8)
ax6.set_xlabel('log10(Real Energy, J)')
ax6.set_ylabel('log10(Data Size D)')
ax6.set_title('Loss vs Real Energy')
ax6.grid(True)

ax7 = fig.add_subplot(gs[2, 1])
im7 = ax7.contourf(np.log10(E_landauer), np.log10(D_grid), Loss_grid, levels=20, cmap='viridis')
fig.colorbar(im7, ax=ax7, label='Loss', shrink=0.8)
ax7.set_xlabel('log10(Landauer Energy, J)')
ax7.set_ylabel('log10(Data Size D)')
ax7.set_title('Loss vs Thermodynamic Landauer Limit')
ax7.grid(True)

ax8 = fig.add_subplot(gs[2, 2])
im8 = ax8.contourf(np.log10(Energy_real/E_landauer), np.log10(D_grid), Loss_grid, levels=20, cmap='plasma')
fig.colorbar(im8, ax=ax8, label='Energy / Landauer Limit', shrink=0.8)
ax8.set_xlabel('log10(E_real / E_landauer)')
ax8.set_ylabel('log10(Data Size D)')
ax8.set_title('Energy Efficiency Relative to Landauer Limit')
ax8.grid(True)

# ---- Manual Layout Adjustment (Safe for 3D + 2D axes) ----
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.4)

plt.show()
