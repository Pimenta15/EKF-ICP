"""
Replication — Eraghi et al. ICCIA 2023
EKF-SLAM vs ICP-EKF-SLAM (Versão Final para Apresentação)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import time
import os

# ── Parameters ───────────────────────────────────────────────────────────────
RNG_SEED  = 42
DT        = 0.025
WHEELBASE = 25.0
SPEED     = 2.0
SIG_V     = 0.3
SIG_G     = np.deg2rad(2.0)
SIG_R     = 0.2
SIG_PHI   = np.deg2rad(2)
MAX_RANGE = 25.0
MAX_STEER = np.deg2rad(30)
N_LM      = 1500
N_RUNS    = 15

Q_m = np.diag([SIG_V**2 * DT**2, SIG_G**2])
R_cov = np.diag([SIG_R**2, SIG_PHI**2]) 

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

# ── Environment ──────────────────────────────────────────────────────────────
def make_map(rng): return rng.uniform(-150, 150, (2, N_LM))
def make_wp():
    wp = []
    for x in np.linspace(-80, 80, 15): wp.append([x, 50.])
    for a in np.linspace(np.pi/2, -np.pi/2, 20): wp.append([80 + 50 * np.cos(a), 50 * np.sin(a)])
    for x in np.linspace(80, -80, 15): wp.append([x, -50.])
    for a in np.linspace(-np.pi/2, np.pi/2, 20): wp.append([-80 - 50 * np.cos(a), 50 * np.sin(a)])
    wp.append(wp[0]); return np.array(wp).T

# ── Motion & Control ─────────────────────────────────────────────────────────
def fmo(s, v, g):
    x, y, th = s; L = v * DT
    if abs(g) < 1e-6: return np.array([x + L * np.cos(th), y + L * np.sin(th), wrap(th)])
    dth = L * np.sin(g) / WHEELBASE; Rt = WHEELBASE / np.tan(g)
    return np.array([x - Rt * np.sin(th) + Rt * np.sin(th + dth),
                     y + Rt * np.cos(th) - Rt * np.cos(th + dth), wrap(th + dth)])

def Fj(s, v, g):
    x, y, th = s; L = v * DT
    if abs(g) < 1e-6: return np.array([[1, 0, -L * np.sin(th)], [0, 1, L * np.cos(th)], [0, 0, 1]])
    dth = L * np.sin(g) / WHEELBASE; Rt = WHEELBASE / np.tan(g)
    return np.array([[1, 0, -Rt * np.cos(th) + Rt * np.cos(th + dth)],
                     [0, 1, -Rt * np.sin(th) + Rt * np.sin(th + dth)], [0, 0, 1]])

def Wj(s, v, g):
    e = 1e-5
    return np.column_stack([(fmo(s, v + e, g) - fmo(s, v - e, g)) / (2 * e),
                            (fmo(s, v, g + e) - fmo(s, v, g - e)) / (2 * e)])

def hobs(rs, lm):
    dx, dy = lm[0] - rs[0], lm[1] - rs[1]; r = np.sqrt(dx * dx + dy * dy)
    return np.array([r, wrap(np.arctan2(dy, dx) - rs[2])]) if r > 1e-9 else None

def ctrl(rs, wp, wi):
    look_ahead_steps = 2
    target_idx = min(wi + look_ahead_steps, wp.shape[1] - 1)
    while wi < wp.shape[1] - 1:
        if np.hypot(wp[0, wi] - rs[0], wp[1, wi] - rs[1]) < 20.: 
            wi += 1; target_idx = min(wi + look_ahead_steps, wp.shape[1] - 1)
        else: break
    dx, dy = wp[0, target_idx] - rs[0], wp[1, target_idx] - rs[1]
    a = wrap(np.arctan2(dy, dx) - rs[2])
    return np.clip(np.arctan2(2 * WHEELBASE * np.sin(a), MAX_RANGE), -MAX_STEER, MAX_STEER), wi

# ── ICP Functions ────────────────────────────────────────────────────────────
def obs_to_body(obs):
    if not obs: return np.zeros((2, 0)), []
    pts = [[z[0] * np.cos(z[1]), z[0] * np.sin(z[1])] for _, z in obs]
    ids = [gid for gid, _ in obs]
    return np.array(pts).T, ids

def icp_svd(Bk, ids_k, Bk1, ids_k1):
    map_k  = {gid: i for i, gid in enumerate(ids_k)}
    map_k1 = {gid: i for i, gid in enumerate(ids_k1)}
    common = sorted(set(map_k) & set(map_k1))
    if len(common) < 2: return 0., np.zeros(2)
    A = np.array([Bk[:,  map_k[g]]  for g in common]).T
    B = np.array([Bk1[:, map_k1[g]] for g in common]).T
    muA = np.mean(A, axis=1, keepdims=True); muB = np.mean(B, axis=1, keepdims=True)
    H = (A - muA) @ (B - muB).T
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0: Vt[1, :] *= -1; Rm = Vt.T @ U.T
    t = (muB - Rm @ muA).flatten()
    angle = np.arctan2(Rm[1, 0], Rm[0, 0])
    return angle, t

def apply_icp_correction(r_prev_updated, angle, t_body):
    Rp = np.array([[np.cos(r_prev_updated[2]), -np.sin(r_prev_updated[2])],
                   [np.sin(r_prev_updated[2]),  np.cos(r_prev_updated[2])]])
    t_global = Rp @ t_body
    return np.array([r_prev_updated[0] + t_global[0],
                     r_prev_updated[1] + t_global[1],
                     wrap(r_prev_updated[2] + angle)])

# ── Trial (SLAM Core) ────────────────────────────────────────────────────────
def trial(lm_map, wp, use_icp, rng, save_history=False):
    x0 = np.array([wp[0,0], wp[1,0], np.pi/6])
    r = x0.copy(); rt = x0.copy()
    S = np.diag([0.01, 0.01, 0.001])
    rmse = []; nees = []; wi = 1
    
    Bprev = np.zeros((2, 0)); ids_prev = []
    r_prev_updated = x0.copy()
    dist_since_icp = 0.0
    slam_map = {}
    
    hist_rt_x, hist_rt_y, hist_est_x, hist_est_y = [], [], [], []

    for step in range(25000):
        g_true, wi = ctrl(rt, wp, wi)
        if wi >= wp.shape[1] - 2: break # Evita o zerinho no final

        rt = fmo(rt, SPEED, g_true + rng.normal(0, SIG_G))

        if save_history and step % 100 == 0:
            hist_rt_x.append(rt[0]); hist_rt_y.append(rt[1])
            hist_est_x.append(r[0]); hist_est_y.append(r[1])

        vn = SPEED + rng.normal(0, SIG_V)
        gn = np.clip(g_true + rng.normal(0, SIG_G), -MAX_STEER, MAX_STEER)
        F = Fj(r, vn, gn); W = Wj(r, vn, gn)
        S = F @ S @ F.T + W @ Q_m @ W.T
        r = fmo(r, vn, gn)
        dist_since_icp += vn * DT

        obs = []
        for gid in range(N_LM):
            z = hobs(rt, lm_map[:, gid])
            if z is not None and z[0] <= MAX_RANGE:
                zn = np.array([z[0] + rng.normal(0, SIG_R), wrap(z[1] + rng.normal(0, SIG_PHI))])
                obs.append((gid, zn))

        if use_icp and obs and dist_since_icp > 0.5:
            Bcurr, ids_curr = obs_to_body(obs)
            if len(ids_prev) >= 2 and len(ids_curr) >= 2:
                angle, t_body = icp_svd(Bcurr, ids_curr, Bprev, ids_prev)
                r_corr = apply_icp_correction(r_prev_updated, angle, t_body)
                alpha = 0.5
                r[0] = alpha * r_corr[0] + (1 - alpha) * r[0]
                r[1] = alpha * r_corr[1] + (1 - alpha) * r[1]
                r[2] = wrap(r[2] + alpha * wrap(r_corr[2] - r[2]))
            r_prev_updated = r.copy()
            Bprev, ids_prev = obs_to_body(obs)
            dist_since_icp = 0.0

        for gid, z_medido in obs:
            if gid not in slam_map:
                lx = r[0] + z_medido[0] * np.cos(r[2] + z_medido[1])
                ly = r[1] + z_medido[0] * np.sin(r[2] + z_medido[1])
                slam_map[gid] = np.array([lx, ly])
                
            lm_pos = slam_map[gid]
            dx, dy = lm_pos[0] - r[0], lm_pos[1] - r[1]
            q = dx**2 + dy**2
            dist = np.sqrt(q)
            
            if dist > 1e-9:
                z_hat = np.array([dist, wrap(np.arctan2(dy, dx) - r[2])])
                H = np.array([[-dx/dist, -dy/dist, 0.0], [dy/q, -dx/q, -1.0]])
                y = z_medido - z_hat; y[1] = wrap(y[1])
                S_inov = H @ S @ H.T + R_cov
                K = S @ H.T @ np.linalg.inv(S_inov)
                r = r + K @ y; r[2] = wrap(r[2])
                S = (np.eye(3) - K @ H) @ S @ (np.eye(3) - K @ H).T + K @ R_cov @ K.T

        if use_icp and obs and len(ids_prev) == 0:
            Bprev, ids_prev = obs_to_body(obs); r_prev_updated = r.copy()

        e = r - rt; e[2] = wrap(e[2])
        rmse.append(np.hypot(e[0], e[1]))
        try: nees.append(e[:2] @ np.linalg.solve(S[:2, :2], e[:2]))
        except: nees.append(np.nan)

    if save_history: return hist_rt_x, hist_rt_y, hist_est_x, hist_est_y
    return np.array(rmse), np.array(nees)

# ── 1. Monte Carlo (Gera as Estatísticas) ────────────────────────────────────
def run_monte_carlo():
    wp = make_wp(); rm = np.random.default_rng(RNG_SEED)
    er, en, ir, inl = [], [], [], []
    for run in range(N_RUNS):
        seed = int(rm.integers(0, 10**6)); lm = make_map(np.random.default_rng(seed))
        re, ne = trial(lm, wp, False, np.random.default_rng(seed + 1))
        ri, ni = trial(lm, wp, True,  np.random.default_rng(seed + 1))
        er.append(re); en.append(ne); ir.append(ri); inl.append(ni)
        print(f"  Run {run+1:2d}/{N_RUNS}  EKF={np.mean(re):.3f}m  ICP-EKF={np.mean(ri):.3f}m")
    
    ml = min(min(len(r) for r in er), min(len(r) for r in ir))
    return (np.mean([r[:ml] for r in er], axis=0), np.nanmean([n[:ml] for n in en], axis=0),
            np.mean([r[:ml] for r in ir], axis=0), np.nanmean([n[:ml] for n in inl], axis=0), wp)

# ── 2. Gráfico Estático ──────────────────────────────────────────────────────
def plot_static(ekf_r, ekf_n, icp_r, icp_n, wp):
    BG, AX, RED, GRN, GR, WH = '#0F0F0F', '#141414', '#D45B5B', '#6DBFA0', '#888', '#E8E8E8'
    fig = plt.figure(figsize=(15, 10), facecolor=BG); gs = GridSpec(2, 3, fig, hspace=0.40, wspace=0.32)
    def sax(ax, t):
        ax.set_facecolor(AX)
        for s in ax.spines.values(): s.set_color('#333')
        ax.tick_params(colors=GR, labelsize=9); ax.xaxis.label.set_color(GR); ax.yaxis.label.set_color(GR)
        ax.set_title(t, color=WH, fontsize=10, pad=7); ax.grid(True, color='#222', lw=0.5)
    
    steps = np.arange(len(ekf_r))
    ax1 = fig.add_subplot(gs[0, :2]); sax(ax1, 'Robot position RMSE along path')
    ax1.plot(steps, ekf_r, color=RED, lw=1.2, label=f'EKF-SLAM (avg {np.mean(ekf_r):.3f} m)')
    ax1.plot(steps, icp_r, color=GRN, lw=1.2, label=f'ICP-EKF-SLAM (avg {np.mean(icp_r):.3f} m)')
    ax1.set_xlabel('Calculation step'); ax1.set_ylabel('RMSE [m]')
    ax1.legend(fontsize=9, framealpha=0.3, facecolor='#1A1A1A', labelcolor='white')
    
    ax2 = fig.add_subplot(gs[1, :2]); sax(ax2, 'Robot position NEES along path')
    nc = 50
    ax2.plot(steps, np.clip(ekf_n, 0, nc), color=RED, lw=1.2, label=f'EKF-SLAM (avg {np.nanmean(ekf_n):.1f})')
    ax2.plot(steps, np.clip(icp_n, 0, nc), color=GRN, lw=1.2, label=f'ICP-EKF-SLAM (avg {np.nanmean(icp_n):.1f})')
    ax2.axhline(5.99, color='#FFAA44', lw=0.8, ls='--', alpha=0.7, label='95% CI bound (χ²₂)')
    ax2.set_xlabel('Calculation step'); ax2.set_ylabel('NEES'); ax2.set_ylim(0, nc)
    ax2.legend(fontsize=9, framealpha=0.3, facecolor='#1A1A1A', labelcolor='white')
    
    ax3 = fig.add_subplot(gs[:, 2]); sax(ax3, 'Robot path (waypoints)')
    ax3.plot(wp[0], wp[1], color='#333333', lw=2, ls='--', label='Planned Path')
    ax3.plot(wp[0, 0], wp[1, 0], 'o', color=GRN, ms=8, label='Start')
    ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]'); ax3.set_aspect('equal')
    ax3.legend(fontsize=9, framealpha=0.3, facecolor='#1A1A1A', labelcolor='white')
    
    pct = (1 - np.mean(icp_r) / np.mean(ekf_r)) * 100
    info = [f'EKF avg RMSE:     {np.mean(ekf_r):.3f} m',
            f'ICP-EKF avg RMSE: {np.mean(icp_r):.3f} m',
            f'RMSE improvement: {pct:.1f}%',
            f'Runs: {N_RUNS}  |  Landmarks: {N_LM}']
    fig.text(0.025, 0.01, '\n'.join(info), color=GR, fontsize=8.5, fontfamily='monospace', va='bottom')
    fig.suptitle('EKF-SLAM vs ICP-EKF-SLAM (Final Results)', color=WH, fontsize=13, y=0.99)
    
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slam_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG); print(f'Gráfico estático salvo → {out}')

# ── 3. Animação GIF ──────────────────────────────────────────────────────────
def generate_gif():
    print("\nSimulando a trajetória para o GIF...")
    wp = make_wp(); rm = np.random.default_rng(RNG_SEED)
    seed = int(rm.integers(0, 10**6)); lm = make_map(np.random.default_rng(seed))
    
    rx_true, ry_true, rx_ekf, ry_ekf = trial(lm, wp, False, np.random.default_rng(seed + 1), save_history=True)
    _, _, rx_icp, ry_icp = trial(lm, wp, True, np.random.default_rng(seed + 1), save_history=True)
    
    print("Montando o GIF...")
    BG, AX, GR, WH = '#0F0F0F', '#141414', '#888', '#E8E8E8'
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(AX); ax.tick_params(colors=GR)
    for s in ax.spines.values(): s.set_color('#333')
    ax.grid(True, color='#222', lw=0.5); ax.set_title("SLAM Dinâmico: EKF vs ICP-EKF", color=WH)
    ax.set_xlim(-150, 150); ax.set_ylim(-90, 90); ax.set_aspect('equal')
    
    ax.scatter(lm[0], lm[1], s=8, c='#444444', alpha=0.5, label='Nuvem de Pontos (Landmarks)')
    ax.plot(wp[0], wp[1], color='#333333', lw=1.5, ls='--', label='Caminho Planejado')
    
    line_true, = ax.plot([], [], color='#5B8FD4', lw=2.5, label='Trajetória Real (Truth)')
    line_ekf,  = ax.plot([], [], color='#D45B5B', lw=2.5, label='EKF-SLAM')
    line_icp,  = ax.plot([], [], color='#6DBFA0', lw=2.5, label='ICP-EKF-SLAM')
    
    pt_true, = ax.plot([], [], 'o', color='#5B8FD4', ms=7)
    pt_ekf,  = ax.plot([], [], 'o', color='#D45B5B', ms=7)
    pt_icp,  = ax.plot([], [], 'o', color='#6DBFA0', ms=7)
    ax.legend(fontsize=9, facecolor='#1A1A1A', labelcolor='white', loc='upper right', framealpha=0.8)

    def init(): return line_true, line_ekf, line_icp, pt_true, pt_ekf, pt_icp

    def update(frame):
        max_f = min(len(rx_true), len(rx_icp))
        if frame >= max_f: frame = max_f - 1
        line_true.set_data(rx_true[:frame], ry_true[:frame])
        line_ekf.set_data(rx_ekf[:frame], ry_ekf[:frame])
        line_icp.set_data(rx_icp[:frame], ry_icp[:frame])
        if frame > 0:
            pt_true.set_data([rx_true[frame-1]], [ry_true[frame-1]])
            pt_ekf.set_data([rx_ekf[frame-1]], [ry_ekf[frame-1]])
            pt_icp.set_data([rx_icp[frame-1]], [ry_icp[frame-1]])
        return line_true, line_ekf, line_icp, pt_true, pt_ekf, pt_icp

    frames = min(len(rx_true), len(rx_icp))
    anim = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slam_animado.gif')
    anim.save(out, writer='pillow', fps=20) 
    print(f"GIF salvo com sucesso → {out}")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 58)
    print('  Gerador Completo: EKF-SLAM vs ICP-EKF-SLAM')
    print('=' * 58)
    t0 = time.time()
    
    # 1. Roda as estatísticas e gera a imagem estática
    ekf_r, ekf_n, icp_r, icp_n, wp = run_monte_carlo()
    plot_static(ekf_r, ekf_n, icp_r, icp_n, wp)
    
    # 2. Gera o GIF animado
    generate_gif()
    
    pct = (1 - np.mean(icp_r) / np.mean(ekf_r)) * 100
    print('\n' + '=' * 58)
    print(f'RESUMO FINAL (Tempo: {time.time() - t0:.1f}s)')
    print(f'EKF-SLAM RMSE:     {np.mean(ekf_r):.3f}m')
    print(f'ICP-EKF-SLAM RMSE: {np.mean(icp_r):.3f}m')
    print(f'Melhoria do ICP:   {pct:.1f}%')
    print('=' * 58)