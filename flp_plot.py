"""Small plotting helpers: a layout as a picture, a training curve."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_layout(env, cfg, path, title=''):
    geo = env.geometry
    fig, ax = plt.subplots(figsize=(6, 6))
    for corners in geo.site_corners:
        xs = [geo.node_pos[c][0] for c in corners] + [geo.node_pos[corners[0]][0]]
        ys = [geo.node_pos[c][1] for c in corners] + [geo.node_pos[corners[0]][1]]
        ax.plot(xs, ys, color='grey', linewidth=2, zorder=1)
    for m, c in enumerate(cfg):
        site = env.cfg_triple[c][0]
        corners = geo.site_corners[site]
        cx = sum(geo.node_pos[k][0] for k in corners) / 4
        cy = sum(geo.node_pos[k][1] for k in corners) / 4
        ax.text(cx, cy, f'W{m + 1}', ha='center', va='center', fontsize=14,
                bbox=dict(facecolor='lightgrey', edgecolor='none'))
    labels = {}
    for name, p in env.pd_positions(cfg).items():
        labels.setdefault(p, []).append(name)
    for (x, y), names in labels.items():
        ax.scatter([x], [y], s=520, color='tab:red' if len(names) == 1 else 'tab:purple', zorder=2)
        ax.text(x, y, '\n'.join(names), ha='center', va='center', color='white', fontsize=8, zorder=3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_training(history, path, title=''):
    steps = [h['steps'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, [h['episode_return'] for h in history])
    axes[0].set_xlabel('environment steps')
    axes[0].set_ylabel('mean episode return')
    axes[1].plot(steps, [h['best_cost_ratio'] for h in history], label='best layout of the episode')
    axes[1].plot(steps, [h['final_cost_ratio'] for h in history], label='last layout of the episode')
    axes[1].set_xlabel('environment steps')
    axes[1].set_ylabel('cost / cost of the initial layout')
    axes[1].legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
