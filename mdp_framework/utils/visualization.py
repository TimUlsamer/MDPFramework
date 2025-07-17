# mdp_framework/utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt

def plot_discrete_mdp(mdp, max_edges_per_node=4, show_rewards=True, figsize=(8, 6)):
    """
    Visualisiert ein diskretes MDP als gerichteten Graphen.
    Zeigt die wichtigsten Übergänge pro Zustand/Aktion.
    """
    import networkx as nx

    G = nx.MultiDiGraph()
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            probs = mdp.P[s, a]
            # Zeige nur stärkste Übergänge (max_edges_per_node)
            idx = np.argsort(probs)[-max_edges_per_node:]
            for s2 in idx:
                if probs[s2] > 0.01:  # kleine Kanten ausblenden
                    label = f"a={a}, p={probs[s2]:.2f}"
                    if show_rewards:
                        label += f", r={mdp.R[s,a]:.2f}"
                    G.add_edge(s, s2, label=label)

    pos = nx.spring_layout(G)
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_weight='bold', arrows=True)
    edge_labels = {(u,v,k): d["label"] for u,v,k,d in G.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Diskretes MDP")
    plt.tight_layout()
    plt.show()

def plot_continuous_reward(mdp, state_grid=None, action=None, resolution=50, figsize=(7,6)):
    """
    Zeigt die Reward-Landschaft für ein 2D-Continuous-MDP und eine feste Aktion (default: Null-Aktion).
    """
    assert mdp.state_dim == 2, "Nur für 2D-State-Space"
    if action is None:
        action = np.zeros(mdp.action_dim)
    if state_grid is None:
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = state_grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s = np.array([X[i, j], Y[i, j]])
            Z[i, j] = mdp.reward_func(s, action)

    plt.figure(figsize=figsize)
    plt.contourf(X, Y, Z, levels=25, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.title("Reward-Landschaft für feste Aktion")
    plt.xlabel("State[0]")
    plt.ylabel("State[1]")
    plt.tight_layout()
    plt.show()

def plot_continuous_policy(mdp, policy_func, resolution=20, action_scale=0.3, figsize=(7,7)):
    """
    Zeichnet ein Policy-Vektorfeld für 2D-Stetig-Zustand, beliebige Aktionsdim (Pfeile).
    """
    assert mdp.state_dim == 2, "Nur für 2D-State-Space"
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s = np.array([X[i, j], Y[i, j]])
            a = policy_func(s)
            U[i, j] = a[0]
            V[i, j] = a[1] if mdp.action_dim > 1 else 0

    plt.figure(figsize=figsize)
    plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1/action_scale, color="red")
    plt.title("Policy-Vektorfeld")
    plt.xlabel("State[0]")
    plt.ylabel("State[1]")
    plt.tight_layout()
    plt.show()

def plot_trajectory(mdp, policy_func=None, start_state=None, T=30, action_noise=0.0, color='blue'):
    """
    Simuliert und plottet eine Trajektorie (nur für 2D-State-Space sinnvoll!).
    """
    assert mdp.state_dim == 2, "Nur für 2D-State-Space"
    s = mdp.sample_state() if start_state is None else np.array(start_state)
    xs, ys = [s[0]], [s[1]]
    for t in range(T):
        if policy_func:
            a = policy_func(s)
        else:
            a = mdp.sample_action()
        a = a + np.random.randn(*a.shape) * action_noise if action_noise > 0 else a
        s, _ = mdp.step(a)
        xs.append(s[0])
        ys.append(s[1])
    plt.plot(xs, ys, marker="o", color=color)
    plt.title("MDP-Trajektorie")
    plt.xlabel("State[0]")
    plt.ylabel("State[1]")
    plt.tight_layout()
    plt.show()

# Optional: Plot-Wrapper für bequeme Auswahl nach Typ
def visualize_mdp(mdp, **kwargs):
    if hasattr(mdp, "n_states"):  # Diskret
        plot_discrete_mdp(mdp, **kwargs)
    elif hasattr(mdp, "state_dim"):
        if mdp.state_dim == 2:
            plot_continuous_reward(mdp)
        else:
            print("Visualisierung nur für 2D-Stetig-MDP implementiert.")
    else:
        raise TypeError("Unbekannter MDP-Typ.")

# Mini-Test
if __name__ == "__main__":
    from mdp_framework.generators.discrete_generator import random_discrete_mdp
    from mdp_framework.generators.continuous_generator import random_continuous_mdp

    print("Diskretes MDP-Beispiel:")
    mdp_d = random_discrete_mdp(5, 2)
    plot_discrete_mdp(mdp_d)

    print("Stetiges 2D-MDP-Beispiel:")
    mdp_c = random_continuous_mdp(2, 2)
    plot_continuous_reward(mdp_c)
    # Beispiel Policy: Greedy (immer Aktion, die auf 0 zeigt)
    plot_continuous_policy(mdp_c, policy_func=lambda s: np.zeros(mdp_c.action_dim))
    plot_trajectory(mdp_c)
