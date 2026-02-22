import matplotlib.pyplot as plt
from graphviz import Digraph
import numpy as np
import os

def plot_convergence(log_likelihoods, output_path="convergence.png"):
    plt.figure(figsize=(10, 6))
    
    # If the first jump is huge, we might want to plot from iteration 1 to see detail
    if len(log_likelihoods) > 5 and (log_likelihoods[1] - log_likelihoods[0]) > (log_likelihoods[-1] - log_likelihoods[1]) * 10:
        plt.plot(range(1, len(log_likelihoods)), log_likelihoods[1:], marker='o', label="Detail (Iter 1+)")
        plt.title("HMM Convergence: Log-Likelihood vs Iterations (Zoomed)")
        print("Note: Convergence plot zoomed in to ignore initial outlier.")
    else:
        plt.plot(range(len(log_likelihoods)), log_likelihoods, marker='o')
        plt.title("HMM Convergence: Log-Likelihood vs Iterations")
    
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Convergence plot saved to {output_path}")

def plot_transition_diagram(A, state_names=None, output_path="transitions"):
    n_states = A.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]
    
    dot = Digraph(comment='HMM State Transitions')
    dot.attr(rankdir='LR')
    
    for i in range(n_states):
        dot.node(state_names[i], state_names[i])
        
    for i in range(n_states):
        for j in range(n_states):
            prob = A[i, j]
            if prob > 0.01:  # Threshold to avoid cluttered diagram
                dot.edge(state_names[i], state_names[j], label=f"{prob:.2f}")
                
    try:
        dot.render(output_path, format='png', cleanup=True)
        print(f"Transition diagram saved to {output_path}.png")
    except Exception as e:
        print(f"\n[WARNING] Could not generate transition diagram image.")
        print(f"Error: {e}")
        print(f"Please ensure Graphviz system software is installed and 'dot' is in your PATH.")
        print(f"The DOT source file has been saved as '{output_path}' which you can render later.")
        dot.save(output_path)

def plot_emissions(B, state_names=None, obs_names=None, output_path="emissions.png"):
    n_states, n_obs = B.shape
    if state_names is None: state_names = [f"S{i}" for i in range(n_states)]
    if obs_names is None: obs_names = [f"O{i}" for i in range(n_obs)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(B, cmap="YlGnBu")
    
    ax.set_xticks(np.arange(n_obs))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(obs_names)
    ax.set_yticklabels(state_names)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(n_states):
        for j in range(n_obs):
            ax.text(j, i, f"{B[i, j]:.2f}", ha="center", va="center", color="black")
            
    ax.set_title("Emission Matrix (B)")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Emission heatmap saved to {output_path}")
