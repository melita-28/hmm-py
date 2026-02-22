import numpy as np
from hmm import HMM
from visualization import plot_convergence, plot_transition_diagram, plot_emissions
import sys

def main():
    # Example Input: 0, 1, 0, 1, 1, 0 (e.g., Fair/Loaded die or similar)
    # The assignment asks for Observed states sequence and number of hidden states as inputs.
    
    print("--- HMM Baum-Welch Implementation ---")
    
    # Simple default sequence if none provided via command line
    # 0 = Observation A, 1 = Observation B
    obs_seq_input = input("Enter observed sequence (comma separated, e.g., 0,1,0,0,1): ")
    if not obs_seq_input:
        obs_seq = np.array([0, 1, 0, 1, 1, 0])
        print("Using default sequence: 0,1,0,1,1,0")
    else:
        obs_seq = np.array([int(x.strip()) for x in obs_seq_input.split(",")])
        
    n_states_input = input("Enter number of hidden states (default 2): ")
    n_states = int(n_states_input) if n_states_input else 2
    
    n_obs_types = len(np.unique(obs_seq))
    if n_obs_types < 2: n_obs_types = 2 # At least 2 for variety
    
    hmm = HMM(n_states=n_states, n_observations=n_obs_types)
    
    print("\nTraining HMM using Baum-Welch...")
    log_likelihoods = hmm.baum_welch(obs_seq, n_iter=50)
    print(f"Log-Likelihood: {log_likelihoods[0]:.2f} -> {log_likelihoods[-1]:.2f}")
    
    params = hmm.get_params()
    
    print("\n--- Final Parameters ---")
    print("\nTransition Matrix (A):")
    print(params["A"])
    print("\nEmission Matrix (B):")
    print(params["B"])
    print("\nInitial Distribution (pi):")
    print(params["pi"])
    
    # Calculate P(O | lambda) for the final lambda
    alpha, scaling_factors = hmm._forward(obs_seq)
    log_prob_O = -np.sum(np.log(scaling_factors))
    print(f"\nFinal Log-Likelihood: {log_prob_O:.4f}")
    print(f"Final P(O | lambda): {np.exp(log_prob_O):.4e}")
    
    print("\nGenerating Visualizations...")
    plot_convergence(log_likelihoods)
    plot_transition_diagram(params["A"])
    plot_emissions(params["B"])
    
    print("\nDone! Feel free to check the generated images.")

if __name__ == "__main__":
    main()
