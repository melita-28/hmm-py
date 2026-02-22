# HMM Baum-Welch Algorithm Implementation

This repository contains an implementation of the Hidden Markov Model (HMM) training using the Baum-Welch algorithm (Expectation-Maximization) from scratch in Python.

## Student Information
- **Name:** Melita Mariam Mathew
- **University Register Number:** TCR24CS046

## Assignment Requirements
- [x] Implement Baum-Welch algorithm.
- [x] Inputs: Observed states sequence and number of hidden states.
- [x] Outputs: Probability of state transitions (A), Emission Matrix (B), Initial Distribution ($\pi$), and $P(O|\lambda)$.
- [x] Visualization: Convergence plot ($P(O|\lambda)$ over iterations) and State Transition Diagram.

## Project Structure
- `hmm.py`: Contains the `HMM` class with Forward, Backward, and Baum-Welch implementations.
- `visualization.py`: Utilities for plotting convergence and transition diagrams.
- `main.py`: Interactive entry point to run the algorithm.

## How to Run

### Prerequisites
- Python 3.8+
- [Optional but recommended] Graphviz installed on your system (for transition diagrams).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/melita-28/hmm-py.git
   cd hmm-baum-welch
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib graphviz
   ```

### Running the Program
Run the following command and follow the prompts:
```bash
python main.py
```
- It will ask for an observed sequence (e.g., `0,1,0,1,1,0`).
- It will ask for the number of hidden states (e.g., `2`).

## Visualizations
The program generates the following files after execution:
- `convergence.png`: Shows how the log-likelihood $P(O|\lambda)$ changes over iterations.
- `transitions.png`: A state transition diagram showing probabilities between states.
- `emissions.png`: A heatmap of the emission matrix.

## Troubleshooting

### Graphviz `ExecutableNotFound` Error
If you see an error saying `failed to execute WindowsPath('dot')`, it means the Graphviz system software is not in your PATH.
1. Download Graphviz from [graphviz.org](https://graphviz.org/download/).
2. During installation, select **"Add Graphviz to the system PATH for all users"**.
3. Restart your terminal/IDE.
