

# Drone MDP Challenge

## Overview

In this project, we will model a drone navigation problem as a **Markov Decision Process (MDP)** and compute an optimal policy using the **Value Iteration** algorithm.

The drone operates in a stochastic grid-world environment containing hazards, portals, repair stations, and a goal state. Our objective is to maximize the expected cumulative reward while safely reaching the target.

---

## Learning Objectives


- Model a real-world problem as an MDP
- Implement the Value Iteration algorithm
- Work with stochastic transitions
- Design and analyze reward functions
- Extract optimal policies from value functions
- Visualize value functions and policies

---

## Environment

The environment is a randomly generated grid with dimensions ranging from **9×9** to **15×15**.

Each state is represented as:

```python
(row, column, damage)
```

where:

- `row` : grid row
- `column` : grid column
- `damage` : drone damage level (0–4)

A damage level of **5** represents a crash and terminates the episode.

---

## Cell Types

| Cell Type | Effect |
|------------|---------|
| Normal Cell | Standard movement cost |
| Wall | Cannot be entered |
| Goal | Terminal state with +300 reward |
| Storm Level 1 | -50 reward, damage +1 |
| Storm Level 2 | -100 reward, damage +1 |
| Storm Level 3 | -125 reward, damage +2 |
| First Aid | +25 reward, damage -1 (single use) |
| Portal | Teleports to paired portal, -5 reward |
| Storm Region | Additional expected penalty |

---

## Actions

The drone can choose one of the following actions:

- North
- South
- East
- West

Actions are **stochastic**. The drone may deviate from the intended direction according to the transition probabilities provided by the environment.

---

## Task

Implement the policy computation function using **Value Iteration**.

Imeplementation must include:

1. Read environment parameters
2. Retrieve all states
3. Initialize state values
4. Perform Value Iteration until convergence
5. Extract the optimal action for each state
6. Return the final policy

Expected output format:

```python
{
    (row, col, damage): action
}
```

---

## Required Visualizations

In addition to the policy implementation, submit a visualization script that generates the following figures.

### 1. Value Function Heatmap

Visualize state values on the grid.

- One heatmap for each damage level

### 2. Convergence Curve

Plot the maximum value update per iteration.

The curve should demonstrate convergence of the Value Iteration process.

### 3. Policy Visualization

Visualize the final policy using directional arrows.

- One figure for each damage level
- Goal and wall cells should be clearly distinguished

---


---

## Running the Project

Install dependencies:

```bash
pip install flask flask-cors
```

Run the server:

```bash
python server.py
```



---





### Technical Report

Report Includes:

- MDP formulation
- Value Iteration implementation details
- Reward modeling
- Transition modeling
- Policy extraction process
- Experimental results
- Required visualizations

---

## Course Information

**Course:** Fundamentals and Applications of Artificial Intelligence  
**Project:** Drone MDP Challenge  

