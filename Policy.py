"""
policy.py — Ultimate Anti-Loop Value Iteration
"""
import os
import json

def compute_policy(api):
    params = api.get_env_params()
    gamma = params['gamma']
    THETA = 1e-5
    MAX_ITER = 2000
    DMG_COST = [1, 2, 5, 9, 15]

    states = api.get_all_states()
    print("[policy.py] Starting Anti-Loop Value Iteration...")

   
    trans_cache = {}

    for s in states:
        if api.is_terminal(s):
            continue
        acts = api.get_possible_actions(s)

        for a in acts:
            t_list = []
            for ns, prob in api.get_transitions(s, a):
                nr, nc, ndmg_capped = ns
                t_dest = api.get_cell_type((nr, nc, 0))

           
                inc = 0
                if t_dest == 'storm':
                    sev = api.get_storm_severity((nr, nc, 0))
                    inc = {1:1, 2:1, 3:2}.get(sev, 1)

                actual_dmg = s[2] + inc
                is_crash = (actual_dmg >= 5)

                if is_crash:
                   
                    t_list.append((ns, prob, -5000.0))
                    continue

               
                if t_dest == 'medkit':
                    r_val = 0.0 # Free step (no step cost)
                    wall_hit = (s[0] == nr and s[1] == nc)
                    if wall_hit: r_val -= 3.0
                    
                    if api.is_in_storm_zone((nr, nc, 0)):
                        z = api.get_storm_zone()
                        if z: r_val -= z['eExpected']

                    new_dmg = max(s[2] - 1, 0)
                    ns_mod = (nr, nc, new_dmg)
                    t_list.append((ns_mod, prob, r_val))

                else:
                    # Standard transition
                    r_val = api.get_reward(s, a, ns)
                    t_list.append((ns, prob, r_val))

            trans_cache[(s, a)] = t_list

    
    # VALUE ITERATION
   
    V = {s: 0.0 for s in states}
    conv_history = []

    for i in range(MAX_ITER):
        delta = 0.0
        V_new = {}
        for s in states:
            if api.is_terminal(s):
                V_new[s] = 0.0
                continue

            acts = api.get_possible_actions(s)
            if not acts:
                V_new[s] = V[s]
                continue

            best_q = -float('inf')
            for a in acts:
                q = sum(prob * (r + gamma * V.get(nxt, 0.0)) for nxt, prob, r in trans_cache[(s, a)])
                if q > best_q:
                    best_q = q

            V_new[s] = best_q
            delta = max(delta, abs(best_q - V[s]))

        V = V_new
        conv_history.append(delta)
        if delta < THETA:
            print(f"[policy.py] Converged in {i+1} iters (Delta: {delta:.2e})")
            break

 
    # POLICY EXTRACTION
    
    policy = {}
    for s in states:
        if api.is_terminal(s):
            continue
        acts = api.get_possible_actions(s)
        if not acts:
            continue

        best_q = -float('inf')
        best_a = None

        for a in acts:
            q = sum(prob * (r + gamma * V.get(nxt, 0.0)) for nxt, prob, r in trans_cache[(s, a)])
            if q > best_q:
                best_q = q
                best_a = a

        if best_a:
            policy[s] = best_a

    # Save visualization artifacts
    _save_data_for_visualizer(V, policy, conv_history, params, api, states, params['rows'], params['cols'])

    return policy

def _save_data_for_visualizer(V, policy, convergence_history, params, api, states, rows, cols):
    is_terminal = {s: api.is_terminal(s) for s in states}
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        conv_path = os.path.join(base_dir, "convergence_data.json")
        with open(conv_path, "w") as f:
            json.dump(convergence_history, f)

        MAX_DAMAGE = 4
        grid_info = []
        for s in states:
            r, c, d = s
            cell_type = api.get_cell_type(s)
            grid_info.append({
                "r": r, "c": c, "d": d,
                "cell": cell_type,
                "V": V.get(s, 0.0),
                "action": policy.get(s, None),
                "terminal": is_terminal[s],
            })

        val_path = os.path.join(base_dir, "value_data.json")
        with open(val_path, "w") as f:
            json.dump({
                "rows": rows, "cols": cols,
                "max_damage": MAX_DAMAGE,
                "params": params, "states": grid_info,
            }, f)
        print(f"[policy.py] Saved convergence_data.json and value_data.json")
    except Exception as e:
        print(f"[policy.py] Warning: could not save visualizer data: {e}")