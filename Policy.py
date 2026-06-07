def compute_policy(api):
    params = api.get_env_params()
    gamma = params['gamma']
    states = api.get_all_states()
    
    trans_cache = {}
    for s in states:
        if api.is_terminal(s): continue
        for a in api.get_possible_actions(s):
            trans_cache[(s, a)] = api.get_transitions(s, a)
    
    print(f"[policy.py] Cache built for {len(trans_cache)} state-action pairs.")
    V = {s: 0.0 for s in states}
    MAX_ITER = 1000
    THETA = 1e-5

    print(f"[policy.py] Starting VI for {len(states)} states...")

    for i in range(MAX_ITER):
        delta = 0.0
        V_new = {}
        for s in states:
            if api.is_terminal(s):
                V_new[s] = 0.0
                continue
            
            best_q = -float('inf')
            for a in api.get_possible_actions(s):
                q = sum(prob * (api.get_reward(s, a, ns) + gamma * V.get(ns, 0.0)) 
                        for ns, prob in trans_cache[(s, a)])
                if q > best_q:
                    best_q = q
            
            V_new[s] = best_q
            delta = max(delta, abs(best_q - V[s]))
            
        V = V_new
        if delta < THETA:
            print(f"[policy.py] Converged in {i+1} iterations.")
            break
    policy = {}
    print("[policy.py] Extracting policy...")
    
    for s in states:
        if api.is_terminal(s):
            continue
            
        best_a = None
        best_q = -float('inf')
        
        for a in api.get_possible_actions(s):
            q = sum(prob * (api.get_reward(s, a, ns) + gamma * V.get(ns, 0.0)) 
                    for ns, prob in trans_cache[(s, a)])
            
            if q > best_q:
                best_q = q
                best_a = a
        
        if best_a:
            policy[s] = best_a

    _save_data_for_visualizer(V, policy, [...], params, api, states, params['rows'], params['cols'])
    
    return policy
def _save_data_for_visualizer(V, policy, convergence_history, params, api, states, rows, cols):
    is_terminal = {s: api.is_terminal(s) for s in states}
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ذخیره همگرایی برای نمودار ۲
        conv_path = os.path.join(base_dir, "convergence_data.json")
        with open(conv_path, "w") as f:
            json.dump(convergence_history, f)

        # ذخیره ارزش‌ها و سیاست برای نمودارهای ۱ و ۳
        grid_info = []
        for s in states:
            r, c, d = s
            grid_info.append({
                "r": r, "c": c, "d": d,
                "cell": api.get_cell_type(s),
                "V": V.get(s, 0.0),
                "action": policy.get(s, None),
                "terminal": is_terminal[s],
            })

        val_path = os.path.join(base_dir, "value_data.json")
        with open(val_path, "w") as f:
            json.dump({"rows": rows, "cols": cols, "max_damage": 4, "params": params, "states": grid_info}, f)
        
        print(f"[policy.py] Data saved to convergence_data.json and value_data.json")
    except Exception as e:
        print(f"[policy.py] Warning: could not save data: {e}")