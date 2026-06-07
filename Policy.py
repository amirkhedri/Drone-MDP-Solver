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
            
            # محاسبه ارزش جدید با استفاده از عمل‌های ممکن
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