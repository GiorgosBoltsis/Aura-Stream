import numpy as np

from .baselines import _context_edge, _highest_below, SAFETY, bba_context_edge, context_joint, mpc_plan

RESIDUAL_STEPS = np.array([-2, -1, 0, 1, 2])
EXPERT_STEPS = np.array([-1, 0, 1])
EXPERTS = ("MPC-Joint", "Ctx-Joint", "BBA+CtxEdge")


def residual_planner(env):
    first, values = mpc_plan(env)
    proposals = [env.encode(first[k], k) for k in range(env.n_edges)]
    features = [*(first / (env.n_bitrates - 1)), *np.clip(values / 10.0, -5.0, 5.0)]
    return np.asarray(features), proposals


def context_planner(env):
    est = env.radio_estimate()
    rate = SAFETY * np.minimum(est, env.edge_available())
    if env.handover_risk() > 0.5:
        rate = rate * 0.8
    if env.buffer < 8.0:
        rate = rate * 0.7
    first = np.array([_highest_below(r) for r in rate])
    proposals = [env.encode(first[k], k) for k in range(env.n_edges)]
    preferred = np.zeros(env.n_edges)
    preferred[_context_edge(env)] = 1.0
    features = [*(first / (env.n_bitrates - 1)), *preferred]
    return np.asarray(features), proposals


def expert_planner(env):
    first, values = mpc_plan(env)
    best = int(np.argmax(values))
    proposals = [env.encode(first[best], best), context_joint(env), bba_context_edge(env)]
    features = []
    for a in proposals:
        idx, edge = env.decode(a)
        features += [idx / (env.n_bitrates - 1), edge / (env.n_edges - 1)]
    features += list(np.clip(values / 10.0, -5.0, 5.0))
    return np.asarray(features), proposals


MODES = {
    "residual": (residual_planner, 6, [len(RESIDUAL_STEPS), 3]),
    "experts": (expert_planner, 9, [len(EXPERTS), len(EXPERT_STEPS)]),
    "residual_ctx": (context_planner, 6, [len(RESIDUAL_STEPS), 3]),
}


def to_env_action(env, action, mode):
    if mode in ("residual", "residual_ctx"):
        delta, edge = int(RESIDUAL_STEPS[int(action[0])]), int(action[1])
        return env.shifted(env.plan[edge], delta)
    if mode == "experts":
        return env.shifted(env.plan[int(action[0])], int(EXPERT_STEPS[int(action[1])]))
    if mode == "factored":
        return env.encode(int(action[0]), int(action[1]))
    return int(action)
