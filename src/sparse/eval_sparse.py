from src.common.entropy import binary_entropy_from_successes


def summarize_eval(step, env_name, algo, seed, returns, lengths):
    successes = [1 if r > 0 else 0 for r in returns]
    mean_return = sum(returns) / len(returns)
    std_return = (sum((x - mean_return) ** 2 for x in returns) / len(returns)) ** 0.5
    return {
        "step": step,
        "env_name": env_name,
        "algo": algo,
        "seed": seed,
        "success_rate": sum(successes) / len(successes),
        "outcome_entropy": binary_entropy_from_successes(successes),
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": sum(lengths) / len(lengths),
    }
