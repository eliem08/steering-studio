# Steering Studio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/steering-studio/blob/main/colab.ipynb)

An experimental activation-intervention sandbox for GPT-2-style transformer blocks.

## Run a controlled steering comparison

    pip install -r requirements.txt
    python activation_steering.py --model-id gpt2 --layer 6 --scale 0.5 --prompt "I feel"

The script computes a difference between mean activations for a positive and contrast prompt, normalizes that direction, then compares greedy baseline generation with an intervention at one block. Results are saved to outputs/steering/steering.json.

    python activation_steering.py --model-id gpt2 --layer 4 --scale 0.1 --device cuda

This initial implementation supports GPT-2-style models exposing model.transformer.h. Other architectures need architecture-specific hook handling. In Colab, select GPU runtime if available.

## Important limitations

This is exploratory code, not a validated control method. A direction derived from two prompts is not guaranteed to represent a single concept; steering can change unrelated capabilities and produce misleading results. Record baselines, random directions, multiple prompts, and task-level regressions before making claims.

## Quickstart

    python main.py

## License

MIT.
