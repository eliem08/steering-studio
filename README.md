# Steering Studio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/steering-studio/blob/main/colab.ipynb)

An experimental activation-intervention sandbox for GPT-2-style transformer blocks.

## Interactive dashboard

    pip install -r requirements.txt
    streamlit run dashboard.py

Set the model, contrast prompts, target layer, and steering scale to compare baseline and intervened generation.

## Run from the CLI

    python activation_steering.py --model-id gpt2 --layer 6 --scale 0.3 --prompt "I feel"

The initial implementation supports GPT-2-style models exposing model.transformer.h. Other architectures need architecture-specific hook handling.

## Important limitations

This is exploratory code, not a validated control method. A direction derived from two prompts is not guaranteed to represent a single concept; steering can change unrelated capabilities and produce misleading results. Record baselines, random directions, multiple prompts, and task-level regressions before making claims.

## License

MIT.
