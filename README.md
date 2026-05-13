# Adversarial Robustness Engine

LightGBM ranker with adversarial training (PGD). Tests robustness of ETF signals under worst‑case feature perturbations (ε=0.5σ). Outputs adversarially‑robust predicted returns and ranking.

- **Attack:** PGD (5 steps, α=0.1)
- **Defense:** Retrained on adversarial examples
- **Ranking metric:** predicted return under perturbation
- **Output:** top 3 ETFs per universe, full ranking table

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
