# Third-party notices

Parts of this repository originate from or are derived from the research code
and data released with **MLLM-CD: Revealing Multimodal Causality with Large
Language Models**:

- Upstream repository: <https://github.com/JinLi-i/MLLM-CD>
- Paper: <https://arxiv.org/abs/2509.17784>

The following legacy files/data were identified as byte-for-byte copies of the
upstream release at the time this notice was written:

- `utils.py`
- `main_MAG.py`
- `main_Lung.py`
- `gemini_utils.py`
- `causal_graph_utils.py`
- `counterfactual_utils.py`
- `MAG9.csv`
- `Lung.csv`

Additional generated files under `results/` may depend on the same upstream
data and workflow.

This notice provides attribution only; it does **not** grant a license. The
upstream repository did not expose a license file when this project was
reviewed. Redistribution and modification rights must therefore be confirmed
with the upstream rights holders. If written permission or a subsequently
published upstream license exists, record it here and preserve its terms.

InfluenceField-specific additions should be kept distinguishable from this
legacy material so that a future clean release can license original code
without implying rights over third-party files or data.
