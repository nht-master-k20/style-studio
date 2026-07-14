import pandas as pd

from experiments.analyze_results import summarize, winners


def sample_df():
    rows = []
    for cond, ct, ci in [("fixed5", 0.20, 0.60), ("fixed20", 0.30, 0.70),
                         ("adaptive", 0.29, 0.71)]:
        for pi in range(2):
            for style in ["a", "b"]:
                rows.append({"condition": cond, "prompt_idx": pi, "style": style,
                             "clip_t": ct + pi * 0.01, "clip_i_style": ci,
                             "stop_step": 7 if cond == "adaptive" else None,
                             "elapsed_sec": 100, "peak_vram_gb": 10})
    return pd.DataFrame(rows)


def test_summarize_means_per_condition():
    s = summarize(sample_df())
    assert set(s.index) == {"fixed5", "fixed20", "adaptive"}
    assert s.loc["fixed20", "clip_t"] > s.loc["fixed5", "clip_t"]


def test_winners_counts_fixed_only():
    w = winners(sample_df(), metric="clip_t", fixed_conditions=["fixed5", "fixed20"])
    # fixed20 thang ca 4 case tren clip_t
    assert w["fixed20"] == 4 and w.get("fixed5", 0) == 0
