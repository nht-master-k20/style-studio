"""Tong hop bang + figure tu results.csv / lpips_layout.csv / cac JSON adaptive.

  python experiments/analyze_results.py --results_dir experiments/results \
      --out_root experiments/outputs
"""
import argparse
import glob
import json
import os

FIXED_CONDITIONS = ["fixed5", "fixed10", "fixed20", "fixed30"]


def summarize(df):
    return df.groupby("condition")[["clip_t", "clip_i_style", "elapsed_sec",
                                    "peak_vram_gb"]].mean().round(4)


def winners(df, metric, fixed_conditions=FIXED_CONDITIONS):
    """Voi moi (prompt_idx, style): fixed condition nao co metric cao nhat. Tra ve dict dem."""
    sub = df[df["condition"].isin(fixed_conditions)]
    counts = {}
    for _, grp in sub.groupby(["prompt_idx", "style"]):
        best = grp.loc[grp[metric].idxmax(), "condition"]
        counts[best] = counts.get(best, 0) + 1
    return counts


def adaptive_gap(df, metric, fixed_conditions=FIXED_CONDITIONS):
    """Gap trung binh giua adaptive va per-case best fixed (>=0 la match/vuot)."""
    gaps = []
    for (pi, style), grp in df.groupby(["prompt_idx", "style"]):
        fixed_best = grp[grp["condition"].isin(fixed_conditions)][metric].max()
        ada = grp[grp["condition"] == "adaptive"][metric]
        if len(ada):
            gaps.append(float(ada.iloc[0]) - float(fixed_best))
    return sum(gaps) / len(gaps) if gaps else None


def plot_figures(df, out_root, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # histogram buoc dung adaptive
    stops = df[df["condition"] == "adaptive"]["stop_step"].dropna()
    if len(stops):
        plt.figure(figsize=(6, 4))
        plt.hist(stops, bins=range(1, 32))
        plt.xlabel("stop step")
        plt.ylabel("#cases")
        plt.title("Adaptive fusion stop step distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "hist_stop_steps.png"), dpi=150)
        plt.close()

    # duong hoi tu r(t) tu JSON adaptive
    plt.figure(figsize=(7, 4.5))
    for jpath in sorted(glob.glob(os.path.join(out_root, "adaptive", "*.json")))[:12]:
        with open(jpath) as f:
            fusion = json.load(f).get("fusion") or {}
        hist = fusion.get("r_history") or []
        if hist:
            steps, rs = zip(*hist)
            plt.plot(steps, rs, alpha=0.6)
    plt.axhline(0.2, color="red", linestyle="--", label="rho")
    plt.xlabel("denoise step")
    plt.ylabel("r(t)")
    plt.title("Teacher-student attention convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "r_curves.png"), dpi=150)
    plt.close()


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--out_root", default="experiments/outputs")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.results_dir, "results.csv"))
    summary = summarize(df)
    lp = os.path.join(args.results_dir, "lpips_layout.csv")
    if os.path.exists(lp):
        ldf = pd.read_csv(lp)
        summary = summary.join(ldf.groupby("condition")["lpips_mean"].mean().round(4))
    summary.to_csv(os.path.join(args.results_dir, "summary_table.csv"))
    print(summary)

    win_rows = []
    for metric in ["clip_t", "clip_i_style"]:
        counts = winners(df, metric)
        gap = adaptive_gap(df, metric)
        print(f"\n[{metric}] per-case winners among fixed: {counts}")
        if gap is None:
            print(f"[{metric}] adaptive mean gap to per-case best fixed: N/A (no adaptive rows yet)")
        else:
            print(f"[{metric}] adaptive mean gap to per-case best fixed: {gap:+.4f}")
        for cond, n in counts.items():
            win_rows.append({"metric": metric, "condition": cond, "wins": n})
        # Skip the adaptive_gap row entirely when there are no adaptive rows yet
        # (gap is None), rather than writing an empty cell into winners.csv --
        # a missing row is a clearer/safer signal to downstream readers than a
        # NaN "wins" value that could be mistaken for a real (zero) gap.
        if gap is not None:
            win_rows.append({"metric": metric, "condition": "adaptive_gap", "wins": gap})
    pd.DataFrame(win_rows).to_csv(os.path.join(args.results_dir, "winners.csv"), index=False)

    plot_figures(df, args.out_root, args.results_dir)
    print(f"\nfigures + tables saved to {args.results_dir}")


if __name__ == "__main__":
    main()
