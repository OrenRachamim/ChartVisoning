#!/usr/bin/env python3
"""Build experiments/REPORT.md from the JSON results of stages 1-4."""

from __future__ import annotations

import pandas as pd

from common import RESULTS, ROOT, load_json

H_SHOW = (1, 2, 3, 5, 10)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def sweep_section() -> str:
    rows = []
    for path in sorted(RESULTS.glob("sweep_*.json")):
        r = load_json(path.stem)
        label, variant = r["name"].split("_", 1)
        for h, m in r["summary"].items():
            rows.append({"label": label, "variant": variant, "h": int(h), "acc": m["acc"], "logreg": m["logreg"],
                         "momentum": m["momentum"], "ic": m["ic"], "mom_ic": m["mom_ic"],
                         "spread_pct": m["spread_pct"], "spread_t": m["spread_t"], "min": r["minutes"]})
    if not rows:
        return "_no sweep results_\n"
    df = pd.DataFrame(rows)
    out = []
    for label in ("ma", "close"):
        d = df[df.label == label]
        if d.empty:
            continue
        out.append(f"\n### תיוג לפי {'תנועת ה-MA (הגרף)' if label == 'ma' else 'מחיר הסגירה (סחיר)'}\n")
        piv = d.pivot_table(index="variant", columns="h", values="acc")
        piv["mean"] = piv.mean(axis=1)
        piv = piv.sort_values("mean", ascending=False).round(3).reset_index()
        piv.columns = ["variant"] + [f"acc@{c}d" if isinstance(c, int) else c for c in piv.columns[1:]]
        out.append("**דיוק (3 מחלקות מאוזנות, סיכוי = 0.333)**\n\n" + md_table(piv))
        piv2 = d.pivot_table(index="variant", columns="h", values="spread_t").round(2).reset_index()
        piv2.columns = ["variant"] + [f"t@{c}d" for c in piv2.columns[1:]]
        out.append("\n**t-stat של ה-spread החתך-רוחבי (top 20% − bottom 20%)**\n\n" + md_table(piv2))
        base = d[d.variant == "base"].sort_values("h")
        if len(base):
            b = base[["h", "acc", "logreg", "momentum", "ic", "mom_ic", "spread_pct", "spread_t"]].round(3)
            out.append("\n**base מול הבייסליינים לפי אופק**\n\n" + md_table(b))
    return "\n".join(out) + "\n"


def walkforward_section() -> str:
    rows, folds = [], []
    for path in sorted(RESULTS.glob("walkforward_*.json")):
        r = load_json(path.stem)
        for h, m in r["pooled"].items():
            rows.append({"run": r["name"], "h": int(h), "acc": m["acc"], "momentum": m["momentum"], "ic": m["ic"],
                         "spread_pct": m["spread_pct"], "spread_t": m["spread_t"]})
        for k, f in enumerate(r["folds"]):
            for h, m in f["summary"].items():
                folds.append({"run": r["name"], "fold": k + 1, "cutoff": f["cutoff"], "h": int(h), "acc": m["acc"],
                              "logreg": m["logreg"], "ic": m["ic"], "spread_t": m["spread_t"]})
    if not rows:
        return "_not run_\n"
    out = ["**מאוחד על כל החלונות מחוץ למדגם (3 קפלים)**\n", md_table(pd.DataFrame(rows).round(3)),
           "\n**לפי קפל**\n", md_table(pd.DataFrame(folds).round(3))]
    return "\n".join(out) + "\n"


def groups_section() -> str:
    out = []
    for path in sorted(RESULTS.glob("groups_*.json")):
        r = load_json(path.stem)
        rows = []
        for key, m in r["mean_over_sectors"].items():
            variant, h = key.rsplit("_", 1)
            rows.append({"variant": variant, "h": int(h), **m})
        df = pd.DataFrame(rows).pivot_table(index="variant", columns="h", values="acc").round(3).reset_index()
        df.columns = ["variant"] + [f"acc@{c}d" for c in df.columns[1:]]
        out.append(f"\n**{r['name']}** – דיוק ממוצע על פני הסקטורים\n\n" + md_table(df))
        sec = []
        for s, d in r["sectors"].items():
            sec.append({"sector": s, "stocks": d["n_stocks"], **{f"{v}@5d": d[v]["5"]["acc"] for v in ("general", "sector_only", "general_ft")}})
        out.append("\nלפי סקטור (אופק 5 ימים):\n\n" + md_table(pd.DataFrame(sec).round(3)))
    return "\n".join(out) + "\n" if out else "_not run_\n"


def perstock_section() -> str:
    out = []
    for path in sorted(RESULTS.glob("perstock_*.json")):
        r = load_json(path.stem)
        rows = []
        for v, m in r["pooled"].items():
            for h, mm in m.items():
                rows.append({"variant": v, "h": int(h), "acc": mm["acc"], "ic": mm["ic"], "spread_t": mm["spread_t"]})
        df = pd.DataFrame(rows)
        acc = df.pivot_table(index="variant", columns="h", values="acc").round(3).reset_index()
        acc.columns = ["variant"] + [f"acc@{c}d" for c in acc.columns[1:]]
        tt = df.pivot_table(index="variant", columns="h", values="spread_t").round(2).reset_index()
        tt.columns = ["variant"] + [f"t@{c}d" for c in tt.columns[1:]]
        out.append(f"\n**{r['name']}** ({r['n_stocks']} מניות, fine-tune {r['ft_epochs']} אפוקים, lr {r['ft_lr']})\n")
        out.append("דיוק מאוחד:\n\n" + md_table(acc))
        out.append("\nt-stat של ה-spread:\n\n" + md_table(tt))
        out.append("\nחלק המניות שבהן המודל הפרטני מדויק יותר מהכללי: " +
                   ", ".join(f"{h}d: {v:.0%}" for h, v in r["share_stocks_improved"].items()))
    return "\n".join(out) + "\n" if out else "_not run_\n"


def main() -> None:
    parts = [
        "# דו\"ח ניסויים – חיזוי סטטיסטי מגרפי MA\n",
        "נוצר אוטומטית מ-`experiments/results/*.json` על ידי `experiments/report.py`.\n",
        "קריאה: דיוק נמדד על 3 מחלקות מאוזנות (תרצילים של סט האימון), כך ש-0.333 = ניחוש. "
        "`logreg` = רגרסיה לוגיסטית על אותם מספרים שהתמונה מציירת; `momentum` = הקו ממשיך בכיוונו. "
        "`ic` = קורלציית ספירמן בין ציון המודל לתשואה בפועל. `spread_t` = t-stat של תשואת long/short "
        "(20% עליונים פחות 20% תחתונים בכל תאריך, תאריכים במרווח האופק); |t| > 2 ≈ מובהק.\n",
        "## שלב 1 – סריקת פרמטרים (תת-מדגם של ~120 מניות, 5 אפוקים)\n", sweep_section(),
        "## שלב 2 – אימות מתגלגל על כל היקום\n", walkforward_section(),
        "## שלב 3 – מודלים לפי סקטור\n", groups_section(),
        "## שלב 4 – מודל פר-מניה\n", perstock_section(),
    ]
    (ROOT / "experiments" / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote experiments/REPORT.md")


if __name__ == "__main__":
    main()
