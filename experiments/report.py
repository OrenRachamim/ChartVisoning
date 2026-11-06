#!/usr/bin/env python3
"""Build experiments/REPORT.md from the JSON results of stages 1-4."""

from __future__ import annotations

import pandas as pd

from common import RESULTS, ROOT, load_json

H_SHOW = (1, 2, 3, 5, 10)

CONCLUSIONS = """## מסקנות (סיכום מנהלים)

**1. חיזוי מחיר (תיוג לפי מחיר הסגירה) – אין יכולת חיזוי סטטיסטית.**
בכל 11 הקונפיגורציות בסריקה, ב-3 הקפלים המתגלגלים על 476 מניות, במודלים לפי סקטור
ובמודלים פר-מניה, הדיוק נשאר 0.33–0.36 (ניחוש = 0.333, ורוב = 0.34–0.35), ה-IC ≈ 0,
וה-spread החתך-רוחבי לא עבר |t| = 2 באף אופק (1, 2, 3, 5 או 10 ימים). ההבדלים בין
גדלי התמונה (32/48/64), אורכי החלון (30/60/120), MA5/10/20, הערוצים ורוחב המודל הם בטווח
הרעש (±0.01). אין "אופטימום" כי אין אות לאופטם.

**2. חיזוי קו ה-MA (הגרף עצמו) – דיוק גבוה, אבל זו האריתמטיקה של הממוצע, לא מידע על השוק.**
MA20 נותן 0.83 ליום אחד, MA10 0.76, MA5 0.69: ככל שהממוצע ארוך יותר, המחר שלו כבר
מוכתב ברובו ממחירים שידועים היום (19 מתוך 20). רגרסיה לוגיסטית על אותם מספרים מגיעה
לאותו דיוק (0.78 מול 0.76), ומומנטום בלבד ל-0.62–0.66. ה-CNN לא מוסיף כמעט דבר מעבר למודל
ליניארי. התוצאה יציבה ב-3 קפלים, אך אינה סחירה: התנועה "החזויה" של ה-MA נובעת ממחירים
שכבר קרו.

**3. אופק.** עבור תיוג MA הדיוק יורד עם האופק (MA10: 0.76 ביום אחד → 0.55 ב-10 ימים),
כלומר "הכי ניתן לחיזוי" הוא 1–2 ימים, מהסיבה הטריוויאלית שלעיל. עבור תיוג מחיר כל האופקים
שווים לניחוש.

**4. פרמטרי תמונה.** ערוץ המחיר בתמונה חשוב לתיוג MA (MA בלבד 0.65 מול 0.70 עם מחיר)
כי המחירים האחרונים קובעים את צעד ה-MA הבא; חלון 30 טוב במעט מ-60/120; גודל תמונה ורוחב
מודל לא משנים. לתיוג מחיר שום פרמטר לא משנה.

**5. קבוצות (סקטורים).** מודל לסקטור לא עוזר: לתיוג MA המודל הכללי טוב יותר (0.770 מול
0.758 ב-5 ימים, יותר נתונים מנצחים); לתיוג מחיר sector_only גבוה במעט ב-10 ימים (0.359)
אך עם spread_t = 0.2, כלומר רעש.

**6. מודל פר-מניה.** כוונון על המניה עצמה לא משפר: per_stock נמוך מהכללי (0.345 מול 0.350
במחיר; 0.825 מול 0.828 ב-MA), ה-ensemble באמצע, והמודל הפרטני מנצח את הכללי רק ב-38%–50%
מהמניות (הטלת מטבע). המלצה: משקל 0–0.3 למודל הפרטני, ובפועל המודל הכללי מספיק.

**7. מה כן יכול לעזור.** המנוף הבא אינו גודל תמונה או ארכיטקטורה אלא מידע: נפח מסחר,
מאפיינים חתך-רוחביים (ביצועים יחסית לסקטור/שוק), נתונים פונדמנטליים, או יעד אחר
(תנודתיות, שאכן ניתנת לחיזוי). מגבלות: 5 שנות נתונים יומיים, מניות גדולות בלבד, CNN קטן על CPU.

"""


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
            b["h"] = b["h"].astype(int)
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
        CONCLUSIONS,
        "## איך לקרוא את הטבלאות\n",
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
