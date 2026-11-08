#!/usr/bin/env python3
"""Build sentiment/REPORT.md from sentiment/results/*.json."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "sentiment" / "results"
STUDIES = [("fg_reconstructed_spx", "Fear & Greed משוחזר מול S&P 500 (2008–2026)"),
           ("vix_spx", "VIX (כציון 0–100, 100 = שאננות) מול S&P 500 (1990–2026)"),
           ("crypto_fg_btc", "Crypto Fear & Greed מול ביטקוין (2018–2026)"),
           ("cnn_fg_real_1y_spx", "Fear & Greed האמיתי של CNN, 250 הימים הזמינים (תיאורי בלבד)")]

CONCLUSIONS = """## מסקנות

**1. הרמה של המדד נושאת אות קטן, מנוגד (contrarian), בעיקר דרך רכיב התנודתיות.**
ב-VIX (36 שנים): כשה-VIX בחמישית התחתונה שלו ("שאננות"), תשואת השבוע הבא של S&P 500 היא
+0.02% לעומת +0.21% בממוצע (t = −3.2), ותשואת 21 יום +0.26% לעומת +0.76% (t = −2.5). פחד פחות
חמדנות ב-21 יום: +1.0%, p = 0.01 מול block-bootstrap. במדד המשוחזר (18 שנים) הסימן זהה אבל חלש
יותר: IC של הרמה −0.06 ל-5 ימים (t = −2.2) ו-−0.15 ל-63 ימים במודל walk-forward (t = −2.3);
אף דלי בודד אינו מובהק. בקריפטו הסימן הפוך (חמדנות ⇒ המשך עלייה) ולא מובהק.

**2. האות לא ניתן להמרה ליתרון מסחרי פשוט.** אף אחד מארבעת כללי long/flat לא עבר את
buy & hold ב-Sharpe באף מדד (למשל VIX: "long אלא אם חמדנות קיצונית" Sharpe 0.51 מול 0.48, p = 0.17
מול תזמון אקראי). הסיבה: השוק עולה גם בתקופות חמדנות, רק פחות; לצאת ממנו עולה יותר ממה שהאות
מחזיר. פחד קיצוני מקדים תשואות טובות יותר (VIX: +3.8% לרבעון מול +2.4%) אבל עם פיזור עצום.

**3. צורת הגרף של המדד לא מוסיפה דבר מעבר לרמה.** 13 מאפייני צורה (שינוי 5/10/20 יום, שיפוע,
rebound מפחד, rollover מחמדנות, ימים מאז פחד…) אינם מובהקים בעקביות; רגרסיה על כולם גרועה
מרגרסיה על הרמה בלבד; CNN על תמונת גרף המדד (60 יום) נופל מתחת ל-base rate בכל אופק ובכל מדד,
ואינו שונה מ-CNN על תוויות מעורבבות. זה אותו ממצא כמו בגרפי המניות: תמונה אינה קלט טוב יותר
מהמספר עצמו.

**4. אירועי "כניסה לפחד קיצוני"** (111 אירועים ב-VIX, 50 במדד המשוחזר) מקדימים תשואת רבעון חיובית
ב-66% מהמקרים עם ממוצע +2%, אבל גם כניסה לחמדנות קיצונית מקדימה +1.2% ב-70% מהמקרים: ההבדל הוא
בגודל, לא בכיוון, ורווחי הסמך חופפים.

**5. מגבלות.** המדד של CNN עצמו זמין רק לשנה אחת; השחזור מתואם 0.84 איתו ומכסה 18 שנים, אבל חסר
רכיב Put/Call ומשתמש ביקום מניות עדכני (הטיית שורדות ברוחב). סדרה אחת מול יעד אחד = מעט תצפיות
בלתי-תלויות (18 שנים ≈ 72 רבעונים), ולכן t של 2 כאן חלש מ-t של 2 במחקר החתך-רוחבי. תקופת
2008–2026 היא בעיקר שוק עולה.

**שורה תחתונה**: Fear & Greed ודומיו הם מדדי משטר תנודתיות. הם אומרים משהו על התשואה הצפויה
(פחות כשהכול רגוע, יותר אחרי פאניקה), בעיקר ל-1 עד 4 שבועות, אבל לא על הכיוון ביום-יומיים,
לא דרך "צורת הגרף", ולא ברמה שמנצחת החזקה פשוטה אחרי עלויות.

"""


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def section(name: str, title: str) -> str:
    p = RES / f"{name}.json"
    if not p.exists():
        return f"## {title}\n\n_not run_\n"
    r = json.load(open(p))
    s = [f"## {title}\n", f"{r['n_days']} ימי מסחר, {r['start']} עד {r['end']}. תמונות: `figures/{name}_overview.png`, `figures/{name}_buckets_events.png`, `figures/{name}_strategy.png`.\n"]
    b = pd.DataFrame(r["buckets"])
    t = b[["bucket", "n_days", "ret_5_pct", "t_5", "ret_21_pct", "t_21", "ret_63_pct", "t_63"]].copy()
    t.columns = ["רמה", "ימים", "5d %", "t", "21d %", "t", "63d %", "t"]
    s.append("**א. תשואת השוק קדימה לפי רמת המדד** (t = Newey-West של העודף מעל הממוצע הבלתי-מותנה; |t|>2 ≈ מובהק). "
             f"ממוצע בלתי-מותנה: 5d {b.base_5_pct.iloc[0]:+.2f}%, 21d {b.base_21_pct.iloc[0]:+.2f}%, 63d {b.base_63_pct.iloc[0]:+.2f}%.\n\n" + md(t.round(2)))
    ic = pd.DataFrame(r["ic"])
    t = ic[["feature", "ic_5", "t_5", "ic_21", "t_21", "ic_63", "t_63"]].round(3)
    s.append("\n**ב. מאפייני צורת הגרף של המדד: קורלציית ספירמן עם התשואה קדימה (IC) ו-t מתוקן לחפיפה**\n\n" + md(t))
    ev = []
    for k, e in r["events"].items():
        if e["n"]:
            ev.append({"אירוע": k, "n": e["n"], "21d %": e["mean_21d_pct"], "hit 21d": e["hit_21d"], "63d %": e["mean_63d_pct"], "hit 63d": e["hit_63d"],
                       "CI 63d": f"[{e['ci_lo_pct'][-1]:+.1f}, {e['ci_hi_pct'][-1]:+.1f}]"})
    if ev:
        s.append("\n**ג. אירועי כניסה לאזור** (היום הראשון באזור אחרי 21 יום מחוצה לו; CI = bootstrap 95%)\n\n" + md(pd.DataFrame(ev).round(2)))
    n = r["null_fear_minus_greed_21d"]
    s.append(f"\n**ד. null של block-bootstrap**: תשואת 21 יום בפחד (<25) פחות בחמדנות (≥75): נצפה {n['observed_pct']:+.2f}%, "
             f"null {n['null_mean_pct']:+.2f}±{n['null_sd_pct']:.2f}%, p = {n['p_value']:.2f}.\n")
    st = []
    for k, v in r["strategies"].items():
        st.append({"כלל": k, "חשיפה": v["exposure"], "CAGR %": v["strategy"]["cagr_pct"], "Sharpe": v["strategy"]["sharpe"], "DD %": v["strategy"]["max_dd_pct"],
                   "B&H CAGR %": v["buy_hold"]["cagr_pct"], "B&H Sharpe": v["buy_hold"]["sharpe"], "p מול תזמון אקראי": v["random_timing_sharpe_p"]})
    s.append("\n**ה. אסטרטגיות long/flat** (p = חלק מהתזמונים האקראיים עם אותה חשיפה שהשיגו Sharpe גבוה יותר)\n\n" + md(pd.DataFrame(st).round(2)))
    if "models" in r:
        rows = []
        for k, m in r["models"].items():
            for h in (5, 21, 63):
                x = m[str(h)] if str(h) in m else m[h]
                rows.append({"מודל": k, "אופק": h, "דיוק": x["accuracy"], "base rate": x["base_rate"], "IC": x["ic"], "t": x["ic_t"], "L/S % לתקופה": x["ls_mean_pct"], "t L/S": x["ls_t"]})
        s.append("\n**ו. מודלים walk-forward (אימון מחדש כל שנה על כל העבר)**. base rate = חלק הימים העולים; מודל חסר ערך מגיע לכל היותר ל-base rate.\n\n" + md(pd.DataFrame(rows).round(3)))
    return "\n".join(s) + "\n"


def components() -> str:
    p = RES / "components.json"
    if not p.exists():
        return ""
    c = json.load(open(p))
    rows = [{"רכיב": k, **{kk: v["ic"][kk] for kk in ("ic_5", "t_5", "ic_21", "t_21", "ic_63", "t_63")}} for k, v in c.items()]
    return "## רכיבי המדד המשוחזר, כל אחד לבד (IC מול S&P 500)\n\n" + md(pd.DataFrame(rows).round(3)) + "\n"


HEAD = """# מחקר: האם מדד Fear & Greed (והדומים לו) חוזה את כיוון השוק?

קוד: `sentiment/`. נתונים: `data/sentiment/`. תמונות: `sentiment/figures/`.

## נתונים
* **CNN Fear & Greed**: הנקודה הציבורית מחזירה רק ~250 ימים אחרונים (היסטוריה ארוכה חסומה). משמש לאימות השחזור בלבד.
* **שחזור המדד** (2008–2026): 6 מתוך 7 הרכיבים שCNN מפרסם, כל אחד כאחוזון ב-500 הימים הקודמים, ממוצע: מומנטום (S&P מול ממוצע 125), עוצמה (שיאים פחות שפלים של 52 שבועות ב-476 מניות), רוחב (McClellan volume summation), VIX מול ממוצע 50, מניות מול אג"ח (20 יום), junk מול IG (HYG מול LQD, 20 יום). יחס Put/Call אינו זמין חינם. **קורלציה עם המדד האמיתי ב-249 ימי החפיפה: 0.84 (0.82 לשינויי 5 ימים).**
* **VIX** (1990–2026) כאחוזון מתגלגל הפוך: 100 = שאננות.
* **Crypto Fear & Greed** של alternative.me (2018–2026), המדד האמיתי, מול ביטקוין.

## שיטה
תשואות קדימה (לוג) ל-1/5/10/21/63 ימים. בגלל חפיפה בין חלונות, כל t-stat הוא Newey-West עם lag = האופק, ונוסף null של block-bootstrap (בלוקים של 21 יום) ו-null של תזמון אקראי. מודלים walk-forward עם אימון מחדש שנתי: רגרסיה לוגיסטית על הרמה בלבד, על מאפייני צורה, CNN על תמונת גרף המדד (60 יום), CNN על המדד + המחיר, ו-CNN על תוויות מעורבבות (null).
חשוב: סדרה אחת מול יעד אחד ⇒ מספר התצפיות הבלתי-תלויות קטן (למשל 18 שנה / 63 יום ≈ 72 תצפיות לרבעון).

"""


def main() -> None:
    parts = [HEAD, CONCLUSIONS] + [section(n, t) for n, t in STUDIES] + [components()]
    (ROOT / "sentiment" / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote sentiment/REPORT.md")


if __name__ == "__main__":
    main()
