# Data Science Midterm Project – Team Graph Out Loud

CSE558 Data Science, Monsoon 2026, IIIT-Delhi

**Delayed in New York: does weather, airline or season decide it, and can a late flight catch up in the air?**

| # | Name | Roll no. | Email | Contribution |
|---|---|---|---|---|
| 1 | Vatsal Garg | 2024611 | vatsal24611@iiitd.ac.in | Dataset and preprocessing (Sections 1–3) |
| 2 | Nikhil Kumar | 2024382 | nikhil24382@iiitd.ac.in | Exploratory data analysis and problem statements (Section 4) |
| 3 | Shakti | 2024519 | shakti24519@iiitd.ac.in | In-flight delay recovery and weather impact analysis (Section 5: P1, P2) |
| 4 | Navanshu Nagar | 2024374 | navanshu24374@iiitd.ac.in | Carrier performance and seasonal cancellation analysis (Section 5: P3, P4) |
| 5 | M. Rayhankhan | 2022269 | rayhan22269@iiitd.ac.in | Weekly delay patterns, synthesis, limitations, code and README (Sections 5, 6) |

Repository: https://github.com/Nikkumar-12345/cse558-midterm-graph-out-loud

## Quick downloads
- **Report (PDF):** [`GraphOutLoud_DS_Midterm_Report.pdf`](GraphOutLoud_DS_Midterm_Report.pdf)
- **Everything in one zip** (code, README, LaTeX source, all figures as separate PDFs, `results.json`, report PDF): [`downloads/GraphOutLoud_DS_Midterm_Full.zip`](downloads/GraphOutLoud_DS_Midterm_Full.zip)
- **All results:** [`report/results.json`](report/results.json); figures in [`report/`](report/)

## Data
- **Flights:** U.S. Bureau of Transportation Statistics (BTS) On-Time Performance records for all
  336,776 domestic flights that departed JFK, LGA and EWR in 2013 (19 columns).
- **Weather:** NOAA hourly observations at the same three airports (26,115 rows × 15 columns).
- **Airlines:** carrier code → name.

All three tables come from the `nycflights13` package (PyPI; also github.com/hadley/nycflights13).
`pip install nycflights13` brings the data, so nothing needs to be downloaded by hand.

## How to run
```bash
cd code
pip install -r requirements.txt
python analysis.py
```
Tested on Python 3.12 in a fresh virtual environment. `requirements.txt` pins `setuptools<81`
because `nycflights13` imports `pkg_resources`, which newer setuptools no longer ships.
It runs in about one minute and writes to `../report/`:
- `results.json`: every number quoted in the report
- `fig_arrdelay.pdf`, `fig_gain.pdf`, `fig_hour_wet.pdf`, `fig_cancel_daily.pdf` (EDA, Figures 1–4)
- `fig_carriers.pdf` (Figure 5), `fig_weekday.pdf` (Figure 6)

The random seed is fixed (558), so the bootstrap results reproduce exactly.

## Method in one paragraph
Flights on the same day share weather and congestion, so they are not independent. The main tests
therefore either cluster standard errors by day (CR1, t distribution with days − 1 d.f.) or use
the day as the unit. P1 and P2 ask whether an effect exceeds a practical threshold, so the
threshold is the boundary of H0. P3–P5 test "no difference" and use the threshold to judge the
confidence interval. Each test is compared with the alternatives listed in the report.

## What the script does
| Step | Report section |
|---|---|
| Raw size and missing values; `cancelled` / `incomplete` flags; HHMM → minutes; weather cleaning; left join on (origin, UTC hour); outlier count; one-hot and sine/cosine encoding to 336,776 × 167 (checked by an assert) | 1–3 |
| Answers to the basic questions; Figures 1–4 | 4 |
| **In-flight delay recovery (P1):** one-sample t on gain vs 5 min with day-clustered SE; independent-flights t, Wilcoxon on g − 5, sign test, day-bootstrap median CI, unpaired Welch; per-day analysis | 5 |
| **Weather impact (P2):** difference in means vs 15 min with day-clustered SE; Welch, pooled t, Levene, Mann–Whitney, airport-hour Welch, month/hour/airport fixed effects, dose–response, hourly autocorrelation | 5 |
| **Carrier performance (P3):** Kruskal–Wallis, Dunn–Holm, Games–Howell simultaneous CIs, ANOVA, Welch ANOVA, Friedman with days as blocks, day-clustered and day-fixed-effect FL − AA, within-airport checks; Figure 5 | 5 |
| **Seasonal cancellation risk (P4):** Kruskal–Wallis on 365 daily rates; chi-square, G-test, Newcombe CI, day-clustered Wald test and pairwise Holm, Welch ANOVA on days, storm-day removal | 5 |
| **Weekly delay patterns (P5):** Welch t on daily means; pooled t, Levene, Mann–Whitney, flight-level and day-clustered versions, Saturday/Sunday contrasts, daily autocorrelation; Figure 6 | 5, 6 |

## Files
- `code/analysis.py`: the full pipeline
- `code/requirements.txt`: package versions used
- `report/`: the LaTeX source, figures, `results.json` and the compiled PDF.
- `downloads/`: the whole project as one zip.
- `report/CSE558_Project_Report.tex`: compile with pdfLaTeX (Overleaf default) twice, with the figure PDFs in the same folder.
