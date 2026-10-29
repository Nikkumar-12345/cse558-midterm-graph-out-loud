"""
CSE558 Data Science - Midterm Project
NYC flight delays (all 336,776 flights out of JFK, LGA and EWR in 2013, BTS
On-Time Performance data as packaged in `nycflights13`) joined to hourly
airport weather (NOAA ASOS, same package).

Run:  python analysis.py            (writes results.json and fig_*.pdf to ../report)
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nycflights13 as nyc

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")
os.makedirs(OUT, exist_ok=True)
R = {}                       # every number quoted in the report lives here
rng = np.random.default_rng(558)

# ---------------------------------------------------------------- plot style
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "xtick.color": INK, "ytick.color": INK,
    "axes.labelcolor": INK, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
})
FIG = (3.4, 1.75)

# ======================================================== 1. DATASET (raw)
f_raw = nyc.flights.copy()
w_raw = nyc.weather.copy()
al = nyc.airlines.copy()
R["raw_rows"], R["raw_cols"] = f_raw.shape
R["w_rows"], R["w_cols"] = w_raw.shape
R["raw_cells"] = int(f_raw.shape[0] * f_raw.shape[1])
R["raw_missing"] = {c: int(v) for c, v in f_raw.isna().sum().items() if v}
R["w_missing"] = {c: int(v) for c, v in w_raw.isna().sum().items() if v}

# ================================================== 2. DATA PREPARATION
f = f_raw.copy()
# (a) duplicates
R["dup_full"] = int(f.duplicated().sum())
R["dup_key"] = int(f.duplicated(["year", "month", "day", "carrier", "flight",
                                  "origin", "sched_dep_time"]).sum())
R["w_dup_key"] = int(w_raw.duplicated(["origin", "time_hour"]).sum())

# (b) missing values carry meaning: no dep_time = cancelled,
#     departed but no arr_delay = diverted / did not complete
f["cancelled"] = f["dep_time"].isna().astype(int)
f["incomplete"] = (f["dep_time"].notna() & f["arr_delay"].isna()).astype(int)
R["n_cancelled"] = int(f.cancelled.sum())
R["n_incomplete"] = int(f.incomplete.sum())
R["tail_missing_all_cancelled"] = bool((f.tailnum.isna() <= f.cancelled.astype(bool)).all())
R["n_tail_missing"] = int(f.tailnum.isna().sum())

# (c) HHMM clock integers -> minutes after midnight (2400 means midnight)
def hhmm_to_min(x):
    x = x.copy()
    return ((x // 100) * 60 + x % 100) % 1440
R["n_2400"] = int((f.dep_time == 2400).sum() + (f.arr_time == 2400).sum())
for c in ["dep_time", "sched_dep_time", "arr_time", "sched_arr_time"]:
    f[c + "_min"] = hhmm_to_min(f[c])

# (d) weather: impossible wind speed, mostly-empty gust column
w = w_raw.copy()
R["w_bad_wind"] = int((w.wind_speed > 100).sum())
R["w_bad_wind_val"] = float(w.wind_speed.max())
w.loc[w.wind_speed > 100, "wind_speed"] = np.nan
R["w_gust_missing_pct"] = round(100 * w.wind_gust.isna().mean(), 1)
w["gusty"] = w.wind_gust.notna().astype(int)
w = w.drop(columns=["wind_gust", "year", "month", "day", "hour"])
R["w_hours_expected"] = 3 * 8760
R["w_hours_present"] = int(len(w))

# (e) join weather on (origin, scheduled hour)
f = f.merge(w, on=["origin", "time_hour"], how="left", indicator=True)
f["weather_missing"] = (f["_merge"] == "left_only").astype(int)
f = f.drop(columns="_merge")
R["n_weather_unmatched"] = int(f.weather_missing.sum())
f["wet"] = (f.precip > 0).astype(float)
f.loc[f.weather_missing == 1, "wet"] = np.nan

# (f) derived features
date = pd.to_datetime(f[["year", "month", "day"]])
f["date"] = date
f["weekday"] = date.dt.dayofweek
f["weekend"] = (f.weekday >= 5).astype(int)
season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
              5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn",
              10: "Autumn", 11: "Autumn"}
f["season"] = f.month.map(season_map)
f["gain"] = f.dep_delay - f.arr_delay          # minutes made up in the air
f["late15"] = (f.dep_delay >= 15).astype(float)
f.loc[f.dep_delay.isna(), "late15"] = np.nan

# (g) outliers: report, do not remove (long delays are the phenomenon)
q1, q3 = f.arr_delay.quantile([.25, .75])
fence = q3 + 1.5 * (q3 - q1)
R["arr_q1"], R["arr_q3"], R["arr_fence"] = float(q1), float(q3), float(fence)
R["n_arr_above_fence"] = int((f.arr_delay > fence).sum())
R["pct_arr_above_fence"] = round(100 * (f.arr_delay > fence).sum() / f.arr_delay.notna().sum(), 1)  # of completed flights
R["w_local_hour_dups"] = int(w_raw.duplicated(["origin", "year", "month", "day", "hour"]).sum())
R["max_dep_delay"] = float(f.dep_delay.max())
R["skew_arr"] = float(stats.skew(f.arr_delay.dropna()))
R["skew_dep"] = float(stats.skew(f.dep_delay.dropna()))
R["kurt_arr"] = float(stats.kurtosis(f.arr_delay.dropna()))

# (h) encoding -> model-ready table
num_cols = ["month", "day", "sched_dep_time_min", "sched_arr_time_min",
            "dep_time_min", "arr_time_min", "dep_delay", "arr_delay",
            "air_time", "distance", "hour", "minute", "gain",
            "temp", "dewp", "humid", "wind_dir", "wind_speed", "precip",
            "pressure", "visib", "gusty", "wet", "cancelled", "incomplete",
            "weather_missing", "weekend", "late15"]
# cyclical encodings for clock and calendar
f["hour_sin"] = np.sin(2 * np.pi * f.sched_dep_time_min / 1440)
f["hour_cos"] = np.cos(2 * np.pi * f.sched_dep_time_min / 1440)
f["doy"] = date.dt.dayofyear
f["doy_sin"] = np.sin(2 * np.pi * f.doy / 365)
f["doy_cos"] = np.cos(2 * np.pi * f.doy / 365)
cyc = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
cat_cols = ["carrier", "origin", "dest", "season", "weekday"]
enc = pd.get_dummies(f[cat_cols].astype(str), prefix=cat_cols, dtype=np.uint8)
X = pd.concat([f[num_cols + cyc].reset_index(drop=True), enc.reset_index(drop=True)], axis=1)
assert X.shape == (336776, 167), X.shape          # size check used in Section 1
R["enc_rows"], R["enc_cols"] = X.shape
R["enc_cells"] = int(X.shape[0] * X.shape[1])
R["enc_onehot"] = {c: int(f[c].nunique()) for c in cat_cols}
R["n_dest"] = int(f.dest.nunique())
R["n_carriers"] = int(f.carrier.nunique())

comp = f[f.arr_delay.notna()].copy()            # completed flights
dep = f[f.dep_delay.notna()].copy()             # departed flights
R["n_completed"] = int(len(comp))
R["n_departed"] = int(len(dep))
R["mean_arr"] = float(comp.arr_delay.mean())
R["median_arr"] = float(comp.arr_delay.median())
R["pct_arr_late15"] = round(100 * (comp.arr_delay >= 15).mean(), 1)
R["pct_early"] = round(100 * (comp.arr_delay < 0).mean(), 1)
R["cancel_rate"] = round(100 * f.cancelled.mean(), 2)


# ============================================================= 4. EDA
# ---- answers to the three basic questions
arr_by_c = comp.groupby("carrier").arr_delay.agg(["mean", "size"]).sort_values("mean", ascending=False)
R["BQ1_worst_all"] = [arr_by_c.index[0], round(arr_by_c["mean"].iloc[0], 1), int(arr_by_c["size"].iloc[0])]
big = arr_by_c[arr_by_c["size"] >= 1000]
R["BQ1_worst_big"] = [big.index[0], round(big["mean"].iloc[0], 1)]
mc = f.groupby("month").cancelled.agg(["sum", "mean"])
R["BQ2_worst_month"] = [int(mc["mean"].idxmax()), int(mc.loc[mc["mean"].idxmax(), "sum"]),
                        round(100 * mc["mean"].max(), 2)]
R["BQ3_r_all"] = float(np.corrcoef(comp.dep_delay, comp.arr_delay)[0, 1])
late_ = comp[comp.dep_delay >= 15]
R["BQ3_r_late"] = float(np.corrcoef(late_.dep_delay, late_.arr_delay)[0, 1])

# Fig 1: arrival delay distribution
fig, ax = plt.subplots(figsize=FIG)
bins = np.arange(-90, 301, 5)
ax.hist(comp.arr_delay.clip(-90, 300), bins=bins, color=BLUE, edgecolor="white", linewidth=0.3)
ax.axvline(comp.arr_delay.median(), color=INK, lw=1, ls="--")
ax.axvline(comp.arr_delay.mean(), color=ORANGE, lw=1.2)
ytop = ax.get_ylim()[1]
ax.text(comp.arr_delay.median() - 4, ytop * 0.92, f"median {comp.arr_delay.median():.0f}", ha="right", color=INK, fontsize=6.5)
ax.text(comp.arr_delay.mean() + 4, ytop * 0.92, f"mean {comp.arr_delay.mean():.1f}", color=INK, fontsize=6.5)
ax.set_xticks(range(-50, 301, 50))
ax.set_xlabel("Arrival delay (min), clipped to [-90, 300]"); ax.set_ylabel("Flights")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k" if v else "0"))
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_arrdelay.pdf")); plt.close(fig)

# Fig 2: time made up in the air by departure-delay bin
cuts = [-30, -5, 0, 15, 30, 60, 120, 240, 1400]
lab = ["<-5", "-5-0", "0-15", "15-30", "30-60", "1-2 h", "2-4 h", ">4 h"]
comp["dep_bin"] = pd.cut(comp.dep_delay, cuts, labels=lab, right=False)
gb = comp.groupby("dep_bin", observed=True).gain.agg(["mean", "median", "size"])
R["gain_by_bin"] = {str(k): [round(v["mean"], 2), float(v["median"]), int(v["size"])] for k, v in gb.iterrows()}
fig, ax = plt.subplots(figsize=FIG)
x = np.arange(len(gb))
ax.bar(x, gb["mean"], color=BLUE, width=0.7, label="mean")
ax.plot(x, gb["median"], "o", color=ORANGE, ms=3, label="median")
ax.axhline(5, color=INK, lw=0.8, ls=":"); ax.text(len(gb) - 0.6, 5.3, "5 min", ha="right", fontsize=6, color=INK)
ax.set_xticks(x); ax.set_xticklabels(gb.index, fontsize=6)
ax.set_xlabel("Departure delay (min)"); ax.set_ylabel("Time made up (min)")
ax.set_ylim(0, 11); ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.12))
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_gain.pdf")); plt.close(fig)

# Fig 3: departure delay by scheduled hour, wet vs dry (lines labelled directly)
d3h = dep[dep.wet.notna() & dep.hour.between(5, 23)]
hm = d3h.groupby(["hour", "wet"]).dep_delay.mean().unstack()
fig, ax = plt.subplots(figsize=FIG)
ax.plot(hm.index, hm[0.0], color=BLUE); ax.plot(hm.index, hm[1.0], color=ORANGE)
ax.text(23.2, hm.loc[23, 1.0] + 2, "precipitation", fontsize=6.5, color=INK, va="bottom")
ax.text(23.2, hm.loc[23, 0.0] - 1, "dry hour", fontsize=6.5, color=INK, va="top")
ax.set_xlim(5, 27); ax.set_xticks(range(5, 24, 3))
ax.set_xlabel("Scheduled departure hour"); ax.set_ylabel("Mean dep. delay (min)")
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_hour_wet.pdf")); plt.close(fig)

# Fig 4: daily cancellation rate across the year
daily = f.groupby("date").agg(cr=("cancelled", "mean"), n=("cancelled", "size"), k=("cancelled", "sum"),
                              season=("season", "first"), weekday=("weekday", "first"))
fig, ax = plt.subplots(figsize=FIG)
ax.plot(daily.index, 100 * daily.cr, color=BLUE, lw=1)
for d, r in daily.sort_values("cr", ascending=False).head(3).iterrows():
    ax.annotate(d.strftime("%d %b"), (d, 100 * r.cr), fontsize=6, color=INK, xytext=(3, -2), textcoords="offset points")
ax.set_ylabel("Flights cancelled (%)")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_cancel_daily.pdf")); plt.close(fig)
R["top_cancel_days"] = [[d.strftime("%Y-%m-%d"), round(100 * r.cr, 1), int(r.k)] for d, r in
                        daily.sort_values("cr", ascending=False).head(5).iterrows()]
R["median_daily_cancel"] = round(100 * daily.cr.median(), 2)
R["max_daily_cancel"] = round(100 * daily.cr.max(), 1)

# ===================================================== 5. HYPOTHESIS TESTS
# Flights on one day share weather and congestion, so the main tests either cluster
# standard errors by day (CR1, t with G-1 d.f.) or use the day as the unit.
def welch(a, b, alternative="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    se = np.sqrt(va + vb); diff = a.mean() - b.mean(); tc = stats.t.ppf(0.975, df)
    return dict(t=float(t), df=float(df), p=float(p), diff=float(diff), se=float(se),
                ci=[float(diff - tc * se), float(diff + tc * se)],
                mean_a=float(a.mean()), mean_b=float(b.mean()), n_a=len(a), n_b=len(b),
                sd_a=float(a.std(ddof=1)), sd_b=float(b.std(ddof=1)))

def clustered(formula, data, term, null=0.0, groups="date"):
    """OLS with standard errors clustered by `groups`; t test of `term` = null with G-1 d.f."""
    d = data.copy(); gid = d[groups].factorize()[0]
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": gid}, use_t=True)
    G = len(np.unique(gid)); df = G - 1
    b, se = float(m.params[term]), float(m.bse[term])
    tc, tc90 = stats.t.ppf(0.975, df), stats.t.ppf(0.95, df)
    t = (b - null) / se
    return dict(est=b, se=se, df=df, G=G, t=float(t), t0=float(b / se),
                p_greater=float(stats.t.sf(t, df)), p_less=float(stats.t.cdf(t, df)),
                p_two=float(2 * stats.t.sf(abs(t), df)),
                ci=[b - tc * se, b + tc * se], ci90=[b - tc90 * se, b + tc90 * se], model=m)

def strip(d):
    return {k: v for k, v in d.items() if k != "model"}

def holm(p):
    return multipletests(p, method="holm")

# --------------------------------------------- Test 1: in-flight recovery
late = comp[comp.dep_delay >= 15].copy()
g = late.gain.values
c1 = clustered("gain ~ 1", late, "Intercept", null=5)
T1 = strip(c1)
T1.update(n=int(len(g)), mean_dep=float(late.dep_delay.mean()), mean_arr=float(late.arr_delay.mean()),
          pct_of_delay=round(100 * g.mean() / late.dep_delay.mean(), 1), median=float(np.median(g)),
          skew=float(stats.skew(g)), cohen_dz=float(g.mean() / g.std(ddof=1)))
# flights treated as independent
se_i = g.std(ddof=1) / np.sqrt(len(g)); tc = stats.t.ppf(0.975, len(g) - 1)
T1["indep_t_vs5"] = float((g.mean() - 5) / se_i); T1["indep_ci"] = [g.mean() - tc * se_i, g.mean() + tc * se_i]
T1["se_ratio"] = float(c1["se"] / se_i)
T1["indep_t_vs0"] = float(stats.ttest_1samp(g, 0).statistic)
# rank-based alternatives against 5 minutes
wz = stats.wilcoxon(g - 5, alternative="greater", method="approx")
T1["wilcoxon_z"], T1["wilcoxon_p"] = float(wz.zstatistic), float(wz.pvalue)
above, below = int((g > 5).sum()), int((g < 5).sum())
T1.update(sign_above5=above, sign_below5=below,
          sign_p=float(stats.binomtest(above, above + below, 0.5, alternative="greater").pvalue))
# day bootstrap for the median gain
days = late.date.values; ud = np.unique(days); idx = {d: np.where(days == d)[0] for d in ud}
bm = []
for _ in range(1000):
    pick = rng.choice(ud, len(ud)); bm.append(np.median(g[np.concatenate([idx[d] for d in pick])]))
T1["median_ci_day_boot"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
# unpaired comparison (wrong design) against 0
T1["unpaired_t"] = float(stats.ttest_ind(late.dep_delay, late.arr_delay, equal_var=False).statistic)
# is it catch-up? other flights, slope on departure delay, air time vs route median
T1["other_mean"] = float(comp[comp.dep_delay < 15].gain.mean())
T1["rho_gain_dep"] = float(stats.spearmanr(late.dep_delay, late.gain)[0])
comp["route_med_air"] = comp.groupby(["origin", "dest"]).air_time.transform("median")
l2 = comp[comp.dep_delay >= 15]; o2 = comp[comp.dep_delay < 15]
T1["air_vs_route_late"] = float((l2.air_time - l2.route_med_air).mean())
T1["air_vs_route_other"] = float((o2.air_time - o2.route_med_air).mean())
# one mean per day
dg = late.groupby("date").gain.agg(["mean", "size"])
dt = stats.ttest_1samp(dg["mean"], 5, alternative="greater")
tcd = stats.t.ppf(0.975, len(dg) - 1); sed = dg["mean"].std(ddof=1) / np.sqrt(len(dg))
T1.update(day_n=int(len(dg)), day_mean=float(dg["mean"].mean()), day_t=float(dt.statistic), day_p=float(dt.pvalue),
          day_ci=[float(dg["mean"].mean() - tcd * sed), float(dg["mean"].mean() + tcd * sed)])
q = dg["size"].rank(method="first"); qn = pd.qcut(q, 4, labels=False)
T1["day_q_fewest"] = float(dg["mean"][qn == 0].mean()); T1["day_q_most"] = float(dg["mean"][qn == 3].mean())  # quartiles of days by number of late flights
T1["rho_nlate_gain"] = float(stats.spearmanr(dg["size"], dg["mean"])[0])
R["T1"] = T1

# --------------------------------------------- Test 2: precipitation
d2 = dep[dep.wet.notna()].copy()
wet_, dry_ = d2[d2.wet == 1].dep_delay, d2[d2.wet == 0].dep_delay
c2 = clustered("dep_delay ~ wet", d2, "wet", null=15)
T2 = strip(c2)
w2 = welch(wet_, dry_)
T2.update(mean_wet=w2["mean_a"], mean_dry=w2["mean_b"], n_wet=w2["n_a"], n_dry=w2["n_b"],
          sd_wet=w2["sd_a"], sd_dry=w2["sd_b"], indep_t_vs15=float((w2["diff"] - 15) / w2["se"]),
          indep_df=w2["df"], indep_ci=w2["ci"])
lev = stats.levene(wet_, dry_, center="median"); T2["levene_W"] = float(lev.statistic)
T2["pooled_t"] = float(stats.ttest_ind(wet_, dry_).statistic)
mw = stats.mannwhitneyu(wet_, dry_, alternative="greater")
T2["prob_sup"] = float(mw.statistic / (len(wet_) * len(dry_))); T2["mw_p"] = float(mw.pvalue)
T2["median_wet"], T2["median_dry"] = float(wet_.median()), float(dry_.median())
T2["late15_wet"] = round(100 * (wet_ >= 15).mean(), 1); T2["late15_dry"] = round(100 * (dry_ >= 15).mean(), 1)
pb = pd.cut(d2.precip, [-0.001, 0, 0.02, 0.1, 5], labels=["0", "0-0.02", "0.02-0.1", ">0.1"])
T2["dose"] = {str(k): round(v, 1) for k, v in d2.groupby(pb, observed=True).dep_delay.mean().items()}
# autocorrelation of hourly mean delay, each airport on a complete hourly grid
oh = d2.groupby(["origin", "time_hour"]).agg(dd=("dep_delay", "mean"), wet=("wet", "first"))
ac = []
for o in ["EWR", "JFK", "LGA"]:
    s = oh.xs(o, level="origin").dd; s.index = pd.to_datetime(s.index)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    ac.append(s.autocorr(1))
T2["acf1_hourly"] = float(np.mean(ac))
# one row per airport-hour
wc = welch(oh[oh.wet == 1].dd, oh[oh.wet == 0].dd)
T2["hour"] = {"diff": wc["diff"], "ci": wc["ci"], "df": wc["df"], "n": int(len(oh)),
              "t_vs15": float((wc["diff"] - 15) / wc["se"]),
              "p_vs15": float(stats.t.sf((wc["diff"] - 15) / wc["se"], wc["df"]))}
# calendar confounding: month, hour and airport fixed effects
c2f = clustered("dep_delay ~ wet + C(month) + C(hour) + C(origin)", d2, "wet", null=15)
T2["fe"] = {k: c2f[k] for k in ["est", "ci", "t", "p_greater", "df"]}
R["T2"] = T2

# --------------------------------------------- Test 3: carriers
cnt = comp.carrier.value_counts(); keep = sorted(cnt[cnt >= 1000].index.tolist())
R["T3_dropped"] = {k: int(v) for k, v in cnt[cnt < 1000].items()}
d3 = comp[comp.carrier.isin(keep)].copy()
groups = [d3[d3.carrier == c].arr_delay.values for c in keep]
H, pk = stats.kruskal(*groups); k_, N_ = len(groups), len(d3)
T3 = {"H": float(H), "df": k_ - 1, "p": float(pk), "N": int(N_), "eps2": float((H - k_ + 1) / (N_ - k_))}
F, pf = stats.f_oneway(*groups); T3["F"], T3["F_df2"] = float(F), int(N_ - k_)
wa = pg.welch_anova(data=d3, dv="arr_delay", between="carrier")
T3["welchF"], T3["welch_df2"], T3["welch_np2"] = float(wa["F"][0]), float(wa["ddof2"][0]), float(wa["np2"][0])
T3["levene_W"] = float(stats.levene(*groups, center="median").statistic)
T3["skew_range"] = [round(min(stats.skew(x) for x in groups), 1), round(max(stats.skew(x) for x in groups), 1)]
summ = d3.groupby("carrier").arr_delay.agg(["mean", "median", "size"]).sort_values("mean")
summ["name"] = summ.index.map(dict(zip(al.carrier, al.name)))
T3["table"] = {c: [round(r["mean"], 2), float(r["median"]), int(r["size"]), r["name"]] for c, r in summ.iterrows()}
best, worst = summ.index[0], summ.index[-1]; T3["best"], T3["worst"] = best, worst
# Dunn's test (rank-based post hoc, tie-corrected) with Holm adjustment
rk = stats.rankdata(d3.arr_delay); d3["rk"] = rk
_, tcount = np.unique(d3.arr_delay, return_counts=True)
tie = (tcount ** 3 - tcount).sum() / (12 * (N_ - 1))
mr = d3.groupby("carrier").rk.mean(); nn = d3.groupby("carrier").size()
pairs, pz = [], []
for i in range(k_):
    for j in range(i + 1, k_):
        a, b = keep[i], keep[j]
        se = np.sqrt((N_ * (N_ + 1) / 12 - tie) * (1 / nn[a] + 1 / nn[b]))
        z = (mr[a] - mr[b]) / se; pairs.append((a, b)); pz.append(2 * stats.norm.sf(abs(z)))
rej = holm(pz)[0]
T3["dunn_sig"] = int(rej.sum()); T3["dunn_nonsig"] = [f"{a}-{b}" for (a, b), r in zip(pairs, rej) if not r]
# Games-Howell simultaneous 95% intervals for mean differences
gh = pg.pairwise_gameshowell(data=d3, dv="arr_delay", between="carrier")
qcrit = stats.studentized_range.ppf(0.95, k_, gh["df"]) / np.sqrt(2)
gh["lo"], gh["hi"] = gh["diff"] - qcrit * gh["se"], gh["diff"] + qcrit * gh["se"]
gh["absd"] = gh["diff"].abs()
gh["sure_gt10"] = (gh.lo > 10) | (gh.hi < -10); gh["sure_lt10"] = (gh.lo > -10) & (gh.hi < 10)
T3["gh_ge10_mean"] = int((gh.absd >= 10).sum())
big10 = gh[gh.absd >= 10]
T3["gh_ge10_pairs"] = [f"{a}-{b}" for a, b in zip(big10.A, big10.B)]
T3["gh_ge10_with_EV_FL"] = int(sum(("EV" in (a, b)) or ("FL" in (a, b)) for a, b in zip(big10.A, big10.B)))
T3["gh_sure_gt10"] = int(gh.sure_gt10.sum()); T3["gh_sure_lt10"] = int(gh.sure_lt10.sum())
T3["gh_undecided"] = int(len(gh) - gh.sure_gt10.sum() - gh.sure_lt10.sum())
r_bw = gh[((gh.A == best) & (gh.B == worst)) | ((gh.A == worst) & (gh.B == best))].iloc[0]
sgn = 1 if r_bw.A == worst else -1
T3["bw_diff"] = float(sgn * r_bw["diff"])
T3["bw_ci_gh"] = sorted([float(sgn * r_bw.lo), float(sgn * r_bw.hi)])
# median gaps of 10+ minutes
med = summ["median"]; mp = [(a, b) for i, a in enumerate(keep) for b in keep[i + 1:] if abs(med[a] - med[b]) >= 10]
T3["median_ge10"] = len(mp); T3["median_ge10_all_FL"] = all("FL" in p for p in mp)
mw_ = [np.median(rng.choice(groups[keep.index(worst)], len(groups[keep.index(worst)]))) -
       np.median(rng.choice(groups[keep.index(best)], len(groups[keep.index(best)]))) for _ in range(1000)]
T3["bw_median_ci"] = [float(np.percentile(mw_, 2.5)), float(np.percentile(mw_, 97.5))]
# Friedman: days as blocks, each carrier's median delay per day
dm = d3.groupby(["date", "carrier"]).arr_delay.median().unstack().dropna()
fr = stats.friedmanchisquare(*[dm[c] for c in dm.columns])
T3["friedman"] = {"chi2": float(fr.statistic), "df": int(dm.shape[1] - 1), "p": float(fr.pvalue),
                  "days": int(len(dm)), "kendall_W": float(fr.statistic / (len(dm) * (dm.shape[1] - 1)))}
# worst minus best with day-clustered errors, and with day fixed effects
sub = d3[d3.carrier.isin([best, worst])].copy(); sub["is_worst"] = (sub.carrier == worst).astype(int)
cb = clustered("arr_delay ~ is_worst", sub, "is_worst"); T3["bw_cluster_ci"] = cb["ci"]
cf = clustered("arr_delay ~ is_worst + C(date)", sub, "is_worst")
T3["bw_dayfe"] = {"est": cf["est"], "ci": cf["ci"]}
T3["within_origin"] = {}
for o in ["EWR", "JFK", "LGA"]:
    so = d3[d3.origin == o]; cs = [c for c in keep if (so.carrier == c).sum() >= 1000]
    Ho, po = stats.kruskal(*[so[so.carrier == c].arr_delay for c in cs])
    T3["within_origin"][o] = [round(float(Ho), 1), len(cs) - 1, float(po)]
# Fig 5: carrier means with day-clustered 95% CIs
rows = []
for c in summ.index:
    cc = clustered("arr_delay ~ 1", d3[d3.carrier == c], "Intercept"); rows.append((c, cc["est"], *cc["ci"]))
fc = pd.DataFrame(rows, columns=["c", "m", "lo", "hi"])
fig, ax = plt.subplots(figsize=FIG)
y = np.arange(len(fc))
ax.axvspan(fc.m.min(), fc.m.min() + 10, color="#e6e5e1", zorder=0, lw=0)
ax.errorbar(fc.m, y, xerr=[fc.m - fc.lo, fc.hi - fc.m], fmt="o", color=BLUE, ms=3.5, lw=1.2, capsize=0)
ax.set_yticks(y); ax.set_yticklabels(fc.c); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean arrival delay (min), day-clustered 95% CI")
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_carriers.pdf")); plt.close(fig)
R["T3"] = T3

# --------------------------------------------- Test 4: seasons
sea = ["Winter", "Spring", "Summer", "Autumn"]
dd = daily.reset_index()
kd = stats.kruskal(*[dd[dd.season == s].cr for s in sea])
T4 = {"H": float(kd.statistic), "df": 3, "p": float(kd.pvalue), "days": int(len(dd))}
T4["levene_W"] = float(stats.levene(*[dd[dd.season == s].cr for s in sea], center="median").statistic)
T4["levene_p"] = float(stats.levene(*[dd[dd.season == s].cr for s in sea], center="median").pvalue)
ct = pd.crosstab(f.season, f.cancelled).reindex(sea)
chi2, p4, dof, exp = stats.chi2_contingency(ct.values, correction=False)
T4.update(chi2=float(chi2), chi2_p=float(p4), min_expected=float(exp.min()),
          G=float(stats.chi2_contingency(ct.values, correction=False, lambda_="log-likelihood")[0]))
T4["rates"] = {s: round(100 * ct.loc[s, 1] / ct.loc[s].sum(), 2) for s in sea}
nc = confint_proportions_2indep(ct.loc["Winter", 1], ct.loc["Winter"].sum(), ct.loc["Autumn", 1],
                                ct.loc["Autumn"].sum(), method="newcomb")
T4["newcombe_ci"] = [100 * nc[0], 100 * nc[1]]
fc4 = f.copy(); fc4["cpct"] = 100 * fc4.cancelled
c4 = clustered("cpct ~ C(season, Treatment('Autumn'))", fc4, "C(season, Treatment('Autumn'))[T.Winter]")
T4["wa_diff"], T4["wa_ci"] = c4["est"], c4["ci"]
T4["width_ratio"] = float((c4["ci"][1] - c4["ci"][0]) / (T4["newcombe_ci"][1] - T4["newcombe_ci"][0]))
wald = c4["model"].f_test([n for n in c4["model"].params.index if n.startswith("C(season")])
T4["wald_F"], T4["wald_df"], T4["wald_p"] = float(np.squeeze(wald.fvalue)), [int(wald.df_num), int(wald.df_denom)], float(wald.pvalue)
wa4 = pg.welch_anova(data=dd.assign(crp=100 * dd.cr), dv="crp", between="season")
T4["welchF"], T4["welch_df"] = float(wa4["F"][0]), [float(wa4["ddof1"][0]), float(wa4["ddof2"][0])]
# pairwise day-clustered comparisons, Holm
pp, pl = [], []
for i in range(4):
    for j in range(i + 1, 4):
        s_ = fc4[fc4.season.isin([sea[i], sea[j]])].copy(); s_["x"] = (s_.season == sea[i]).astype(int)
        pp.append(clustered("cpct ~ x", s_, "x")["p_two"]); pl.append(f"{sea[i]}-{sea[j]}")
T4["pairwise_sig"] = [l for l, r in zip(pl, holm(pp)[0]) if r]
# storm days
T4["feb8_9"] = [int(daily.loc["2013-02-08", "k"]), int(daily.loc["2013-02-09", "k"])]
f2 = f[~f.date.isin(pd.to_datetime(["2013-02-08", "2013-02-09"]))]
T4["drop2_winter"] = round(100 * f2[f2.season == "Winter"].cancelled.mean(), 2)
T4["drop2_summer"] = round(100 * f2[f2.season == "Summer"].cancelled.mean(), 2)
worst10 = daily.sort_values("cr", ascending=False).head(10).index
T4["worst10_share"] = round(100 * daily.loc[worst10, "k"].sum() / f.cancelled.sum(), 1)
d10 = dd[~dd.date.isin(worst10)]
k10 = stats.kruskal(*[d10[d10.season == s].cr for s in sea]); T4["drop10_H"], T4["drop10_p"] = float(k10.statistic), float(k10.pvalue)
f10 = fc4[~fc4.date.isin(worst10)]
T4["drop10_rates"] = {s: round(f10[f10.season == s].cancelled.mean() * 100, 2) for s in sea}
c10 = clustered("cpct ~ C(season, Treatment('Autumn'))", f10, "C(season, Treatment('Autumn'))[T.Winter]")
T4["drop10_wa"], T4["drop10_wa_ci"] = c10["est"], c10["ci"]
T4["day_medians"] = {s: round(100 * dd[dd.season == s].cr.median(), 2) for s in sea}
R["T4"] = T4

# --------------------------------------------- Test 5: weekend vs weekday
dl = comp.groupby("date").agg(ad=("arr_delay", "mean"), we=("weekend", "first"), wd=("weekday", "first"))
we_d, wd_d = dl[dl.we == 1].ad, dl[dl.we == 0].ad
T5 = welch(we_d, wd_d)
T5["levene_W"] = float(stats.levene(we_d, wd_d, center="median").statistic)
T5["levene_p"] = float(stats.levene(we_d, wd_d, center="median").pvalue)
T5["pooled_t"] = float(stats.ttest_ind(we_d, wd_d).statistic)
T5["mw_day_p"] = float(stats.mannwhitneyu(we_d, wd_d).pvalue)
wf, wdf = comp[comp.weekend == 1].arr_delay, comp[comp.weekend == 0].arr_delay
fw = welch(wf, wdf); T5["flight"] = {k: fw[k] for k in ["t", "df", "diff", "ci"]}
mf = stats.mannwhitneyu(wf, wdf); T5["flight_prob_sup"] = float(mf.statistic / (len(wf) * len(wdf)))
T5["flight_medians"] = [float(wf.median()), float(wdf.median())]
c5 = clustered("arr_delay ~ weekend", comp, "weekend"); T5["flight_cluster"] = {"est": c5["est"], "ci": c5["ci"]}
cs = comp.assign(sat=(comp.weekday == 5).astype(int), sun=(comp.weekday == 6).astype(int))
c5b = clustered("arr_delay ~ sat + sun", cs, "sat"); c5c = clustered("arr_delay ~ sat + sun", cs, "sun")
T5["sat"] = {"est": c5b["est"], "ci": c5b["ci"]}; T5["sun"] = {"est": c5c["est"], "ci": c5c["ci"]}
T5["by_weekday_mean"] = {int(k): round(v, 2) for k, v in comp.groupby("weekday").arr_delay.mean().items()}
T5["scheduled_by_weekday"] = {int(k): int(v) for k, v in f.groupby("weekday").size().items()}
T5["acf1_daily"] = float(dl.ad.autocorr(1))
# Fig 6: daily mean arrival delay by weekday
fig, ax = plt.subplots(figsize=FIG)
data = [dl[dl.wd == i].ad.values for i in range(7)]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color=INK, lw=1),
                flierprops=dict(marker="o", ms=2, mfc=GREY, mec=GREY), whiskerprops=dict(color=GREY),
                capprops=dict(color=GREY))
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor(ORANGE if i >= 5 else BLUE); b.set_edgecolor("white")
ax.set_xticks(range(1, 8)); ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_ylabel("Daily mean arr. delay (min)")
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_weekday.pdf")); plt.close(fig)
R["T5"] = T5

with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
