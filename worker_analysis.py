

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from lifelines import KaplanMeierFitter

# ── Style ──────────────────────────────────────────────────────────────────
ACCENT  = "#E94560"   # red    — converters / workers
LIGHT   = "#0F3460"   # navy   — non-converters / admins
GOLD    = "#F5A623"   # amber  — highlights
GREEN   = "#2ECC71"   # green  — worker positive
GREY    = "#AAAAAA"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.edgecolor":    "#CCCCCC",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "DejaVu Sans",
    "grid.color":        "#E0E0E0",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
})

OUT = "C:/Users/HP/Documents/2026 Data Projects/trial conversion/charts"


print("=" * 60)
print("1. LOADING DATA & CLASSIFYING ACTIVITIES")
print("=" * 60)

df = pd.read_csv("C:/Users/HP/Documents/2026 Data Projects/trial conversion/DA_task.csv")
df.columns = df.columns.str.lower()
for col in ["timestamp", "converted_at", "trial_start", "trial_end"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")
df = df.drop_duplicates()

# ── Activity classification ──────────────────────────────────────────────────
# WORKER actions: things only an employee would do
WORKER_ACTIVITIES = {
    "PunchClock.PunchedIn",           # worker clocks in
    "PunchClock.PunchedOut",          # worker clocks out
    "PunchClockStartNote.Add.Completed",  # worker adds note on clock-in
    "PunchClockEndNote.Add.Completed",    # worker adds note on clock-out
    "Break.Activate.Started",         # worker starts a break
    "Break.Activate.Finished",        # worker ends a break
    "Scheduling.Availability.Set",    # worker sets their own availability
    "Scheduling.ShiftSwap.Created",   # worker requests a swap
    "Scheduling.ShiftHandover.Created",  # worker requests a handover
    "Scheduling.OpenShiftRequest.Created",  # worker requests an open shift
    "Mobile.Schedule.Loaded",         # worker (or admin) views schedule on mobile
    "Shift.View.Opened",              # worker views shift details
    "ShiftDetails.View.Opened",       # worker views shift details
    "Absence.Request.Created",        # worker creates absence request
}

# ADMIN actions: things only a manager/admin would do
ADMIN_ACTIVITIES = {
    "Scheduling.Shift.Created",       # admin creates shifts
    "Scheduling.Shift.AssignmentChanged",  # admin changes assignments
    "Scheduling.Template.ApplyModal.Applied",  # admin applies templates
    "Scheduling.Shift.Approved",      # admin approves shifts for payroll
    "Scheduling.ShiftSwap.Accepted",  # admin approves swap
    "Scheduling.ShiftHandover.Accepted",  # admin approves handover
    "Scheduling.OpenShiftRequest.Approved",  # admin approves open shift request
    "Absence.Request.Approved",       # admin approves absence
    "Absence.Request.Rejected",       # admin rejects absence
    "Timesheets.BulkApprove.Confirmed",  # admin bulk approves timesheets
    "Integration.Xero.PayrollExport.Synced",  # admin exports payroll
    "Revenue.Budgets.Created",        # admin creates budgets
    "Communication.Message.Created",  # admin sends team message
    "PunchClock.Entry.Edited",        # admin edits time entries
}

df["activity_type"] = df["activity_name"].apply(
    lambda x: "worker" if x in WORKER_ACTIVITIES
              else ("admin" if x in ADMIN_ACTIVITIES else "shared")
)

print("\nActivity classification summary:")
print(df.groupby("activity_type")["activity_name"].count().rename("event_count"))
print("\nWorker activity breakdown:")
print(df[df["activity_type"] == "worker"]["activity_name"].value_counts())
print("\nAdmin activity breakdown:")
print(df[df["activity_type"] == "admin"]["activity_name"].value_counts())


# 2. ORG-LEVEL FEATURE ENGINEERING — ADMIN vs WORKER SPLIT

print("\n" + "=" * 60)
print("2. BUILDING ADMIN vs WORKER FEATURE MATRIX")
print("=" * 60)

org_meta = df.drop_duplicates("organization_id")[
    ["organization_id", "converted", "converted_at", "trial_start", "trial_end"]
].copy()

# Total events
total_events     = df.groupby("organization_id").size().rename("total_events")
admin_events     = df[df["activity_type"] == "admin"].groupby("organization_id").size().rename("admin_events")
worker_events    = df[df["activity_type"] == "worker"].groupby("organization_id").size().rename("worker_events")

# Unique activity types
admin_unique     = df[df["activity_type"] == "admin"].groupby("organization_id")["activity_name"].nunique().rename("admin_unique_activities")
worker_unique    = df[df["activity_type"] == "worker"].groupby("organization_id")["activity_name"].nunique().rename("worker_unique_activities")

# Active days
admin_days       = df[df["activity_type"] == "admin"].groupby("organization_id")["timestamp"].apply(
    lambda x: x.dt.date.nunique()).rename("admin_active_days")
worker_days      = df[df["activity_type"] == "worker"].groupby("organization_id")["timestamp"].apply(
    lambda x: x.dt.date.nunique()).rename("worker_active_days")

# Specific worker activity flags
punchclock_orgs  = df[df["activity_name"].isin(["PunchClock.PunchedIn", "PunchClock.PunchedOut"])]\
                     .groupby("organization_id").size().rename("punchclock_events")
availability_orgs = df[df["activity_name"] == "Scheduling.Availability.Set"]\
                      .groupby("organization_id").size().rename("availability_events")
swap_orgs        = df[df["activity_name"].isin(["Scheduling.ShiftSwap.Created", "Scheduling.ShiftHandover.Created"])]\
                     .groupby("organization_id").size().rename("shift_swap_events")
absence_orgs     = df[df["activity_name"] == "Absence.Request.Created"]\
                     .groupby("organization_id").size().rename("absence_request_events")
mobile_orgs      = df[df["activity_name"] == "Mobile.Schedule.Loaded"]\
                     .groupby("organization_id").size().rename("mobile_schedule_views")

# First worker activity timing
first_worker_ts  = df[df["activity_type"] == "worker"].groupby("organization_id")["timestamp"].min().rename("first_worker_ts")
first_admin_ts   = df[df["activity_type"] == "admin"].groupby("organization_id")["timestamp"].min().rename("first_admin_ts")

# Merge everything
org = org_meta\
    .merge(total_events,      on="organization_id", how="left")\
    .merge(admin_events,      on="organization_id", how="left")\
    .merge(worker_events,     on="organization_id", how="left")\
    .merge(admin_unique,      on="organization_id", how="left")\
    .merge(worker_unique,     on="organization_id", how="left")\
    .merge(admin_days,        on="organization_id", how="left")\
    .merge(worker_days,       on="organization_id", how="left")\
    .merge(punchclock_orgs,   on="organization_id", how="left")\
    .merge(availability_orgs, on="organization_id", how="left")\
    .merge(swap_orgs,         on="organization_id", how="left")\
    .merge(absence_orgs,      on="organization_id", how="left")\
    .merge(mobile_orgs,       on="organization_id", how="left")\
    .merge(first_worker_ts,   on="organization_id", how="left")\
    .merge(first_admin_ts,    on="organization_id", how="left")

# Fill nulls with 0
fill_cols = ["admin_events", "worker_events", "admin_unique_activities",
             "worker_unique_activities", "admin_active_days", "worker_active_days",
             "punchclock_events", "availability_events", "shift_swap_events",
             "absence_request_events", "mobile_schedule_views"]
org[fill_cols] = org[fill_cols].fillna(0)

# Derived fields
org["has_any_worker_activity"]   = (org["worker_events"] > 0).astype(int)
org["has_punchclock"]            = (org["punchclock_events"] > 0).astype(int)
org["has_availability"]          = (org["availability_events"] > 0).astype(int)
org["has_shift_swap"]            = (org["shift_swap_events"] > 0).astype(int)
org["has_absence_request"]       = (org["absence_request_events"] > 0).astype(int)
org["has_mobile_view"]           = (org["mobile_schedule_views"] > 0).astype(int)

org["worker_event_share"]        = org["worker_events"] / org["total_events"].replace(0, 1)

org["days_to_conversion"]        = np.where(
    org["converted"],
    (org["converted_at"] - org["trial_start"]).dt.total_seconds() / 86400,
    np.nan
)

org["hours_to_first_worker"]     = np.where(
    org["first_worker_ts"].notna(),
    (org["first_worker_ts"] - org["trial_start"]).dt.total_seconds() / 3600,
    np.nan
)

org["hours_to_first_admin"]      = (
    (org["first_admin_ts"] - org["trial_start"]).dt.total_seconds() / 3600
).clip(lower=0)

total_orgs   = len(org)
conv_rate    = org["converted"].mean()
print(f"\nTotal orgs: {total_orgs}")
print(f"Overall conversion rate: {conv_rate:.1%}")
print(f"\nOrgs with ANY worker activity: {org['has_any_worker_activity'].sum()} ({org['has_any_worker_activity'].mean():.1%})")
print(f"Orgs with ZERO worker activity: {(org['has_any_worker_activity']==0).sum()} ({(org['has_any_worker_activity']==0).mean():.1%})")


# 2. CHART: CONVERSION OVERVIEW  ->  01_conversion_overview.png

print("\n" + "=" * 60)
print("2b. CONVERSION OVERVIEW CHART")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Overall Trial Conversion Overview", fontsize=15, fontweight="bold")

# Panel 1 - Donut: converted vs not
ax = axes[0]
n_conv   = int(org["converted"].sum())
n_noconv = total_orgs - n_conv
donut_labels = [
    f"Converted\n{conv_rate:.1%}  ({n_conv})",
    f"Not Converted\n{1-conv_rate:.1%}  ({n_noconv})",
]
ax.pie(
    [n_conv, n_noconv],
    labels=donut_labels,
    colors=[ACCENT, LIGHT],
    startangle=90,
    wedgeprops=dict(width=0.55),
)
ax.set_title("Overall Conversion Rate")

# Panel 2 - Event volume distribution by outcome
ax = axes[1]
cap95 = org["total_events"].quantile(0.95)
for label, grp, color in [
    ("Converted",     True,  ACCENT),
    ("Not Converted", False, LIGHT),
]:
    data = org.loc[org["converted"] == grp, "total_events"].clip(upper=cap95)
    ax.hist(data, bins=35, alpha=0.7, label=label, color=color, edgecolor="white")
ax.set_xlabel("Total Events During Trial (capped 95th pct)")
ax.set_ylabel("Organisations")
ax.set_title("Event Volume Distribution")
ax.legend()

# Panel 3 - Active days distribution by outcome
ax = axes[2]
for label, grp, color in [
    ("Converted",     True,  ACCENT),
    ("Not Converted", False, LIGHT),
]:
    data = org.loc[org["converted"] == grp, "admin_active_days"]
    ax.hist(data, bins=20, alpha=0.7, label=label, color=color, edgecolor="white")
ax.set_xlabel("Active Days During Trial")
ax.set_ylabel("Organisations")
ax.set_title("Active Days Distribution")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/01_conversion_overview.png", dpi=150, bbox_inches="tight")
plt.close()


# CHART: KEY METRIC DISTRIBUTIONS  ->  02_feature_distributions.png

print("FEATURE DISTRIBUTIONS CHART")

from scipy.stats import mannwhitneyu as _mwu

metric_pairs = [
    ("total_events",      "Total Events"),
    ("admin_events",      "Admin Events"),
    ("worker_events",     "Worker Events"),
    ("admin_active_days", "Active Days"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Key Metrics: Converters vs Non-Converters", fontsize=15, fontweight="bold")

for ax, (col, label) in zip(axes.flatten(), metric_pairs):
    c_data  = org.loc[org["converted"],  col].dropna()
    nc_data = org.loc[~org["converted"], col].dropna()
    ax.boxplot(
        [c_data, nc_data],
        labels=["Converted", "Not Converted"],
        patch_artist=True,
        boxprops=dict(facecolor=ACCENT, alpha=0.7),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=LIGHT),
        capprops=dict(color=LIGHT),
        flierprops=dict(marker="o", color=GREY, alpha=0.3, markersize=3),
    )
    ax.set_title(label)
    ax.set_ylabel(label)
    _, pval = _mwu(c_data, nc_data, alternative="two-sided")
    sig_lbl = "p < 0.05 *" if pval < 0.05 else f"p = {pval:.3f}"
    ax.set_xlabel(f"Mann-Whitney  {sig_lbl}")

plt.tight_layout()
plt.savefig(f"{OUT}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 02_feature_distributions.png saved")

# 3. THE HEADLINE FINDING — WORKER ADOPTION vs CONVERSION

print("\n" + "=" * 60)
print("3. HEADLINE FINDING — WORKER ADOPTION vs CONVERSION")
print("=" * 60)

worker_conv    = org.loc[org["has_any_worker_activity"] == 1, "converted"].mean()
no_worker_conv = org.loc[org["has_any_worker_activity"] == 0, "converted"].mean()
uplift         = worker_conv - no_worker_conv

print(f"\nConversion rate WITH worker activity:    {worker_conv:.1%}")
print(f"Conversion rate WITHOUT worker activity: {no_worker_conv:.1%}")
print(f"Uplift from worker adoption:             +{uplift*100:.1f} percentage points")

# Chi-square test
ct = pd.crosstab(org["has_any_worker_activity"], org["converted"])
chi2, p, _, _ = chi2_contingency(ct)
print(f"Chi-square p-value: {p:.4f}  ({'SIGNIFICANT' if p < 0.05 else 'not significant'})")

# ── Chart 1: The Headline ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("The Core Finding: Worker Adoption & Conversion", fontsize=15, fontweight="bold")

# Bar: conversion rates
ax = axes[0]
groups = ["No Worker\nActivity", "Worker\nActivity Present"]
rates  = [no_worker_conv * 100, worker_conv * 100]
colors = [LIGHT, ACCENT]
bars   = ax.bar(groups, rates, color=colors, edgecolor="white", width=0.5)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Rate\nby Worker Adoption")
ax.set_ylim(0, max(rates) * 1.35)
ax.annotate(f"+{uplift*100:.1f}pp uplift",
            xy=(1, worker_conv*100), xytext=(0.5, worker_conv*100 + 5),
            fontsize=11, color=ACCENT, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ACCENT))

# Donut: how many orgs had worker activity
ax = axes[1]
n_worker    = org["has_any_worker_activity"].sum()
n_no_worker = total_orgs - n_worker
sizes  = [n_worker, n_no_worker]
labels = [f"Workers Used App\n({n_worker} orgs, {n_worker/total_orgs:.0%})",
          f"Admin Only\n({n_no_worker} orgs, {n_no_worker/total_orgs:.0%})"]
wedges, texts = ax.pie(sizes, labels=labels, colors=[ACCENT, LIGHT],
                       startangle=90, wedgeprops=dict(width=0.55))
ax.set_title("Proportion of Orgs\nWith Worker Activity")

# Worker activity events breakdown
ax = axes[2]
worker_act_counts = df[df["activity_type"] == "worker"]["activity_name"].value_counts()
short_names = {
    "Mobile.Schedule.Loaded":           "Mobile Schedule Viewed",
    "PunchClock.PunchedIn":             "Punch Clock In",
    "Scheduling.Availability.Set":      "Availability Set",
    "ShiftDetails.View.Opened":         "Shift Details Viewed",
    "Absence.Request.Created":          "Absence Request",
    "PunchClock.PunchedOut":            "Punch Clock Out",
    "Break.Activate.Started":           "Break Started",
    "Break.Activate.Finished":          "Break Finished",
    "PunchClockEndNote.Add.Completed":  "Clock-Out Note",
    "PunchClockStartNote.Add.Completed":"Clock-In Note",
    "Scheduling.ShiftSwap.Created":     "Shift Swap Request",
    "Scheduling.ShiftHandover.Created": "Shift Handover",
    "Scheduling.OpenShiftRequest.Created": "Open Shift Request",
    "Shift.View.Opened":                "Shift View",
    "Absence.Request.Created":          "Absence Request",
}
worker_act_counts.index = [short_names.get(x, x) for x in worker_act_counts.index]
ax.barh(worker_act_counts.index[::-1], worker_act_counts.values[::-1],
        color=ACCENT, edgecolor="white", alpha=0.85)
ax.set_xlabel("Total Events")
ax.set_title("Worker Activity\nBreakdown (All Orgs)")

plt.tight_layout()
plt.savefig(f"{OUT}/01_worker_adoption_headline.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 1 saved")

# 4. SPECIFIC WORKER ACTIVITIES vs CONVERSION

print("\n" + "=" * 60)
print("4. EACH WORKER ACTIVITY vs CONVERSION RATE")
print("=" * 60)

worker_flags = {
    "Any Worker Activity":    "has_any_worker_activity",
    "Punch Clock Used":       "has_punchclock",
    "Availability Set":       "has_availability",
    "Shift Swap/Handover":    "has_shift_swap",
    "Absence Request":        "has_absence_request",
    "Mobile Schedule View":   "has_mobile_view",
}

results = []
for label, col in worker_flags.items():
    yes_orgs = org[org[col] == 1]
    no_orgs  = org[org[col] == 0]
    conv_yes = yes_orgs["converted"].mean()
    conv_no  = no_orgs["converted"].mean()
    uplift_v = conv_yes - conv_no
    n_yes    = len(yes_orgs)
    ct_v     = pd.crosstab(org[col], org["converted"])
    if ct_v.shape == (2, 2):
        chi2_v, p_v, _, _ = chi2_contingency(ct_v)
    else:
        p_v = 1.0
    results.append({
        "Activity":    label,
        "N_Orgs":      n_yes,
        "Pct_Orgs":    n_yes / total_orgs * 100,
        "Conv_Yes":    conv_yes * 100,
        "Conv_No":     conv_no * 100,
        "Uplift_pp":   uplift_v * 100,
        "p_value":     p_v,
        "Significant": p_v < 0.05,
    })

res_df = pd.DataFrame(results).sort_values("Uplift_pp", ascending=False)
print(res_df.to_string(index=False))

# ── Chart 2: Worker Activity Uplift ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Worker Activity Impact on Conversion", fontsize=14, fontweight="bold")

ax = axes[0]
colors_bar = [ACCENT if row.Significant else GOLD for row in res_df.itertuples()]
bars = ax.barh(res_df["Activity"][::-1], res_df["Uplift_pp"][::-1],
               color=colors_bar[::-1], edgecolor="white")
for bar, row in zip(bars, res_df[::-1].itertuples()):
    label = f"+{row.Uplift_pp:.1f}pp  (p={row.p_value:.3f}{'*' if row.Significant else ''})"
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            label, va="center", fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Conversion Uplift (percentage points)")
ax.set_title("Uplift by Worker Activity Type\n(red = statistically significant)")
ax.set_xlim(-2, res_df["Uplift_pp"].max() + 15)

ax = axes[1]
x = np.arange(len(res_df))
w = 0.35
ax.bar(x - w/2, res_df["Conv_Yes"], width=w, label="Did this activity", color=ACCENT, alpha=0.85, edgecolor="white")
ax.bar(x + w/2, res_df["Conv_No"],  width=w, label="Did NOT do this",   color=LIGHT,  alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(res_df["Activity"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Rate: Did vs Did Not\nDo Each Worker Activity")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/02_worker_activity_uplift.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 2 saved")


# 5. THE THREE COMPANY ARCHETYPES

print("\n" + "=" * 60)
print("5. THREE COMPANY ARCHETYPES")
print("=" * 60)

def classify_org(row):
    if row["worker_events"] > 0 and row["admin_events"] > 0:
        return "Both Admin + Workers Active"
    elif row["admin_events"] > 0 and row["worker_events"] == 0:
        return "Admin Only (Workers Never Joined)"
    else:
        return "Minimal / No Engagement"

org["archetype"] = org.apply(classify_org, axis=1)

arch_summary = org.groupby("archetype").agg(
    n_orgs=("organization_id", "count"),
    conv_rate=("converted", "mean"),
    avg_admin_events=("admin_events", "mean"),
    avg_worker_events=("worker_events", "mean"),
    avg_active_days=("admin_active_days", "mean"),
).round(2)
arch_summary["conv_rate_pct"] = arch_summary["conv_rate"] * 100
print(arch_summary)

# ── Chart 3: Three Archetypes ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("The Three Company Archetypes", fontsize=15, fontweight="bold")

archetype_colors = {
    "Both Admin + Workers Active":      ACCENT,
    "Admin Only (Workers Never Joined)": LIGHT,
    "Minimal / No Engagement":           GREY,
}

ax = axes[0]
arch_order = ["Both Admin + Workers Active", "Admin Only (Workers Never Joined)", "Minimal / No Engagement"]
rates = [arch_summary.loc[a, "conv_rate_pct"] if a in arch_summary.index else 0 for a in arch_order]
cols  = [archetype_colors[a] for a in arch_order]
short = ["Both Active", "Admin Only", "No Engagement"]
bars  = ax.bar(short, rates, color=cols, edgecolor="white")
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{rate:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Rate\nby Archetype")
ax.set_ylim(0, max(rates) * 1.4)

ax = axes[1]
n_vals = [arch_summary.loc[a, "n_orgs"] if a in arch_summary.index else 0 for a in arch_order]
wedges, texts, autotexts = ax.pie(n_vals, labels=short, colors=cols,
                                   autopct="%1.0f%%", startangle=90,
                                   wedgeprops=dict(width=0.55))
ax.set_title("Distribution of\nOrganisations")

ax = axes[2]
admin_avg  = [arch_summary.loc[a, "avg_admin_events"]  if a in arch_summary.index else 0 for a in arch_order]
worker_avg = [arch_summary.loc[a, "avg_worker_events"] if a in arch_summary.index else 0 for a in arch_order]
x = np.arange(len(short)); w = 0.35
ax.bar(x - w/2, admin_avg,  width=w, label="Admin Events",  color=LIGHT,  alpha=0.85, edgecolor="white")
ax.bar(x + w/2, worker_avg, width=w, label="Worker Events", color=ACCENT, alpha=0.85, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
ax.set_ylabel("Avg Events per Org")
ax.set_title("Admin vs Worker\nEvent Volume by Archetype")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/03_three_archetypes.png", dpi=150, bbox_inches="tight")
plt.close()
print(" Chart 3 saved")


# 6. WORKER ADOPTION FUNNEL (THE NEW FUNNEL)

print("\n" + "=" * 60)
print("6. THE WORKER ADOPTION FUNNEL")
print("=" * 60)

admin_setup     = (org["admin_events"] > 0).sum()
shift_created   = (df["activity_name"] == "Scheduling.Shift.Created").groupby(df["organization_id"]).any().sum()
mobile_viewed   = org["has_mobile_view"].sum()
worker_any      = org["has_any_worker_activity"].sum()
punchclock_used = org["has_punchclock"].sum()
converted_n     = org["converted"].sum()

funnel_data = {
    "Trial Started":                   total_orgs,
    "Admin Set Up Shifts":             int(shift_created),
    "Mobile Schedule Viewed":          int(mobile_viewed),
    "Any Worker Activity":             int(worker_any),
    "Workers Punched In/Out":          int(punchclock_used),
    "Converted to Paid":               int(converted_n),
}

funnel_df = pd.DataFrame(list(funnel_data.items()), columns=["Stage", "Count"])
funnel_df["Pct_Total"]   = funnel_df["Count"] / total_orgs * 100
funnel_df["Drop_off"]    = (1 - funnel_df["Count"] / funnel_df["Count"].shift(1)) * 100
funnel_df.loc[0, "Drop_off"] = 0
print(funnel_df.to_string(index=False))

# ── Chart 4: New Worker Funnel ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
stage_colors = [LIGHT, LIGHT, GOLD, ACCENT, ACCENT, GREEN]
bars = ax.barh(funnel_df["Stage"][::-1], funnel_df["Count"][::-1],
               color=stage_colors[::-1], edgecolor="white", linewidth=0.5)
for bar, row in zip(bars, funnel_df[::-1].itertuples()):
    pct_str  = f"{row.Pct_Total:.1f}% of all orgs"
    drop_str = f"  (↓{row.Drop_off:.0f}% drop)" if row.Drop_off > 0 else ""
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
            f"{int(row.Count)}  —  {pct_str}{drop_str}", va="center", fontsize=9)
ax.set_xlabel("Number of Organisations")
ax.set_title("The Worker Adoption Funnel\n(blue = admin stages | gold = handoff | red/green = worker stages)",
             fontsize=13)
ax.set_xlim(0, total_orgs * 1.55)
plt.tight_layout()
plt.savefig(f"{OUT}/04_worker_adoption_funnel.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved")


# 7. CONVERSION RATE BY WORKER ENGAGEMENT DEPTH

print("\n" + "=" * 60)
print("7. CONVERSION RATE BY WORKER ENGAGEMENT DEPTH")
print("=" * 60)

worker_act_flags = ["has_punchclock", "has_availability", "has_shift_swap",
                    "has_absence_request", "has_mobile_view"]
org["worker_depth_score"] = org[worker_act_flags].sum(axis=1)

depth_summary = org.groupby("worker_depth_score").agg(
    n_orgs=("organization_id", "count"),
    conv_rate=("converted", "mean")
).reset_index()
depth_summary["conv_rate_pct"] = depth_summary["conv_rate"] * 100
print(depth_summary)

# ── Chart 5: Worker Depth vs Conversion ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Worker Engagement Depth vs Conversion", fontsize=14, fontweight="bold")

ax = axes[0]
bar_colors = [GREY if s == 0 else ACCENT for s in depth_summary["worker_depth_score"]]
bars = ax.bar(depth_summary["worker_depth_score"], depth_summary["conv_rate_pct"],
              color=bar_colors, edgecolor="white")
for bar, rate in zip(bars, depth_summary["conv_rate_pct"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Number of Different Worker Activity Types Used (0–5)")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("More Worker Activity Types = Higher Conversion")
ax.set_xticks(depth_summary["worker_depth_score"])

ax = axes[1]
ax.bar(depth_summary["worker_depth_score"], depth_summary["n_orgs"],
       color=[GREY if s == 0 else LIGHT for s in depth_summary["worker_depth_score"]],
       edgecolor="white")
for _, row in depth_summary.iterrows():
    ax.text(row["worker_depth_score"], row["n_orgs"] + 2,
            str(int(row["n_orgs"])), ha="center", fontsize=10)
ax.set_xlabel("Number of Different Worker Activity Types Used")
ax.set_ylabel("Number of Organisations")
ax.set_title("How Many Orgs Reached Each Depth Level")
ax.set_xticks(depth_summary["worker_depth_score"])

plt.tight_layout()
plt.savefig(f"{OUT}/05_worker_depth_conversion.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 5 saved")


# 8. TIME TO WORKER ADOPTION — HOW FAST DID WORKERS JOIN?

print("\n" + "=" * 60)
print("8. SPEED OF WORKER ADOPTION")
print("=" * 60)

worker_timing = org[org["hours_to_first_worker"].notna()].copy()
worker_timing["days_to_first_worker"] = worker_timing["hours_to_first_worker"] / 24

for conv, label in [(True, "Converters"), (False, "Non-converters")]:
    data = worker_timing.loc[worker_timing["converted"] == conv, "days_to_first_worker"]
    print(f"\n{label} — days to first worker activity:")
    print(f"  Median: {data.median():.1f}  |  Mean: {data.mean():.1f}  |  P25: {data.quantile(0.25):.1f}  |  P75: {data.quantile(0.75):.1f}")

# Mann-Whitney test on timing
c  = worker_timing.loc[worker_timing["converted"],  "days_to_first_worker"].dropna()
nc = worker_timing.loc[~worker_timing["converted"], "days_to_first_worker"].dropna()
u, p = mannwhitneyu(c, nc, alternative="two-sided")
print(f"\nMann-Whitney p-value: {p:.4f}  ({'SIGNIFICANT' if p < 0.05 else 'not significant'})")

# Cumulative: % of orgs that had worker activity by day N
worker_timing2 = org.copy()
worker_timing2["days_to_first_worker"] = worker_timing2["hours_to_first_worker"] / 24

# ── Chart 6: Speed of Worker Adoption ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Speed of Worker Adoption", fontsize=14, fontweight="bold")

ax = axes[0]
cap = worker_timing["days_to_first_worker"].quantile(0.95)
for conv, label, color in [(True, "Converted", ACCENT), (False, "Not Converted", LIGHT)]:
    data = worker_timing.loc[worker_timing["converted"] == conv, "days_to_first_worker"].clip(upper=cap)
    ax.hist(data, bins=25, color=color, alpha=0.7, label=label, edgecolor="white")
ax.set_xlabel("Days to First Worker Activity (capped at 95th pct)")
ax.set_ylabel("Organisations")
ax.set_title(f"Timing of First Worker Activity\n(Mann-Whitney p={p:.3f})")
ax.legend()

ax = axes[1]
days_range = range(0, 31)
for conv, label, color in [(True, "Converted", ACCENT), (False, "Not Converted", LIGHT)]:
    subset = worker_timing2[worker_timing2["converted"] == conv]
    total_subset = len(subset)
    cum_pct = []
    for d in days_range:
        n = (subset["days_to_first_worker"] <= d).sum()
        cum_pct.append(n / total_subset * 100)
    ax.plot(list(days_range), cum_pct, color=color, linewidth=2, label=label)
    ax.fill_between(list(days_range), cum_pct, alpha=0.1, color=color)
ax.set_xlabel("Day of Trial")
ax.set_ylabel("% of Orgs with Worker Activity")
ax.set_title("Cumulative Worker Adoption Curve\nby Conversion Outcome")
ax.legend()
ax.axvline(7, color=GREY, linestyle="--", linewidth=1, label="Day 7")

plt.tight_layout()
plt.savefig(f"{OUT}/06_worker_adoption_speed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 6 saved")


# 9. ADMIN vs WORKER EVENT RATIO# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. ADMIN vs WORKER EVENT RATIO ANALYSIS")
print("=" * 60)

for conv, label in [(True, "Converters"), (False, "Non-converters")]:
    subset = org[org["converted"] == conv]
    print(f"\n{label}:")
    print(f"  Avg admin events:  {subset['admin_events'].mean():.1f}")
    print(f"  Avg worker events: {subset['worker_events'].mean():.1f}")
    print(f"  Worker event share: {subset['worker_event_share'].mean():.1%}")

# ── Chart 7: Admin vs Worker Split ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Admin vs Worker Activity: Converters vs Non-Converters", fontsize=14, fontweight="bold")

ax = axes[0]
labels_g = ["Non-Converters", "Converters"]
admin_means  = [org.loc[~org["converted"], "admin_events"].mean(),
                org.loc[ org["converted"], "admin_events"].mean()]
worker_means = [org.loc[~org["converted"], "worker_events"].mean(),
                org.loc[ org["converted"], "worker_events"].mean()]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, admin_means,  width=w, label="Admin Events",  color=LIGHT,  edgecolor="white")
ax.bar(x + w/2, worker_means, width=w, label="Worker Events", color=ACCENT, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels_g)
ax.set_ylabel("Avg Events per Org")
ax.set_title("Average Event Volume\nby Type & Outcome")
ax.legend()

ax = axes[1]
for conv, label, color in [(True, "Converted", ACCENT), (False, "Not Converted", LIGHT)]:
    data = org.loc[org["converted"] == conv, "worker_event_share"] * 100
    ax.hist(data, bins=20, color=color, alpha=0.7, label=label, edgecolor="white")
ax.set_xlabel("Worker Events as % of Total Events")
ax.set_ylabel("Organisations")
ax.set_title("Worker Event Share Distribution")
ax.legend()

ax = axes[2]
ax.scatter(
    org.loc[~org["converted"], "admin_events"].clip(upper=200),
    org.loc[~org["converted"], "worker_events"].clip(upper=100),
    color=LIGHT, alpha=0.4, s=20, label="Not Converted"
)
ax.scatter(
    org.loc[org["converted"], "admin_events"].clip(upper=200),
    org.loc[org["converted"], "worker_events"].clip(upper=100),
    color=ACCENT, alpha=0.6, s=25, label="Converted"
)
ax.set_xlabel("Admin Events (capped 200)")
ax.set_ylabel("Worker Events (capped 100)")
ax.set_title("Admin vs Worker Activity\nScatter by Outcome")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/07_admin_vs_worker_split.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 7 saved")


# 10. PREDICTIVE MODELS — WORKER FEATURES ONLY vs ALL FEATURES

print("\n" + "=" * 60)
print("10. PREDICTIVE MODELS — DOES WORKER DATA IMPROVE PREDICTION?")
print("=" * 60)

y = org["converted"].astype(int)

# Model A: Admin features only (original analysis)
admin_feat = ["admin_events", "admin_unique_activities", "admin_active_days"]
XA = org[admin_feat].fillna(0)
XAs = StandardScaler().fit_transform(XA)

# Model B: Worker features only
worker_feat = ["worker_events", "worker_unique_activities", "worker_active_days",
               "has_punchclock", "has_availability", "has_shift_swap",
               "has_absence_request", "has_mobile_view", "worker_depth_score"]
XW = org[worker_feat].fillna(0)
XWs = StandardScaler().fit_transform(XW)

# Model C: Combined
combined_feat = admin_feat + worker_feat
XC = org[combined_feat].fillna(0)
XCs = StandardScaler().fit_transform(XC)

models_to_test = {
    "Admin Features Only":    (XAs, XA),
    "Worker Features Only":   (XWs, XW),
    "Admin + Worker Combined":(XCs, XC),
}

rf_results = {}
for name, (Xs, X) in models_to_test.items():
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    rf_results[name] = (scores.mean(), scores.std())
    print(f"{name:<30}  AUC = {scores.mean():.3f} ± {scores.std():.3f}")

# ── Chart 8: Model Comparison ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names  = list(rf_results.keys())
means  = [v[0] for v in rf_results.values()]
stds   = [v[1] for v in rf_results.values()]
colors_m = [LIGHT, ACCENT, GOLD]
bars   = ax.bar(names, means, yerr=stds, color=colors_m, capsize=6, edgecolor="white")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f"{m:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(0.5, color=GREY, linestyle="--", linewidth=1, label="Random chance (0.5)")
ax.set_ylim(0.45, 0.75)
ax.set_ylabel("CV ROC-AUC (higher = better prediction)")
ax.set_title("Does Adding Worker Data Improve Conversion Prediction?\n(Random Forest, 5-fold CV)",
             fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/08_model_comparison_worker_vs_admin.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 8 saved")


# 11. SURVIVAL ANALYSIS — BY WORKER ADOPTION

print("\n" + "=" * 60)
print("11. SURVIVAL ANALYSIS BY WORKER ADOPTION")
print("=" * 60)

surv = org.copy()
surv["duration"] = np.where(surv["converted"], surv["days_to_conversion"].fillna(30), 30.0)
surv["duration"] = surv["duration"].clip(0.5, 30)
surv["event"]    = surv["converted"].astype(int)

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(12, 5))

groups = [
    ("Workers Active",      surv["has_any_worker_activity"] == 1, ACCENT),
    ("Admin Only",          surv["has_any_worker_activity"] == 0, LIGHT),
]
for label, mask, color in groups:
    sub = surv[mask]
    kmf.fit(sub["duration"], event_observed=sub["event"], label=f"{label} (n={mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)

ax.set_xlabel("Days into Trial")
ax.set_ylabel("Probability of NOT Yet Converting")
ax.set_title("Kaplan-Meier: Time to Conversion\nWorker-Active Orgs vs Admin-Only Orgs", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/09_survival_worker_vs_admin.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 9 saved")


# 12. THE HANDOFF PROBLEM — HOW LONG BEFORE WORKERS JOIN AFTER ADMIN SETUP?

print("\n" + "=" * 60)
print("12. THE HANDOFF GAP — ADMIN SETUP TO WORKER ADOPTION")
print("=" * 60)

handoff = org[org["first_worker_ts"].notna() & org["first_admin_ts"].notna()].copy()
handoff["handoff_gap_hrs"] = (
    (handoff["first_worker_ts"] - handoff["first_admin_ts"]).dt.total_seconds() / 3600
).clip(lower=0)
handoff["handoff_gap_days"] = handoff["handoff_gap_hrs"] / 24

print(f"Orgs where we can measure handoff gap: {len(handoff)}")
print(f"\nHandoff gap (hours from admin setup to first worker activity):")
print(f"  Median: {handoff['handoff_gap_hrs'].median():.1f} hrs ({handoff['handoff_gap_days'].median():.1f} days)")
print(f"  Mean:   {handoff['handoff_gap_hrs'].mean():.1f} hrs")
print(f"  P25:    {handoff['handoff_gap_hrs'].quantile(0.25):.1f} hrs")
print(f"  P75:    {handoff['handoff_gap_hrs'].quantile(0.75):.1f} hrs")

for conv, label in [(True, "Converters"), (False, "Non-converters")]:
    data = handoff.loc[handoff["converted"] == conv, "handoff_gap_hrs"].dropna()
    print(f"\n  {label}: median gap = {data.median():.1f} hrs")

# Buckets
buckets = [(0, 24, "Same day"), (24, 72, "1–3 days"), (72, 168, "3–7 days"), (168, 9999, "7+ days")]
print("\nHandoff gap distribution:")
for lo, hi, label in buckets:
    n = ((handoff["handoff_gap_hrs"] >= lo) & (handoff["handoff_gap_hrs"] < hi)).sum()
    conv_r = handoff.loc[(handoff["handoff_gap_hrs"] >= lo) & (handoff["handoff_gap_hrs"] < hi), "converted"].mean()
    print(f"  {label:<15}: {n:>4} orgs  |  conv rate: {conv_r:.1%}")

# ── Chart 10: The Handoff Gap ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("The Handoff Gap: How Long Before Workers Join After Admin Setup?",
             fontsize=14, fontweight="bold")

ax = axes[0]
cap = handoff["handoff_gap_days"].quantile(0.95)
for conv, label, color in [(True, "Converted", ACCENT), (False, "Not Converted", LIGHT)]:
    data = handoff.loc[handoff["converted"] == conv, "handoff_gap_days"].clip(upper=cap)
    ax.hist(data, bins=25, color=color, alpha=0.7, label=label, edgecolor="white")
ax.set_xlabel("Days from Admin Setup to First Worker Activity")
ax.set_ylabel("Organisations")
ax.set_title("Handoff Gap Distribution")
ax.legend()

ax = axes[1]
bucket_labels = ["Same Day", "1–3 Days", "3–7 Days", "7+ Days"]
bucket_counts  = []
bucket_conv    = []
for lo, hi, label in buckets:
    mask = (handoff["handoff_gap_hrs"] >= lo) & (handoff["handoff_gap_hrs"] < hi)
    bucket_counts.append(mask.sum())
    bucket_conv.append(handoff.loc[mask, "converted"].mean() * 100 if mask.sum() > 0 else 0)

x = np.arange(len(bucket_labels)); w = 0.35
ax2 = ax.twinx()
ax.bar(x, bucket_counts, width=0.6, color=LIGHT, alpha=0.7, edgecolor="white", label="N Orgs")
ax2.plot(x, bucket_conv, color=ACCENT, marker="o", linewidth=2, markersize=8, label="Conv Rate %")
ax.set_xticks(x); ax.set_xticklabels(bucket_labels)
ax.set_ylabel("Number of Organisations", color=LIGHT)
ax2.set_ylabel("Conversion Rate (%)", color=ACCENT)
ax.set_title("Handoff Speed vs Conversion Rate\n(faster handoff = higher conversion?)")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT}/10_handoff_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 10 saved")

