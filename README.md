# Worker vs Admin Adoption Analysis

A re-analysis of trial conversion data through a new lens: **did workers (employees) actually adopt the product after the admin set it up?**

This analysis was motivated by a key observation from the initial study — converters and non-converters looked almost identical in overall behaviour. The hypothesis: **the original analysis was measuring mostly admin activity**, and the real conversion signal might lie in whether *employees* showed up.

---

## The Hypothesis

This product serves two distinct user groups:

| User | Role | Key Actions |
|------|------|-------------|
| **Admin / Manager** | Sets up the schedule, manages the team | Create shifts, approve timesheets, export payroll |
| **Worker / Employee** | Lives inside the schedule day-to-day | Punch in/out, view mobile schedule, set availability, request swaps |

An admin can build a perfect schedule. But if no worker ever opens the app, the product has delivered zero value to the actual workforce — and the company is unlikely to pay for it.

---

## The Surprising Finding

**The hypothesis was wrong — but in a revealing way.**

| Group | N Orgs | Conversion Rate |
|-------|--------|----------------|
| Any worker activity present | 477 (49%) | 21.0% |
| Admin only — zero worker activity | 489 (51%) | 21.7% |

Worker adoption does **not** significantly predict conversion (p = 0.85). The two groups convert at almost exactly the same rate.

But this finding is itself the insight. Here is what it tells us:

---

## What We Actually Discovered

### 1. Half of all trialling companies never got a single worker onto the platform

**489 out of 966 companies (51%)** had zero worker-side activity. The admin set things up, and nobody on the team joined. Yet these companies convert at the same rate as companies where workers are actively using the product.

This means **conversion is not a product-experience decision at all.** It is being made at the admin/procurement level — often driven by pricing, timing, and sales conversations — completely independent of whether the workforce found value.

### 2. The three company archetypes

| Archetype | N Orgs | Conv Rate |
|-----------|--------|-----------|
| Both Admin + Workers Active | 371 (38%) | 21.0% |
| Admin Only — Workers Never Joined | 489 (51%) | **22.0%** |
| Minimal / No Engagement | 106 (11%) | 20.0% |

Strikingly, **admin-only companies convert at a slightly higher rate** than companies where workers are active. This further confirms that whoever is making the purchase decision is not asking "did my team love it?" — they are asking "did I see enough to justify the cost?"

### 3. Worker depth does show a weak positive trend

When we measure how many *types* of worker activities occurred (0–5 scale):

| Worker Activity Types Used | Conv Rate |
|---------------------------|-----------|
| 0 (none) | 21.6% |
| 1 | 19.0% |
| 2 | 23.6% |
| 3 | **25.0%** |
| 4 | 23.1% |

There is a weak upward trend from depth score 2 onward, but numbers are too small at the higher levels to draw firm conclusions.

### 4. Worker activity is dominated by one action: Mobile Schedule Loaded

Of all 58,230 worker events in the dataset:

| Activity | Events | % of Worker Events |
|----------|--------|-------------------|
| Mobile.Schedule.Loaded | 49,540 | **85%** |
| PunchClock.PunchedIn | 4,833 | 8% |
| ShiftDetails.View.Opened | 1,433 | 2% |
| Scheduling.Availability.Set | 1,084 | 2% |
| All other worker activities | 1,340 | 2% |

Workers are primarily just *viewing* the schedule. They are not yet self-managing (swapping shifts, setting availability, requesting handovers). This is a very passive form of adoption.

### 5. The handoff from admin to worker is almost instant — when it happens at all

For companies where workers did join:
- Median time from first admin action to first worker action: **less than 1 hour**
- 82% of handoffs happen on the **same day** as admin setup

This is not a timing problem. When workers join, they join quickly. The problem is that **51% of companies never bridge the gap at all.**

### 6. Adding worker data does not improve conversion prediction

| Model | Features Used | ROC-AUC |
|-------|--------------|---------|
| Random Forest | Admin features only | 0.515 |
| Random Forest | Worker features only | 0.514 |
| Random Forest | Admin + Worker combined | 0.520 |

All three models perform near chance (0.5). Worker data adds virtually nothing to predictive power. Conversion is being driven by factors outside the product entirely.

---

## Revised Interpretation of the Conversion Problem

The original analysis said: *"Converters and non-converters look identical."*

This analysis adds: *"They look identical because neither worker adoption nor admin engagement predicts conversion. The purchase decision is happening outside the product."*

This points to one conclusion: **the key missing data is not behavioural — it is contextual.**

What we do not have, but would explain most of the variance:
- Was there a sales conversation during the trial?
- What price was the company shown?
- What is the company size / industry?
- Did the company come from a paid ad, a referral, or organic search?
- Was there a competitor evaluation happening simultaneously?

---

## What This Means for the Business

### The mobile onboarding gap is a product health problem, not a conversion problem
51% of companies not getting workers onto the platform is a signal that the product is not being fully experienced — even if it doesn't directly hurt conversion today. Over time, low worker adoption likely increases churn *after* conversion (post-paid), which will show up in renewal data.

**Recommendation**: Trigger an automated email to admins who have created shifts but had zero worker activity within 48 hours — with a simple "share with your team" link. This improves product health and long-term retention, even if it does not immediately move trial conversion.

### Focus conversion resources on the purchase decision, not the product experience
Since conversion is not driven by product behaviour, resources spent on in-app onboarding nudges and feature tours will have limited impact on trial-to-paid rates. The higher-leverage investments are:
- **Sales touchpoints at days 14–21** before trial urgency
- **Capturing acquisition source and company size** to identify which channels bring high-intent buyers
- **Pricing and packaging experiments** to reduce the friction at the payment decision moment

### Punch clock usage as a long-term health signal
Only 211 orgs (22%) had any punch clock activity. Punch clock usage means workers are using the platform for live, time-critical operations — this is the deepest form of adoption. While it doesn't predict trial conversion, it almost certainly predicts **post-conversion retention and expansion**. Tracking this metric in the paid customer base is recommended.

---

## Project Structure

```
worker-adoption-analysis/
├── worker_analysis.py              # Full re-analysis pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── org_worker_features.csv     # Org-level feature matrix (generated)
├── charts/
│   ├── 01_worker_adoption_headline.png
│   ├── 02_worker_activity_uplift.png
│   ├── 03_three_archetypes.png
│   ├── 04_worker_adoption_funnel.png
│   ├── 05_worker_depth_conversion.png
│   ├── 06_worker_adoption_speed.png
│   ├── 07_admin_vs_worker_split.png
│   ├── 08_model_comparison_worker_vs_admin.png
│   ├── 09_survival_worker_vs_admin.png
│   └── 10_handoff_gap.png
└── sql/
    └── mart_worker_adoption.sql    # dbt-style mart model
```

---

## Setup

```bash
pip install -r requirements.txt
python worker_analysis.py
```

---

## Limitations

- `Mobile.Schedule.Loaded` is classified as a worker activity but could also be an admin checking their own schedule. This inflates the "worker activity" count and likely means true worker-only adoption is even lower than 49%.
- No user-level data — we cannot distinguish between one worker using the app 50 times vs 50 workers using it once each. Org-level aggregation hides this.
- No post-conversion data — punch clock and worker adoption may strongly predict *retention* after conversion, which this dataset cannot show.
- Sample size for deep worker engagement (depth score 3–5) is small (n < 75), making those conversion rates unreliable.
