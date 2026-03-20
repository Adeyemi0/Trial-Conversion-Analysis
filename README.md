# Trial Conversion Analysis
### Why Do Only 1 in 5 Trial Companies Convert?

**Author:** Adediran Adeyemi
**Website:** [www.adediranadeyemi.com](https://www.adediranadeyemi.com)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Context and Motivation](#business-context-and-motivation)
3. [The Central Question](#the-central-question)
4. [Data Overview](#data-overview)
5. [Methodology and Analytical Approach](#methodology-and-analytical-approach)
6. [Key Findings](#key-findings)
7. [The Admin vs Worker Framework](#the-admin-vs-worker-framework)
8. [Statistical Analysis](#statistical-analysis)
9. [Predictive Modelling](#predictive-modelling)
10. [SQL Data Models](#sql-data-models)
11. [Recommendations](#recommendations)
12. [Limitations and Honest Caveats](#limitations-and-honest-caveats)
13. [Setup and Usage](#setup-and-usage)

---

## Executive Summary

This project analyses trial-to-paid conversion behaviour across **966 trialling organisations** of a B2B workforce scheduling platform. Over 102,895 product events were analysed across a 30-day trial window to understand what separates companies that convert from those that do not.

![Conversion Overview](charts/01_conversion_overview.png)

**The headline finding is counterintuitive and important:** converters and non-converters are statistically indistinguishable in their product behaviour. Three separate machine learning models all scored near random chance (AUC 0.48), and no single activity, feature, or behavioural metric significantly predicts who pays.

A second layer of analysis examined whether the product serves two distinct user types — admins and workers — differently. This framework revealed that **51% of trialling organisations never got a single employee onto the platform** after the admin set it up. The product was configured for a workforce that never arrived. This does not directly drive conversion but is a structural risk to post-conversion retention.

The most actionable conclusion from the full analysis is that conversion is a procurement decision driven by external factors (pricing, sales conversations, budget cycles, acquisition channel) rather than a product experience decision. The data needed to explain conversion does not currently exist in the product event log. Collecting it is the highest-leverage next step.

**Three concrete, immediately executable recommendations emerge:**
- Capture company size, industry, and acquisition source at signup
- Trigger an automated worker onboarding nudge 48 hours after shift creation with zero worker activity
- Deploy proactive CS outreach between Days 14 and 21 of the trial, before deadline urgency sets in

---

## Business Context and Motivation

This is a B2B SaaS product for workforce scheduling. Businesses use it to build rotas, manage shift assignments, handle time and attendance, process absence requests, and export payroll. It serves two fundamentally different users within the same account.

**Admins** are managers or business owners. They set up the schedule, create shifts, approve timesheets, and manage the team. They are typically the ones who sign up for the trial and make the purchasing decision.

**Workers** are the employees. They view their schedule on mobile, clock in and out, set their availability, and request swaps or handovers. They do not make the purchasing decision, but they are the ones whose daily work lives depend on the product.

This distinction matters enormously for understanding product adoption, churn risk, and conversion. An analysis that treats all events as equivalent misses the fact that admin activity and worker activity tell very different stories about how deeply a product has been embedded in a business.

The motivation for building this analysis around that admin/worker distinction came directly from the data. An initial pass treating all events equally found that converters and non-converters looked essentially the same. That result prompted the question: are we measuring the right users? The answer shaped the entire analytical framework.

---

## The Central Question

> **What does a company do during its trial that predicts whether it will pay?**

The assumption going in was straightforward, companies that engage more deeply with the product, use more features, stay active for more days, and get more employees onto the platform should be more likely to convert. This is the standard product-led growth hypothesis.

This analysis tests that hypothesis rigorously across multiple dimensions. The answer challenges it at every level. The insight that falls out of that challenge is more useful than a simple confirmation would have been.

---

## Data Overview

| Field | Type | Description |
|-------|------|-------------|
| `organization_id` | string | Unique identifier for each trialling organisation |
| `activity_name` | string | Name of the product activity performed |
| `timestamp` | datetime | When the activity occurred |
| `converted` | boolean | Whether the organisation converted to paid |
| `converted_at` | datetime | Timestamp of conversion (null if not converted) |
| `trial_start` | datetime | When the trial started |
| `trial_end` | datetime | Trial expiry date (trial_start + 30 days) |

**Dataset dimensions after cleaning:**

| Metric | Value |
|--------|-------|
| Raw rows | 170,526 |
| Rows after deduplication | 102,895 |
| Duplicate rows removed | 67,631 (40%) |
| Unique organisations | 966 |
| Unique activity types | 28 |
| Overall conversion rate | 21.3% |
| Converted organisations | 206 |
| Non-converted organisations | 760 |

**Data quality steps applied:**
- Exact deduplication across all 7 columns removed 67,631 rows (40% of raw data)
- All datetime columns parsed with error coercion; no invalid timestamps found after cleaning
- Zero events found outside the trial window (before `trial_start` or after `trial_end`)
- Negative time-to-first-activity values clipped to zero

**Activity classification:** All 28 activity types were manually classified into two groups based on product logic.

Worker activities are actions that only an employee would perform: clocking in or out, viewing the mobile schedule, setting availability, requesting shift swaps or handovers, requesting open shifts, viewing shift details, and creating absence requests.

Admin activities are actions that only a manager or owner would perform: creating and assigning shifts, applying templates, approving timesheets, bulk-approving shifts, exporting payroll to Xero, creating budget entries, approving or rejecting absence requests, and editing time clock entries.

---

## Methodology and Analytical Approach

The analysis was structured in three layers.

### Layer 1: Overall Behavioural Analysis

The first layer treated all 966 organisations equally and tested whether total engagement (event count, active days, unique features used, time to first activity) differs between converters and non-converters. Mann-Whitney U tests were used rather than t-tests because the distributions are highly right-skewed; the non-parametric test makes no assumptions about normality.

Chi-square tests were applied to all binary feature flags (did this organisation use this activity at least once?) against the conversion outcome. Point-biserial correlation was calculated between continuous engagement metrics and the binary conversion variable.

![Feature Distributions](charts/02_feature_distributions.png)

### Layer 2: Admin vs Worker Segmentation

The second layer split all events into admin-side and worker-side activity based on the manual classification described above. This was motivated by the observation that overall event volume does not differentiate converters from non-converters. The hypothesis was that it is specifically worker adoption, not admin setup, that should predict conversion in a product that serves an entire workforce.

Each organisation was classified into one of three archetypes based on which sides of the product were used during the trial: both admin and workers active, admin only with workers never joining, and minimal or no engagement.

Worker engagement depth was measured on a 0 to 5 scale by counting how many distinct types of worker activity occurred (punch clock, availability setting, shift swaps/handovers, absence requests, mobile schedule view).

The handoff gap was measured as the time between the first admin action and the first worker action for organisations where workers did eventually join.

### Layer 3: Predictive Modelling

Three models were trained on the org-level feature matrix to test whether any combination of behavioural signals could predict conversion: Logistic Regression, Random Forest, and Gradient Boosting. All three were evaluated using 5-fold cross-validated ROC-AUC. Features included total event counts, admin-specific event counts, worker-specific event counts, active days, time to first activity, binary activity flags, and the worker depth score. Standard scaling was applied before logistic regression.

Separate models were trained on admin features only, worker features only, and the full combined feature set to isolate the marginal predictive contribution of the admin/worker split.

Survival analysis (Kaplan-Meier and Cox Proportional Hazards) was applied to model time-to-conversion across different engagement groups. K-Means clustering (k=4) was used to segment organisations by behavioural profile and assess whether any natural cluster showed elevated conversion rates.

---

## Key Findings

### Finding 1: Converters and Non-Converters Are Statistically Indistinguishable

| Metric | Converters (median) | Non-Converters (median) | p-value |
|--------|--------------------|-----------------------|---------|
| Total events | 8 | 8 | 0.851 |
| Unique activities | 2 | 2 | 0.650 |
| Active days | 1 | 1 | 0.820 |
| Time to first activity | less than 10 minutes | less than 10 minutes | 0.153 |

No metric achieves statistical significance. The two groups are, by every behavioural measure available, the same. Three machine learning models trained to detect the difference between them all scored an AUC of approximately 0.48.

![Correlation Heatmap](charts/07_correlation_heatmap.png)

This is not a data quality problem. It is the data telling us, clearly, that in-app behaviour does not explain who converts.

### Finding 2: Conversion Is a Deadline Decision

52% of all conversions happen in the final 9 days of the trial. Nearly half of all conversions occur on Day 30 itself. Event volume spikes in the final week for both converters and non-converters alike, which means late-trial activity is not a signal of impending conversion — it is a signal of an approaching deadline.

![Trial Timing](charts/08_trial_timing.png)

![Retention and Activity Decay](charts/09_retention_decay.png)

This pattern is consistent with a procurement decision model, not a product value realisation model. Companies are not waking up on Day 8 because they experienced a eureka moment. They are waiting for the clock to run out and then making a cost-benefit decision.

### Finding 3: The Funnel Has One Critical Break Point

The product funnel reveals that the single largest drop-off is not at conversion — it is at the mobile schedule view stage, where 46% of organisations that created a shift never went on to view the schedule on mobile. This is the admin-to-worker handoff failing in plain sight.

![Funnel Analysis](charts/03_funnel_analysis.png)

### Finding 4: Feature Usage Tells a Consistent Story

Across all 28 activity types, no feature shows a statistically significant difference in adoption rate between converters and non-converters. The feature with the strongest positive association is `Scheduling.Shift.Created`, which is the core admin action and the most common activity in the dataset.

![Feature Usage](charts/06_feature_usage.png)

![Feature Uplift Detail](charts/06b_feature_uplift_detail.png)

The activation definition that produces the highest conversion uplift is simply "created at least one shift" (+4.0 percentage points), which is more a reflection of baseline engagement than a meaningful predictor.

![Activation Uplift](charts/04_activation_uplift.png)

### Finding 5: Time to Value Is Identical Across Groups

Converters and non-converters both engage within minutes of starting their trial. There is no early-mover advantage. Speed of initial engagement does not predict conversion.

![Time to Value](charts/05_time_to_value.png)

### Finding 6: Cohort Performance Is Consistent Over Time

Conversion rates across weekly cohorts show no meaningful trend. There is no evidence that earlier or later cohorts convert at different rates. The 21.3% rate is stable across the observed period.

![Cohort Analysis](charts/10_cohort_analysis.png)

---

## The Admin vs Worker Framework

The most important analytical decision in this project was to separate admin and worker activity rather than treating all events as equivalent. The rationale for this is grounded in how the product actually works.

When an admin creates shifts, approves timesheets, and sets up templates, they are performing configuration tasks. These are one-off or periodic actions. They tell us the admin understands the product and has used it for its intended purpose.

When workers clock in, view their schedule, set availability, or request swaps, they are performing operational tasks. These are recurring, daily-life actions. They tell us the product has been embedded into how the business actually functions.

Only the second category creates genuine switching costs. If the admin is the only user, the product can be cancelled with a single decision by one person. If workers are clocking in through it every day, cancellation means disrupting the live operations of the entire workforce. That is a fundamentally different retention dynamic.

### The Headline Worker Adoption Finding

![Worker Adoption Headline](charts/01_worker_adoption_headline.png)

51% of trialling organisations had the admin set up the product but no worker ever used it. The conversion rates between the two groups are virtually identical (21.7% vs 21.0%, p = 0.85). Worker adoption does not predict conversion.

### Worker Activity Impact on Conversion

![Worker Activity Uplift](charts/02_worker_activity_uplift.png)

No individual worker activity type achieves statistical significance as a conversion predictor. Punch clock usage shows the strongest positive association (+1.8 percentage points) but does not reach the 0.05 significance threshold.

### The Three Company Archetypes

![Three Archetypes](charts/03_three_archetypes.png)

| Archetype | N Orgs | Conv Rate | Risk Profile |
|-----------|--------|-----------|-------------|
| Both Admin + Workers Active | 371 (38%) | 21.0% | Low churn risk |
| Admin Only, Workers Never Joined | 489 (51%) | 22.0% | High churn risk |
| Minimal or No Engagement | 106 (11%) | 20.0% | Low intent |

The Incomplete Setup group is both the largest single group and the most commercially concerning. They convert at the average rate but carry the highest post-conversion retention risk.

### The Worker Adoption Funnel

![Worker Adoption Funnel](charts/04_worker_adoption_funnel.png)

### Worker Engagement Depth vs Conversion

![Worker Depth Conversion](charts/05_worker_depth_conversion.png)

There is a weak upward trend from depth score 2 onward, with organisations using 3 or more distinct worker activity types converting at 25.0%. However, numbers are too small at higher depth levels to draw firm conclusions.

### Speed of Worker Adoption

![Worker Adoption Speed](charts/06_worker_adoption_speed.png)

When workers do join, they join quickly. The median time from first admin action to first worker action is less than one hour. 82% of handoffs happen on the same day as admin setup. The problem is not timing — it is that 51% of companies never bridge the gap at all.

### Admin vs Worker Split by Conversion Outcome

![Admin vs Worker Split](charts/07_admin_vs_worker_split.png)

### The Handoff Gap

![Handoff Gap](charts/10_handoff_gap.png)

---

## Statistical Analysis

### Hypothesis Tests

**Mann-Whitney U (Converters vs Non-Converters)**

| Metric | U-statistic | p-value | Significant |
|--------|-------------|---------|------------|
| Total events | 78,944 | 0.851 | No |
| Unique activities | 79,822 | 0.650 | No |
| Unique activities | 79,822 | 0.650 | No |
| Active days | 78,982 | 0.820 | No |
| Time to first activity | 73,208 | 0.153 | No |

**Chi-square Tests (Activity Performed vs Conversion)**

No activity type achieves statistical significance at the 0.05 threshold. The activity with the strongest (non-significant) positive association is `Scheduling.Shift.Created` (chi-square = 0.78, p = 0.38). The activity with the most notable negative association is `Communication.Message.Created` (chi-square = 1.99, p = 0.16), suggesting that messaging-heavy organisations may be experiencing friction rather than engagement.

### Survival Analysis

![Kaplan-Meier](charts/11_kaplan_meier.png)

Kaplan-Meier curves stratified by engagement level (above vs below median event count) show a modestly faster conversion trajectory for high-engagement organisations, but both groups converge near Day 30. The Cox Proportional Hazards model identifies `total_events` and `unique_activities` as having the highest hazard coefficients, though confidence intervals are wide and overlap substantially.

![Cox PH Coefficients](charts/12_cox_ph_coefficients.png)

### Clustering

![Clustering](charts/16_clustering.png)

K-Means clustering with k=4 (selected by elbow method) produces four segments with conversion rates between 20% and 23%. The differences across clusters are small and consistent with random variation. No cluster represents a meaningfully higher-converting behavioural profile.

### Threshold Analysis

![Threshold Analysis](charts/17_threshold_analysis.png)

Organisations that create 100 or more shifts during their trial convert at 25.3% compared to 21% for lower-volume organisations. This is the closest thing to a meaningful threshold in the dataset, but it reflects power-user behaviour rather than a triggerable activation moment.

---

## Predictive Modelling

![Model Comparison Admin vs Worker](charts/08_model_comparison_worker_vs_admin.png)

| Model | Features | CV ROC-AUC |
|-------|----------|-----------|
| Logistic Regression | All features | 0.486 |
| Random Forest | All features | 0.480 |
| Gradient Boosting | All features | 0.477 |
| Random Forest | Admin features only | 0.515 |
| Random Forest | Worker features only | 0.514 |
| Random Forest | Admin + Worker combined | 0.520 |

![Logistic Regression Coefficients](charts/13_logistic_regression.png)

![Predictive Models Feature Importance](charts/14_predictive_models.png)

![Model Comparison](charts/15_model_comparison.png)

All models perform near random chance (0.50). Adding worker features to admin features produces a marginal improvement of 0.005 in AUC, which is not meaningful. The combined model does not outperform either subset in a practically significant way.

The top features by Random Forest importance are `total_events`, `time_to_first_activity_hrs`, `unique_activities`, and `active_days`. These are the same features that show no significant difference between converters and non-converters in the univariate tests. The model is learning the average behaviour of the dataset, not a conversion signal.

**Interpretation:** The absence of a predictive signal is itself the signal. It points directly to a data gap: the variables that actually drive conversion (company size, acquisition channel, pricing, sales touchpoints) are not in the dataset. Collecting these variables is more valuable than further modelling of existing behavioural data.

### Survival Analysis by Worker Adoption

![Survival Worker vs Admin](charts/09_survival_worker_vs_admin.png)

The Kaplan-Meier survival curves stratified by worker adoption status show that worker-active and admin-only organisations follow nearly identical conversion trajectories throughout the trial. Both groups spike sharply in conversions in the final 9 days, confirming that the deadline effect dominates for both segments.

---

## SQL Data Models

The SQL layer is built in a dbt-compatible staging and mart architecture. It exists to translate the analytical findings into operational models that can live in a data warehouse and drive ongoing business monitoring.

### Data Lineage

```
raw.da_task (source)
    └── stg_trial_events              (staging: cleaned, validated, enriched)
            ├── mart_trial_goals      (mart: per-org goal completion tracking)
            │       └── mart_trial_activation   (mart: activation status and tiers)
            └── mart_worker_adoption  (mart: admin vs worker segmentation)
```

### `stg_trial_events`

**File:** `sql/stg_trial_events.sql`

**Purpose:** This staging model is the single source of truth for cleaned event data. It exists to separate data quality logic from analytical logic. Any downstream mart can reference this model without needing to repeat deduplication, datetime parsing, or boundary validation.

**What it does:**
- Deduplicates exact duplicate rows using `ROW_NUMBER()` partitioned by all meaningful fields
- Parses and casts all datetime columns with explicit null handling for the `converted_at` field (which is null for non-converters)
- Removes events outside the trial window (before `trial_start` or after `trial_end`)
- Adds derived fields used by multiple downstream models: `trial_day_number`, `hours_since_trial_start`, and `days_to_conversion`

**Grain:** One row per organisation per event, deduplicated.

### `mart_trial_goals`

**File:** `sql/mart_trial_goals.sql`

**Purpose:** This mart translates the analytical findings into a trackable definition of what "good trial behaviour" looks like. The five goals were defined based on what the analysis identified as the most meaningful product interactions, grounded in product-value logic rather than pure statistical lift.

**The rationale for each goal:**

Goal 1 (Created at least one shift) is the single most important admin action. 87.8% of trialling organisations hit this, and converters do so at a slightly higher rate (89.8% vs 87.2%). It represents the moment the admin has moved from exploration to actual use. An organisation that never creates a shift has not used the product for its core purpose.

Goal 2 (Viewed the mobile schedule at least once) addresses the largest single friction point in the funnel. Only 47% of organisations that created shifts also opened the mobile schedule. This step represents the admin-to-worker handoff. Without it, the product has only ever been used by one person.

Goal 3 (Set workforce availability at least once) signals that employees are configuring their own preferences within the platform. This is a leading indicator that the product is moving from admin-only configuration to genuine workforce adoption.

Goal 4 (Active on 3 or more distinct days) captures sustained engagement rather than a single-session trial. Threshold analysis validated that 3 active days corresponds to meaningfully higher long-term conversion likelihood compared to 1 or 2 active days.

Goal 5 (Used 3 or more distinct activity types) captures breadth of platform exploration. Organisations that explore multiple features show a +1.4 percentage point conversion uplift and represent the strongest behavioural predictor in the logistic regression model.

**Grain:** One row per `organization_id`.

**Key fields:** `goal_1_created_shift` through `goal_5_explored_3plus_features`, `goals_completed_count`, and all supporting diagnostic signals.

### `mart_trial_activation`

**File:** `sql/mart_trial_activation.sql`

**Purpose:** This mart defines "Trial Activation" as the completion of all five goals and assigns each organisation to an activation tier. It is designed to be the operational layer that CS and product teams can query directly.

**Why a composite activation definition:** No single behaviour predicts conversion. But a composite definition covering the core scheduling action, the mobile handoff, workforce configuration, sustained use, and feature breadth captures the full range of "has this organisation experienced core product value?" That question, even if it does not directly predict trial conversion, is the correct measure of product health during a trial.

**Activation tiers:**
- Fully Activated: all 5 goals complete
- Partially Activated: 3 to 4 goals complete
- Early Engagement: 1 to 2 goals complete
- No Engagement: 0 goals complete

**Intervention flags built into the model:**
- `is_near_activated_not_converted`: 3+ goals complete but not yet converted. High-value CS target.
- `is_zero_engagement`: zero events or active days. Requires immediate re-engagement outreach.
- `is_activated_churned`: all 5 goals complete but did not convert. Requires post-mortem analysis.

**Grain:** One row per `organization_id`.

### `mart_worker_adoption`

**File:** `sql/mart_worker_adoption.sql`

**Purpose:** This mart operationalises the admin vs worker segmentation framework. It classifies every event in the staging layer as either admin-side or worker-side based on product logic, then aggregates to the organisation level to produce the three-archetype classification.

**Why this model exists as a separate mart:** The admin/worker distinction is not just an analytical concept. It should be a live operational metric. Product and CS teams need to know, in real time, which organisations have bridged the admin-to-worker gap and which have not. This model makes that visible in the warehouse without requiring analysts to re-run the classification logic each time.

**Intervention flags:**
- `flag_admin_only_not_converted`: admin activity present, zero worker activity, not converted. Primary target for the 48-hour worker onboarding nudge.
- `flag_no_punchclock_not_converted`: workers joined but never used the punch clock. Secondary target for feature education outreach.
- `flag_deep_worker_not_converted`: worker depth score of 3 or above but did not convert. Analytically interesting for further investigation.

**Grain:** One row per `organization_id`.

---

## Recommendations

### Recommendation 1: Capture the Missing Data (Priority: Immediate)

The single highest-leverage action available is to start collecting the data that this analysis proved matters. The current event log captures what users do inside the product. It captures nothing about why they decided to trial it, how large their company is, what price they were shown, or whether a salesperson spoke to them.

**Actions:**
- Add company size (number of employees) and industry to the signup flow
- Instrument UTM parameter capture at the signup URL to record acquisition source
- Integrate CRM data to log whether a sales or CS touchpoint occurred during the trial
- Record the pricing tier and plan shown to each trialling organisation

**Expected outcome:** Within two cohorts of trials with enriched data, conversion drivers will become identifiable. The current analysis has proven they exist outside the product. This makes them findable.

### Recommendation 2: Fix the Worker Onboarding Handoff (Priority: This Month)

51% of trialling organisations never got a worker onto the platform. When workers do join, they join within hours of admin setup. The problem is not slow adoption — it is the connection never being made. This is an operational gap that can be closed with a single automated email.

**The trigger:** Admin creates at least one shift AND no worker-side activity has been recorded within 48 hours.

**The message:** "Your schedule is live. Here is how to share it with your team." Include a direct link to the mobile app download and a one-click "share schedule" feature if not already present.

**Expected outcome:** Reduction in the admin-only rate from 51%. Improved post-conversion retention for companies that do activate their workforce. This does not directly drive trial conversion but protects long-term revenue per customer.

### Recommendation 3: Deploy CS Outreach at Days 14 to 21 (Priority: This Quarter)

52% of conversions happen in the final 9 days of the trial, driven by deadline urgency. The optimal window for sales and CS intervention is Days 14 to 21, before that urgency sets in. This gives time to address pricing questions, remove blockers, and move the decision earlier in the trial timeline.

**Targeting:** Flag all organisations at Day 14 that are active (have had at least one event in the last 7 days) but have not yet converted. Prioritise the Partially Activated segment (3 to 4 goals complete) as the highest-conversion-probability group within the non-converted population.

**Additional focus:** Organisations showing the 8 to 14 day engagement fade pattern currently convert at only 9.1%. These are companies that engaged meaningfully in week one and then went quiet. A targeted Day 10 alert for "was active, now silent" organisations would catch this group before they disengage permanently.

**Expected outcome:** Shift some Day 30 conversions earlier in the trial. Recover a portion of mid-trial faders. Reduce dependency on deadline urgency as the primary conversion mechanism.

---

## Limitations and Honest Caveats

**No demographic or firmographic data.** The dataset contains no information about company size, industry, geography, or team size. These variables almost certainly explain a significant share of conversion variance. The absence of a behavioural signal does not mean no signal exists — it means the signal is in data we do not have.

**No acquisition source data.** Whether a company arrived via a paid ad, an organic search, a referral, or a sales outreach is likely a strong predictor of intent and conversion likelihood. This is entirely absent from the current dataset.

**No pricing or plan information.** The conversion decision is fundamentally a value-for-money assessment. Without knowing what price each organisation was shown, we cannot model the most important variable in that assessment.

**No CRM or sales interaction data.** Whether a human spoke to a trialling organisation during the trial is likely one of the strongest conversion predictors available. It is not recorded in the current dataset.

**`Mobile.Schedule.Loaded` classification.** This activity was classified as a worker action because it represents an employee viewing their schedule on the mobile app. However, an admin could also open the mobile view to check how the schedule looks to their team. This inflates the worker activity count and means the true rate of worker-only adoption is likely lower than 49%.

**No user-level granularity.** The dataset is at the organisation level. We cannot distinguish between one worker using the app 50 times and 50 workers using it once each. The worker adoption classification is therefore a minimum bound on engagement, not a precise measure of workforce reach.

**No post-conversion data.** The analysis covers the trial period only. The hypothesis that worker adoption predicts post-conversion retention is analytically well-motivated but cannot be tested without renewal, expansion, and churn data from paying customers. Connecting this analysis to post-conversion outcomes is the most important next step.

**Small sample sizes at high worker depth.** Only 73 organisations reached a worker depth score of 3 or above. Conversion rates at these levels (25.0% at depth 3, 23.1% at depth 4) should be treated as indicative rather than conclusive.

---

## Setup and Usage

### Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0
lifelines>=0.27.0
statsmodels>=0.14.0
```


### Running the Analysis

Place the raw data file `DA_task.csv` in the root directory, then run code inside the:

```bash
untitled.ipynb
```

---

*Adediran Adeyemi · [www.adediranadeyemi.com](https://www.adediranadeyemi.com)*
