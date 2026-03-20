-- =============================================================================
-- MART LAYER: mart_worker_adoption
-- Description : Classifies each trialling organisation by whether workers
--               (employees) actually adopted the product after admin setup.
--
-- Key Finding: 50.6% of organisations had ZERO worker activity despite
-- the admin setting up shifts. Conversion rates across archetypes are
-- nearly identical (~21%), meaning worker adoption alone does not drive
-- conversion — but the absence of worker data masked this in prior analyses.
--
-- Three Archetypes:
--   1. Both Admin + Workers Active  — admin set up, workers engaged
--   2. Admin Only                   — admin set up, workers never joined
--   3. Minimal / No Engagement      — neither side used the product meaningfully
--
-- Grain: One row per organization_id
-- =============================================================================

WITH stg AS (
    SELECT * FROM {{ ref('stg_trial_events') }}
),

-- ── Classify each event as admin or worker ───────────────────────────────────
classified AS (

    SELECT
        *,
        CASE
            WHEN activity_name IN (
                'PunchClock.PunchedIn',
                'PunchClock.PunchedOut',
                'PunchClockStartNote.Add.Completed',
                'PunchClockEndNote.Add.Completed',
                'Break.Activate.Started',
                'Break.Activate.Finished',
                'Scheduling.Availability.Set',
                'Scheduling.ShiftSwap.Created',
                'Scheduling.ShiftHandover.Created',
                'Scheduling.OpenShiftRequest.Created',
                'Mobile.Schedule.Loaded',
                'Shift.View.Opened',
                'ShiftDetails.View.Opened',
                'Absence.Request.Created'
            ) THEN 'worker'
            WHEN activity_name IN (
                'Scheduling.Shift.Created',
                'Scheduling.Shift.AssignmentChanged',
                'Scheduling.Template.ApplyModal.Applied',
                'Scheduling.Shift.Approved',
                'Scheduling.ShiftSwap.Accepted',
                'Scheduling.ShiftHandover.Accepted',
                'Scheduling.OpenShiftRequest.Approved',
                'Absence.Request.Approved',
                'Absence.Request.Rejected',
                'Timesheets.BulkApprove.Confirmed',
                'Integration.Xero.PayrollExport.Synced',
                'Revenue.Budgets.Created',
                'Communication.Message.Created',
                'PunchClock.Entry.Edited'
            ) THEN 'admin'
            ELSE 'shared'
        END AS activity_actor_type

    FROM stg

),

-- ── Aggregate to org level ───────────────────────────────────────────────────
org_agg AS (

    SELECT
        organization_id,

        -- Meta
        MAX(converted)          AS converted,
        MAX(converted_at)       AS converted_at,
        MAX(trial_start)        AS trial_start,
        MAX(trial_end)          AS trial_end,
        MAX(days_to_conversion) AS days_to_conversion,

        -- Volume by actor type
        COUNT(*)                                                    AS total_events,
        SUM(CASE WHEN activity_actor_type = 'admin'  THEN 1 ELSE 0 END) AS admin_events,
        SUM(CASE WHEN activity_actor_type = 'worker' THEN 1 ELSE 0 END) AS worker_events,

        -- Unique activities by type
        COUNT(DISTINCT CASE WHEN activity_actor_type = 'admin'
              THEN activity_name END)                               AS admin_unique_activities,
        COUNT(DISTINCT CASE WHEN activity_actor_type = 'worker'
              THEN activity_name END)                               AS worker_unique_activities,

        -- Active days by type
        COUNT(DISTINCT CASE WHEN activity_actor_type = 'admin'
              THEN DATE(event_timestamp) END)                       AS admin_active_days,
        COUNT(DISTINCT CASE WHEN activity_actor_type = 'worker'
              THEN DATE(event_timestamp) END)                       AS worker_active_days,

        -- Worker activity flags
        MAX(CASE WHEN activity_name IN ('PunchClock.PunchedIn','PunchClock.PunchedOut')
                 THEN 1 ELSE 0 END)                                 AS has_punchclock,
        MAX(CASE WHEN activity_name = 'Scheduling.Availability.Set'
                 THEN 1 ELSE 0 END)                                 AS has_availability_set,
        MAX(CASE WHEN activity_name IN ('Scheduling.ShiftSwap.Created',
                                        'Scheduling.ShiftHandover.Created')
                 THEN 1 ELSE 0 END)                                 AS has_shift_swap_or_handover,
        MAX(CASE WHEN activity_name = 'Absence.Request.Created'
                 THEN 1 ELSE 0 END)                                 AS has_absence_request,
        MAX(CASE WHEN activity_name = 'Mobile.Schedule.Loaded'
                 THEN 1 ELSE 0 END)                                 AS has_mobile_schedule_view,

        -- First timestamps
        MIN(CASE WHEN activity_actor_type = 'admin'
                 THEN event_timestamp END)                          AS first_admin_event_ts,
        MIN(CASE WHEN activity_actor_type = 'worker'
                 THEN event_timestamp END)                          AS first_worker_event_ts

    FROM classified
    GROUP BY 1

),

-- ── Derived fields and archetype classification ───────────────────────────────
final AS (

    SELECT
        organization_id,
        converted,
        converted_at,
        days_to_conversion,
        trial_start,
        trial_end,
        total_events,
        admin_events,
        worker_events,
        admin_unique_activities,
        worker_unique_activities,
        admin_active_days,
        worker_active_days,
        has_punchclock,
        has_availability_set,
        has_shift_swap_or_handover,
        has_absence_request,
        has_mobile_schedule_view,
        first_admin_event_ts,
        first_worker_event_ts,

        -- Worker adoption flag
        CAST(worker_events > 0 AS BOOLEAN)                         AS has_any_worker_activity,

        -- Worker depth score (0–5): how many distinct worker activity types occurred
        (
            CAST(has_punchclock            AS INTEGER) +
            CAST(has_availability_set      AS INTEGER) +
            CAST(has_shift_swap_or_handover AS INTEGER) +
            CAST(has_absence_request       AS INTEGER) +
            CAST(has_mobile_schedule_view  AS INTEGER)
        )                                                           AS worker_depth_score,

        -- Worker event share
        ROUND(
            worker_events * 1.0 / NULLIF(total_events, 0), 3
        )                                                           AS worker_event_share,

        -- Handoff gap: time from first admin action to first worker action (hours)
        CASE
            WHEN first_worker_event_ts IS NOT NULL
             AND first_admin_event_ts  IS NOT NULL
            THEN DATEDIFF('hour', first_admin_event_ts, first_worker_event_ts)
            ELSE NULL
        END                                                         AS handoff_gap_hours,

        -- ── ARCHETYPE ──────────────────────────────────────────────────────────
        CASE
            WHEN worker_events > 0 AND admin_events > 0
                THEN 'Both Admin + Workers Active'
            WHEN admin_events  > 0 AND worker_events = 0
                THEN 'Admin Only (Workers Never Joined)'
            ELSE
                'Minimal / No Engagement'
        END                                                         AS adoption_archetype,

        -- ── INTERVENTION FLAGS ────────────────────────────────────────────────
        -- Admin set up but no workers — prime target for mobile onboarding nudge
        CAST(
            admin_events > 0
            AND worker_events = 0
            AND NOT converted
        AS BOOLEAN)                                                 AS flag_admin_only_not_converted,

        -- Workers joined but didn't use punch clock — missing a key workflow
        CAST(
            worker_events > 0
            AND has_punchclock = 0
            AND NOT converted
        AS BOOLEAN)                                                 AS flag_no_punchclock_not_converted,

        -- Deep worker engagement but didn't convert — investigate friction
        CAST(
            worker_depth_score >= 3
            AND NOT converted
        AS BOOLEAN)                                                 AS flag_deep_worker_not_converted

    FROM org_agg

)

SELECT * FROM final
ORDER BY
    CASE adoption_archetype
        WHEN 'Both Admin + Workers Active'         THEN 1
        WHEN 'Admin Only (Workers Never Joined)'   THEN 2
        ELSE 3
    END,
    worker_depth_score DESC,
    organization_id
