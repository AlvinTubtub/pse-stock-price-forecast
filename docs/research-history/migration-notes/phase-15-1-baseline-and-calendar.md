# Phase 15.1 preservation and calendar review

## Frontend baseline

The preservation baseline is the frontend behavior captured in the Phase 1
contract, commit `f2e41b0`, not simply `origin/main`. That contract predates reset
commit `17bcdfb` and explicitly documents the formal-study file, Models page,
chatbot consumers, fixed study prose, and separate operational outputs (sections
1, 3, 6 and 15). The reset commit first records six application-file changes,
but commit timing alone cannot establish when uncommitted work was authored.
The earlier contract is evidence that the research presentation was already
intended at discovery, not a new backend implementation requirement.

Preserve all six files: compare/page.tsx (documented research/operational view),
learn-stocks/page.tsx (research-separation explanations), CompanyDetailView.tsx
(operational versus research labeling), lib/ai/context.ts (documented chatbot
consumer), lib/data.ts (documented static research loader), and lib/types.ts
(types supporting that contract). No accidental reset-only change has been
established that warrants reverting legitimate UI work. This is a behavioral
baseline determination, not proof of an earlier committed byte-identical tree.

The 243-date claims belong to the explicitly frozen research section; they are
not the rebuilt backend's dynamic evaluation size. The static research asset
must not become modeling input, generated evaluation evidence, or an exporter
source. It remains outside the operational artifact lifecycle. No frontend
application files or research JSON were changed in Phase 15.1.

## Calendar maintenance

`backend/config/pse_holidays.py` contains reviewable ISO dates and source links
for the 2026 PSE/SCCP closure baseline and separately announced Eid closures.
Review PSE/SCCP notices for emergency changes and add the next year's confirmed
dates before year-end. The configuration is not an evergreen calendar and
does not infer closures from absent raw observations or scrape during inference.

`PSETradingCalendar()` uses configured closures. `with_holidays()` unions extra
closures with that baseline. Both training and daily CLI entry points already
use this constructor, so no workflow flags are required. `--holiday` remains an
additive emergency override. The resulting calendar is passed through
orchestration and next-day inference; the frontend exporter uses the resulting
forecast date rather than independently computing another calendar.

No model retuning, raw-data changes, forecast regeneration, or live ingestion
are needed for these changes. Existing September 14 forecasts remain unchanged.
