# Manuscript Alignment Status and Remaining Steps

## Status

The authoritative manuscript has been reconciled with the corrected formal experiment. The editable DOCX, matching PDF, and acceptance record are ready for Git review. No commit or push has been performed.

Completed work:

- preserved the authoritative source manuscript unchanged;
- aligned the final data scope, model-specific targets, tuning procedures, benchmark-first tests, corporate-action policy, and application architecture;
- inserted the audited Chapter IV results and report-ready tables;
- replaced the affected methodology and system diagrams;
- retained the earlier June 30, 2026 company-selection statistics only as clearly labeled preliminary evidence;
- removed active Streamlit and browser-upload claims;
- rendered and visually reviewed all 359 PDF pages; and
- rechecked all 155 protected experiment artifact hashes with zero mismatches.

## Remaining steps

1. Review the repository change list and confirm that only the intended manuscript package and supporting reports will be committed.
2. Keep `formal_evidence/` out of the Git commit unless the group explicitly approves storing the full evidence bundle in Git or Git LFS.
3. Stage the selected files under `reports/FORMAL_CORRECTED_20260828_02/`.
4. Inspect the staged diff and staged file sizes.
5. Create one documentation commit with the reviewed files.
6. Push the documentation branch to GitHub and verify the remote commit hash.

Adviser review remains deferred and is not represented as completed approval.
