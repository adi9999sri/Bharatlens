# BharatLens — Initial dataset report (v0.1)

## Overview
- **Rows:** 100
- **Language:** Telugu (`te`)
- **Unit:** One synthetic headline per row
- **Origin:** Originally written fictional practice examples; no publisher full text copied
- **Usage:** Development/testing of CSV ingestion, schema validation, and annotation workflow
- **Labeling:** Rule-guided human-authored labels, not independently verified
- **License note:** These original examples are development assets; review reuse terms before importing third-party sources

## Label distribution

| Category | Positive examples |
|---|---:|
| environment | 20 |
| culture | 18 |
| education | 19 |
| agriculture | 20 |
| water | 22 |

- **Other:** 15 rows with all five labels zero
- **Overlapping rows:** 10 intentionally multi-label rows

## Key limitations
1. Small, synthetic, and deliberately balanced; real Telugu news has different vocabulary, writing style, and category frequencies.
2. No verified inter-annotator agreement or independently labelled test set.
3. Headline-only, no long-form body content; cannot validate full-text classifier performance.
4. Sources, locations and publication dates are not real news metadata.
5. Do not report accuracy on a random split of these templates as evidence of production readiness.

## Next steps
- Source permission-compliant real news material and record provenance.
- Add metadata: date, publication, location, permitted content reference.
- Build a manually reviewed, independently labelled evaluation set.
- Split by story/time/source to limit leakage before benchmarking ML, transformers, and LLMs.
