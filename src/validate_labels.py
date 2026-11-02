"""Validate BharatLens's multi-label Telugu news CSV without third-party packages."""

import csv
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'telugu_news_labeled.csv'
LABELS = ('environment', 'culture', 'education', 'agriculture', 'water')
REQUIRED = ('article_id', 'headline', 'source', 'source_url', 'rights_status', *LABELS)


def validate_csv(path=DATA_PATH):
    errors = []
    warnings = []
    counts = Counter()
    seen_ids = set()
    seen_headlines = set()
    valid_rows = 0
    all_rows = 0

    with open(path, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError('CSV has no header')
        missing_cols = [field for field in REQUIRED if field not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f'Missing columns: {missing_cols}')

        for row_number, row in enumerate(reader, start=2):
            all_rows += 1
            row_errors = []
            for field in ('article_id', 'headline', 'source', 'rights_status'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    row_errors.append(f'{field} must be non-empty')
            if any(value is None for value in row.values()):
                row_errors.append('row has fewer values than header')
            if None in row:
                row_errors.append('row has more values than header')

            article_id = (row.get('article_id') or '').strip()
            headline = (row.get('headline') or '').strip()
            if article_id:
                if article_id in seen_ids:
                    row_errors.append(f'duplicate article_id: {article_id}')
                seen_ids.add(article_id)
            if headline:
                if headline in seen_headlines:
                    warnings.append(f'Row {row_number}: duplicate headline: {headline}')
                seen_headlines.add(headline)

            for label in LABELS:
                if row[label] not in ('0', '1'):
                    row_errors.append(f'{label} must be 0 or 1 (got {row[label]!r})')

            if row.get('source') == 'synthetic_original' and (row.get('rights_status') != 'original'):
                warnings.append(f'Row {row_number}: review synthetic source rights status')
            if row.get('source') != 'synthetic_original' and not (row.get('source_url') or '').strip():
                warnings.append(f'Row {row_number}: non-synthetic article lacks source URL')

            if row_errors:
                errors.extend(f'Row {row_number}: {error}' for error in row_errors)
            else:
                valid_rows += 1
                positives = [label for label in LABELS if row[label] == '1']
                counts.update(positives)
                if not positives:
                    counts['other'] += 1
                if len(positives) > 1:
                    counts['multi_label'] += 1

    print(f'Rows: {all_rows} | Valid: {valid_rows} | Invalid: {all_rows - valid_rows}')
    for label in (*LABELS, 'other', 'multi_label'):
        print(f'  {label}: {counts[label]}')
    for label in LABELS:
        if counts[label] == 0:
            warnings.append(f'No valid positive examples for {label}')
    for warning in warnings:
        print(f'WARNING: {warning}')
    for error in errors:
        print(f'ERROR: {error}')
    return not errors


if __name__ == '__main__':
    raise SystemExit(0 if validate_csv() else 1)
