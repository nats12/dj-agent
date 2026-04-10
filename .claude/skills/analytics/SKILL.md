---
name: analytics
description: Library analytics — genre/BPM/key distributions, coverage gaps, metadata completeness scoring.
---

# Analytics

Deep analysis of your library's composition — find gaps, imbalances, and metadata issues.

## What It Reports

- **Genre distribution** — top genres with bar chart
- **BPM distribution** — histogram by 5-BPM buckets, highlights gaps
- **Key coverage** — how many of the 24 Camelot positions you cover
- **Metadata completeness** — percentage score + breakdown of what's missing
- **BPM gaps** — tempo ranges with zero tracks (e.g., no 115-120 BPM tracks)

## Usage

```python
from dj_agent.analytics import analyse_library, format_analytics
report = analyse_library(tracks)
print(format_analytics(report))
```

## Workflow

1. `analytics` or `analyse library` — generate full report
2. Identify gaps (e.g., "I have no tracks at 135-140 BPM in minor keys")
3. Use insights to guide track acquisition
