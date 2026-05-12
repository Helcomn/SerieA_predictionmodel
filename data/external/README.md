# External Data

Place optional external football datasets here.

## Understat xG

Save the Understat/Kaggle match or game stats CSV as:

```text
data/external/understat_matches.csv
```

Or refresh it directly from Understat:

```bash
python src/update_understat.py
```

Supported schemas:

- Match-level rows with columns like `date`, `league`, `team_h`, `team_a`, `h_xg`, `a_xg`
- Team-per-match rows with columns like `date`, `league`, `club_name`, `home_away`, `xG`

CSV and Parquet files in this directory are ignored by Git.
