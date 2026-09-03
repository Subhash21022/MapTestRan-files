# Fetching the CMRL data yourself

Everything here runs from a plain terminal — VS Code, Antigravity, PowerShell,
Windows Terminal. Nothing depends on Claude Code.

---

## TL;DR

Open a terminal in `D:\mapathon` and run this **once a day**:

```bash
python -m src.cmrl_stationflow
python -m src.cmrl_extras
```

That is the whole job. Everything below is detail.

---

## Why once a day, and not just once

CMRL's dashboard is backed by this API:

```
https://commuters-dataapi.chennaimetrorail.org/api/PassengerFlow/stationData/{n}
```

`{n}` looks like "days back", but **it is not**. It has only two states:

| `{n}` | Returns |
|---|---|
| `0` | today, **partial** (only hours elapsed so far) |
| `1` and above | the **same** single previous day — `1`, `2`, `30`, `365` are byte-identical |

There is no history. A first attempt looped `1..28` and collected 28 identical
copies of one day; the giveaway was every date summing to exactly 192,613.

So the only way to build a usable sample is to collect once a day and let it
accumulate. The scripts append to a dated store and de-duplicate, so running
twice in a day is harmless and missing a day only costs that day.

---

## One-time setup (new machine only)

```bash
cd D:\mapathon
python -m pip install -r requirements.txt
python -m src.check_env
```

`check_env` should end with **"Environment is ready."** If a package fails,
re-run the install line.

---

## The daily commands

### 1. Station-wise boardings — the modelling target

```bash
python -m src.cmrl_stationflow
```

Expect output like:

```
offset 0 (today, partial): 2026-08-18 = 9,566 boardings
offset 1 (previous complete day): 2026-08-17 = 376,881 boardings
store now holds 2 complete day(s)
matched 40/41 CMRL stations to the canonical list
```

Writes:
- `data/raw/ridership/chennai/cmrl_stationflow_daily.csv` — the growing store
- `data/processed/chennai_station_target.csv` — the model's target

### 2. Hourly profile and ticket mix

```bash
python -m src.cmrl_extras
```

Writes `cmrl_hourly.csv` and `cmrl_ticket_mix.csv` into the same folder.
These are system-wide, not per-station, so they cannot train the model — they
give peak-hour share (11.9% at 08:00) and the fare-media story for the map.

---

## Checking it worked

```bash
python -c "import pandas as pd; d=pd.read_csv('data/raw/ridership/chennai/cmrl_stationflow_daily.csv'); print(d.groupby(['date','is_partial'])['boardings'].sum().to_string())"
```

Two things to look for:

- **Weekday totals around 370-390k**, weekends around 190-200k. A weekday
  reading near 190k means a weekend got mislabelled.
- **Every date different.** Identical totals across dates means the store has
  duplicated one day.

Count usable weekdays (weekends do not count toward the target):

```bash
python -c "import pandas as pd; d=pd.read_csv('data/raw/ridership/chennai/cmrl_stationflow_daily.csv'); d=d[~d.is_partial]; d['dow']=pd.to_datetime(d.date).dt.dayofweek; print('weekdays collected:', d[d.dow<5].date.nunique())"
```

| Weekdays | What it supports |
|---|---|
| 1-4 | ranking only — keep training on Bengaluru |
| 5-9 | usable, noisy per-station means |
| **10+** | train on Chennai directly, drop the cross-city transfer |

---

## The automatic option (already set up on this machine)

A Windows scheduled task runs both scripts daily at 09:00, so you do not have
to remember.

```powershell
# is it healthy?  LastTaskResult 0 = success
Get-ScheduledTaskInfo -TaskName "MapathonCMRLFetch" | Select LastRunTime,LastTaskResult,NextRunTime

# run it now
Start-ScheduledTask -TaskName "MapathonCMRLFetch"

# what happened
Get-Content D:\mapathon\logs\cmrl_fetch.log -Tail 30
```

It is configured to catch up a run missed because the machine was off.

---

## Standalone: fetching without this repo

If you are in a different environment and just want the raw numbers, this needs
only `requests`:

```python
import requests, json

API = "https://commuters-dataapi.chennaimetrorail.org/api/PassengerFlow"
HEAD = {
    "User-Agent": "student-research",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://commuters-data.chennaimetrorail.org/passengerflow",
}

# 0 = today (partial), 1 = previous complete day. Nothing else exists.
data = requests.get(f"{API}/stationData/1", headers=HEAD, timeout=60).json()

for block in data:
    total = next(s for s in block["series"] if s["name"] == "Total")
    print(f"line {block['line']}: {sum(total['data']):,} boardings")
    print(total["data"])
```

**The response carries no station names** — only values, in platform order
along each line. The name lists live in `LINE1_STATIONS` and `LINE2_STATIONS`
in `src/cmrl_stationflow.py`, read off the dashboard's chart configuration.
If CMRL opens a new station those lists must be updated, or every value shifts
onto the wrong station. `fetch_day()` raises rather than guessing when the
lengths stop matching.

The other two endpoints are `hourlybaseddata/{n}` and `allTicketCount/{n}`,
same offset rules.

---

## Troubleshooting

**"no data returned by the API"** — CMRL's server is down or the shape changed.
Open <https://commuters-data.chennaimetrorail.org/passengerflow> in a browser;
if the dashboard is broken too, wait.

**"line 01 returned N values but 26 station names are configured"** — a station
opened or closed. Open the dashboard, read the station order off the
"Station Wise Passenger Flow" charts, update the lists in
`src/cmrl_stationflow.py`. This is deliberately a hard error: silently
accepting it would misattribute every station's ridership.

**Totals look doubled** — the store has the same day twice. Delete
`cmrl_stationflow_daily.csv` and re-run; you lose accumulated history, so check
first with the verification command above.

**`ModuleNotFoundError: No module named 'src'`** — you are not in the project
root. `cd D:\mapathon` first.
