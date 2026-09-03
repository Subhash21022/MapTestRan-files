# Scheduled data collection

## Why this exists

CMRL publishes station-wise ridership through the API behind
<https://commuters-data.chennaimetrorail.org/passengerflow>:

```
GET https://commuters-dataapi.chennaimetrorail.org/api/PassengerFlow/stationData/{n}
```

`{n}` looks like a day offset but **is not one**. Verified by hashing the
responses: `n=0` returns the current (partial) day, and every value from `1` to
`365` returns the *same* single previous day. There is no history.

A first attempt looped `1..28` and collected 28 identical copies of one day -
detectable only because every date summed to exactly 192,613.

So the only way to build a usable weekday sample is to collect once a day and
accumulate. `src/cmrl_stationflow.py` appends to a dated store and
de-duplicates on `(date, line, station)`, which makes re-runs harmless.

## The scheduled task

| | |
|---|---|
| Name | `MapathonCMRLFetch` |
| Runs | daily, 09:00 |
| Script | `scripts/fetch_daily.ps1` |
| Log | `logs/cmrl_fetch.log` |
| Store | `data/raw/ridership/chennai/cmrl_stationflow_daily.csv` |
| Output | `data/processed/chennai_station_target.csv` |

Configured with `-StartWhenAvailable`, so a run missed because the machine was
off happens at the next opportunity. It runs as the logged-on user, because the
Python interpreter here is a Microsoft Store execution alias that does not
resolve reliably in a non-interactive session.

### Check on it

```powershell
Get-ScheduledTaskInfo -TaskName "MapathonCMRLFetch" | Select LastRunTime,LastTaskResult,NextRunTime
```

`LastTaskResult` of `0` means success. Log tail:

```powershell
Get-Content D:\mapathon\logs\cmrl_fetch.log -Tail 30
```

### Run it now

```powershell
Start-ScheduledTask -TaskName "MapathonCMRLFetch"
```

### Remove it

```powershell
Unregister-ScheduledTask -TaskName "MapathonCMRLFetch" -Confirm:$false
```

## When is there enough data?

The target is **mean weekday boardings**, so weekend days do not count toward
the sample. Rough guide:

| Complete weekdays | What it supports |
|---|---|
| 1–4 | Relative ranking only. Keep training on Bengaluru |
| 5–9 | Usable; expect noisy per-station means |
| 10+ | Train on Chennai directly and drop the cross-city transfer |

Ten weekdays takes about two calendar weeks from a standing start.

Note the first day collected (16 Aug 2026) is a **Sunday following Independence
Day** - unrepresentative even for a weekend, and excluded from the weekday
target automatically.
