"""Offline federal five-day calendar, versioned by publication date.

Unknown annual transfers are NOT invented: statutory holidays/weekends remain
available and transfer_known=0 explicitly marks the incomplete calendar.
Anonymous warehouses are not mapped to regions.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd

CALENDAR_VERSION = "ru-federal-2024-2026-v1"
SOURCES = {
    "holidays": "https://mintrud.gov.ru/labour/relationship/351",
    "2024": "https://pravo.gov.ru/novye-postupleniya/index.cgi?from=14.08.2023",
    "2025": "https://publication.pravo.gov.ru/document/0001202410040023",
    "2025_document": "https://storage.consultant.ru/ondb/attachments/202410/04/iddoc_286124_idnews_56549_Vyhodnye_2025_sVh.pdf",
    "2026": "https://static.government.ru/media/files/4jeB8hNKm69ggOa9yDiYOli6YoAyM21i.pdf",
}
TRANSFERS = {
    2024: ("2023-08-18", [("01-06", "05-10"), ("01-07", "12-31"), ("04-27", "04-29"), ("11-02", "04-30"), ("12-28", "12-30")]),
    2025: ("2024-10-04", [("01-04", "05-02"), ("01-05", "12-31"), ("02-23", "05-08"), ("03-08", "06-13"), ("11-01", "11-03")]),
    2026: ("2025-09-24", [("01-03", "01-09"), ("01-04", "12-31")]),
}
FIXED = [(1,d) for d in range(1,9)] + [(2,23),(3,8),(5,1),(5,9),(6,12),(11,4)]


@lru_cache(maxsize=256)
def year_calendar(year, as_of):
    year, as_of = int(year), pd.Timestamp(as_of)
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    frame = pd.DataFrame(index=dates)
    holiday = {pd.Timestamp(year=year,month=m,day=d) for m,d in FIXED}
    published, pairs = TRANSFERS.get(year, ("2262-01-01", []))
    known = pd.Timestamp(published) <= as_of
    pairs = pairs if known else []
    explicit_sources = {pd.Timestamp(f"{year}-{a}") for a,b in pairs}
    off = set(holiday)
    # Statutory automatic observance, except January and explicitly moved days.
    for h in sorted(holiday):
        if h.month != 1 and h.weekday() >= 5 and h not in explicit_sources:
            nxt = h + pd.Timedelta(days=1)
            while nxt.weekday() >= 5 or nxt in off:
                nxt += pd.Timedelta(days=1)
            off.add(nxt)
    working_override = set()
    transferred = set()
    for src,dst in pairs:
        src,dst = pd.Timestamp(f"{year}-{src}"),pd.Timestamp(f"{year}-{dst}")
        if src not in holiday:
            working_override.add(src)
        off.add(dst)
        transferred.add(dst)
    frame["holiday"] = frame.index.isin(holiday).astype(float)
    frame["working"] = ((frame.index.weekday < 5) & ~frame.index.isin(off)) | frame.index.isin(working_override)
    frame["working"] = frame.working.astype(float)
    frame["working_saturday"] = ((frame.index.weekday >= 5) & (frame.working > 0)).astype(float)
    frame["transferred"] = frame.index.isin(transferred).astype(float)
    frame["transfer_known"] = float(known)
    return frame


def calendar_days(dates, as_of):
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    years = sorted(set(idx.year))
    source = pd.concat([year_calendar(y,str(pd.Timestamp(as_of).date())) for y in years])
    return source.reindex(idx).set_axis(idx)


def weekly_calendar(weeks, as_of):
    rows=[]
    for week in pd.DatetimeIndex(weeks).unique():
        around = pd.date_range(week-pd.Timedelta(days=21),week+pd.Timedelta(days=27))
        cal=calendar_days(around,as_of)
        days=cal.loc[week:week+pd.Timedelta(days=6)]
        off=(cal.working.to_numpy()==0)
        longest=current=0
        for i,value in enumerate(off):
            current=current+1 if value else 0
            # Include only blocks touching the target week.
            if value and i>=21 and i-current<28:longest=max(longest,current)
        holidays=cal.index[cal.holiday>0]
        before=[(week-h).days for h in holidays if h<week]
        after=[(h-week).days for h in holidays if h>=week]
        rows.append(dict(week=week,ru_workdays=days.working.sum(),ru_holidays=days.holiday.sum(),
                         ru_working_saturdays=days.working_saturday.sum(),ru_transferred=days.transferred.sum(),
                         ru_transfer_known=days.transfer_known.min(),ru_long_break=min(longest,21),
                         ru_days_after_holiday=min(before,default=28),ru_days_before_holiday=min(after,default=28),
                         year_sin=np.sin(2*np.pi*week.dayofyear/365.2425),
                         year_cos=np.cos(2*np.pi*week.dayofyear/365.2425),
                         month_cross=int(week.month!=(week+pd.Timedelta(days=6)).month)))
    return pd.DataFrame(rows).set_index("week")
