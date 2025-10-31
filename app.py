from fastapi import FastAPI, HTTPException, Header, Query
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
import os
import requests
import pytz
import math

# ========== ENV CONFIG ==========
GRAPH_VERSION = "v20.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"

FB_ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN", "")
DEFAULT_ACCOUNT_ID = os.getenv("FB_AD_ACCOUNT_ID", "")
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "America/Chicago")
EXPECTED_API_KEY = os.getenv("API_KEY", "")  # optional protection

# ========== FASTAPI APP ==========
app = FastAPI(
    title="Meta Ads Pacing API",
    description="Fetches Meta budgets+spend, computes daily/flight pacing at ad set or campaign level, optional 7-day series and filters.",
    version="1.8.0",
)

# ---------- helpers ----------
def require_api_key(header_key: Optional[str], query_key: Optional[str]):
    if not EXPECTED_API_KEY:
        return
    supplied = header_key or query_key
    if supplied != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def account_tz(tz_name: str) -> pytz.BaseTzInfo:
    try:
        return pytz.timezone(tz_name)
    except Exception:
        return pytz.timezone("UTC")

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def classify_ratio(r: Optional[float], lower: float = 0.9, upper: float = 1.1) -> str:
    if r is None or r != r:
        return "n/a"
    if r < lower:
        return "behind"
    if r > upper:
        return "ahead"
    return "on track"

def to_float(s: Any) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0

def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

def fb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params or {})
    params["access_token"] = FB_ACCESS_TOKEN
    # Support absolute "next" URLs
    url = path if path.startswith("https://") else f"{GRAPH_BASE}{path}"
    r = requests.get(url, params=params, timeout=60)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ===== Entity fetchers =====

def get_adsets(account_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Paginated fetch of ad sets with key budget/date fields.
    Respects 'limit' but pulls multiple pages if needed.
    """
    fields = ",".join([
        "id","name","effective_status","configured_status",
        "start_time","end_time",
        "daily_budget","lifetime_budget","budget_remaining",
        "campaign{id,name,spend_cap}","account_id"
    ])
    out: List[Dict[str, Any]] = []
    params = {"fields": fields, "limit": min(limit, 200)}
    path = f"/{account_id}/adsets"

    while True:
        data = fb_get(path, params)
        out.extend(data.get("data", []))
        if len(out) >= limit:
            break
        paging = data.get("paging", {})
        next_url = paging.get("next")
        if not next_url:
            break
        path = next_url
        params = {}

    return out[:limit]


def get_campaigns(account_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Paginated fetch of campaigns with budget/date fields (for CBO)."""
    fields = ",".join([
        "id","name","effective_status","configured_status",
        "start_time","stop_time",
        "daily_budget","lifetime_budget","spend_cap","pacing_type","account_id"
    ])
    out: List[Dict[str, Any]] = []
    params = {"fields": fields, "limit": min(limit, 200)}
    path = f"/{account_id}/campaigns"

    while True:
        data = fb_get(path, params)
        out.extend(data.get("data", []))
        if len(out) >= limit:
            break
        paging = data.get("paging", {})
        next_url = paging.get("next")
        if not next_url:
            break
        path = next_url
        params = {}

    return out[:limit]


# ===== Insights helpers =====

def get_insights_single(entity_id: str, **kw) -> Dict[str, Any]:
    params = {"fields": "spend"}
    params.update(kw)
    data = fb_get(f"/{entity_id}/insights", params)
    arr = data.get("data", [])
    return arr[0] if arr else {}


def get_insights_series(entity_id: str, since: str, until: str) -> List[Dict[str, Any]]:
    params = {
        "fields": "spend",
        "time_range[since]": since,
        "time_range[until]": until,
        "time_increment": 1
    }
    data = fb_get(f"/{entity_id}/insights", params)
    return data.get("data", [])


# ===== Pacing math =====

def compute_daily_expected_so_far(daily_budget: float, now: datetime, tz_name: str) -> float:
    tz = account_tz(tz_name)
    now_tz = now.astimezone(tz)
    start_of_day = now_tz.replace(hour=0, minute=0, second=0, microsecond=0)
    hours_elapsed = max(0.0, (now_tz - start_of_day).total_seconds() / 3600.0)
    return daily_budget * min(1.0, hours_elapsed / 24.0)


def compute_flight_expected(lifetime_budget: float, start: Optional[datetime], end: Optional[datetime], now: datetime) -> Optional[float]:
    if not lifetime_budget or not start or not end or end <= start:
        return None
    total = (end - start).total_seconds()
    elapsed = (now - start).total_seconds()
    ratio = clamp01(elapsed / total)
    return lifetime_budget * ratio


def parse_fb_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return dateparser.parse(s).astimezone(timezone.utc)


def worst_key(item: Dict[str, Any], basis: str, overspend_first: bool) -> Tuple[float, int]:
    """
    Sorting helper.
    - basis: 'flight' or 'today'
    - overspend_first:
        False -> "most behind" first (lowest ratios first), None at end
        True  -> "most overspend" first (highest ratios first), None at end
    """
    ratio = item.get("pacing_flight_ratio") if basis == "flight" else item.get("pacing_today_ratio")
    if ratio is None or ratio != ratio:
        return (float("inf"), 1) if not overspend_first else (float("-inf"), 1)
    return (ratio, 0) if not overspend_first else (-ratio, 0)


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/pacing")
def pacing(
    account_id: Optional[str] = None,
    level: str = "adset",  # now supports: adset, campaign
    limit: int = 25,
    tz: Optional[str] = None,
    include_paused: bool = False,
    min_daily_budget: float = 1.0,
    include_series_7d: bool = False,
    sort_by: str = "flight",
    overspend_first: bool = False,
    campaign_filter: Optional[str] = None,   # substring on campaign name
    campaign_ids: Optional[str] = None,      # comma-separated campaign IDs (exact match)
    name_filter: Optional[str] = None,       # substring on ad set or campaign name
    include_campaign_overlay: bool = False,    # add campaign series on adset charts
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_api_key_query: Optional[str] = Query(default=None, alias="x_api_key"),
):
    require_api_key(x_api_key, x_api_key_query)

    if not FB_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Server missing FB_ACCESS_TOKEN.")
    acct = account_id or DEFAULT_ACCOUNT_ID
    if not acct:
        raise HTTPException(status_code=400, detail="Provide ?account_id=act_XXXX or set FB_AD_ACCOUNT_ID env var.")

    level_lc = level.lower()
    if level_lc not in ("adset", "campaign"):
        raise HTTPException(status_code=400, detail="Supports level=adset or level=campaign.")

    tz_name = tz or DEFAULT_TZ

    # Caches to avoid redundant campaign-level calls when overlaying campaign data on adset rows
    campaign_today_cache: Dict[str, float] = {}
    campaign_cum_cache: Dict[str, float] = {}
    campaign_series_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}  # (cid, since, until) -> series rows

    # Preprocess campaign_ids into a set for quick matching
    campaign_id_set: Optional[Set[str]] = None
    if campaign_ids:
        campaign_id_set = {c.strip() for c in campaign_ids.split(",") if c.strip()}

    name_filter_lc = name_filter.lower() if name_filter else None
    campaign_filter_lc = campaign_filter.lower() if campaign_filter else None

    now = datetime.now(timezone.utc)
    items_out: List[Dict[str, Any]] = []

    # =====================
    # level = ADSET branch
    # =====================
    if level_lc == "adset":
        adsets = get_adsets(acct, limit=limit)

        for adset in adsets:
            campaign = adset.get("campaign") or {}
            campaign_name = (campaign.get("name") or "")
            campaign_id = (campaign.get("id") or "")

            # Campaign ID filter (exact match list)
            if campaign_id_set and campaign_id not in campaign_id_set:
                continue

            # Campaign name substring filter (case-insensitive)
            if campaign_filter_lc and (campaign_filter_lc not in campaign_name.lower()):
                continue

            adset_id = adset.get("id")
            name = adset.get("name") or ""

            # Ad set name substring filter (case-insensitive)
            if name_filter_lc and (name_filter_lc not in name.lower()):
                continue

            eff_status = adset.get("effective_status")
            cfg_status = adset.get("configured_status")
            start_time = parse_fb_time(adset.get("start_time"))
            end_time = parse_fb_time(adset.get("end_time"))

            effective_ok_statuses = {"ACTIVE", "ADSET_ACTIVE"}
            is_active = (eff_status in effective_ok_statuses) or (cfg_status in effective_ok_statuses)

            # Budgets (Meta returns minor units)
            daily_budget = to_float(adset.get("daily_budget") or 0) / 100.0
            lifetime_budget = to_float(adset.get("lifetime_budget") or 0) / 100.0

            # Skip paused unless include_paused
            if (not include_paused) and (not is_active):
                continue

            # Skip no-budget ad sets unless include_paused=true
            if (daily_budget < min_daily_budget) and (lifetime_budget <= 0):
                if not include_paused:
                    continue

            # Spend today
            ins_today = get_insights_single(adset_id, date_preset="today")
            spend_today = to_float(ins_today.get("spend", 0.0))

            # Spend to date (since start or last 90 days if no start)
            since = (start_time or (now - timedelta(days=90))).date().isoformat()
            until = now.astimezone(account_tz(tz_name)).date().isoformat()
            ins_cum = get_insights_single(adset_id, **{"time_range[since]": since, "time_range[until]": until})
            spend_to_date = to_float(ins_cum.get("spend", 0.0))

            # Expectations
            expected_today_so_far = compute_daily_expected_so_far(daily_budget, now, tz_name) if daily_budget > 0 else None
            expected_to_date = compute_flight_expected(lifetime_budget, start_time, end_time, now) if lifetime_budget > 0 else None

            pacing_today_ratio = spend_today / expected_today_so_far if expected_today_so_far and expected_today_so_far > 0 else None
            pacing_flight_ratio = spend_to_date / expected_to_date if expected_to_date and expected_to_date > 0 else None

            # Neutralize "today" pacing for non-active sets so they don't show "behind"
            if not is_active:
                expected_today_so_far = None
                pacing_today_ratio = None

            # Remaining & recommendation
            days_remaining = avg_daily_needed = None
            recommendation = None
            if lifetime_budget > 0 and end_time:
                days_remaining = max(1, math.ceil((end_time - now).total_seconds() / 86400))
                remaining = max(0.0, lifetime_budget - spend_to_date)
                avg_daily_needed = remaining / days_remaining
                if daily_budget > 0:
                    if avg_daily_needed > daily_budget * 1.1:
                        recommendation = f"Increase daily budget to ~{avg_daily_needed:.0f} to hit flight target."
                    elif avg_daily_needed < daily_budget * 0.9:
                        recommendation = f"Decrease daily budget to ~{avg_daily_needed:.0f} to avoid overspend."
                    else:
                        recommendation = "Daily budget looks aligned with flight target."

            item = {
                "id": adset_id,
                "name": name,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "status": eff_status or cfg_status,
                "budget_type": "DAILY" if daily_budget > 0 else ("LIFETIME" if lifetime_budget > 0 else "NONE"),
                "daily_budget": daily_budget or None,
                "lifetime_budget": lifetime_budget or None,
                "start_time": iso(start_time),
                "end_time": iso(end_time),
                "spend_today": spend_today,
                "spend_to_date": spend_to_date,
                "expected_today_so_far": expected_today_so_far,
                "expected_to_date": expected_to_date,
                "pacing_today_ratio": pacing_today_ratio,
                "pacing_flight_ratio": pacing_flight_ratio,
                "pacing_today_status": classify_ratio(pacing_today_ratio) if pacing_today_ratio is not None else "n/a",
                "pacing_flight_status": classify_ratio(pacing_flight_ratio) if pacing_flight_ratio is not None else "n/a",
                "days_remaining": days_remaining,
                "avg_daily_needed": avg_daily_needed,
                "recommendation": recommendation,
            }

            # Optional: 7-day series (today inclusive)
            tzobj = account_tz(tz_name)
            today_local = now.astimezone(tzobj).date()
            since_7 = (today_local - timedelta(days=6)).isoformat()  # 6 days ago + today = 7 rows
            until_7 = today_local.isoformat()

            if include_series_7d:
                series_rows = get_insights_series(adset_id, since_7, until_7)
                series = [{"date": r.get("date_start"), "spend": to_float(r.get("spend", 0.0))}
                          for r in series_rows if r.get("date_start")]
                item["spend_series_7d"] = series

            # NEW: Optional campaign overlay for charts (adds a second line)
            if include_campaign_overlay and campaign_id:
                # today
                if campaign_id not in campaign_today_cache:
                    ins_c_today = get_insights_single(campaign_id, date_preset="today")
                    campaign_today_cache[campaign_id] = to_float(ins_c_today.get("spend", 0.0))
                item["campaign_spend_today"] = campaign_today_cache[campaign_id]

                # to-date
                if campaign_id not in campaign_cum_cache:
                    ins_c_cum = get_insights_single(campaign_id, **{"time_range[since]": since, "time_range[until]": until})
                    campaign_cum_cache[campaign_id] = to_float(ins_c_cum.get("spend", 0.0))
                item["campaign_spend_to_date"] = campaign_cum_cache[campaign_id]

                # 7d series overlay (uses same local date window)
                key7 = (campaign_id, since_7, until_7)
                if key7 not in campaign_series_cache:
                    c_series_rows = get_insights_series(campaign_id, since_7, until_7)
                    campaign_series_cache[key7] = [{"date": r.get("date_start"), "spend": to_float(r.get("spend", 0.0))}
                                                   for r in c_series_rows if r.get("date_start")]
                item["campaign_spend_series_7d"] = campaign_series_cache[key7]

            items_out.append(item)

    # ========================
    # level = CAMPAIGN branch
    # ========================
    else:
        campaigns = get_campaigns(acct, limit=limit)

        for camp in campaigns:
            campaign_id = camp.get("id")
            campaign_name = camp.get("name") or ""

            # Campaign ID filter (exact match list)
            if campaign_id_set and campaign_id not in campaign_id_set:
                continue

            # Campaign name substring filters
            if campaign_filter_lc and (campaign_filter_lc not in campaign_name.lower()):
                continue
            if name_filter_lc and (name_filter_lc not in campaign_name.lower()):
                continue

            eff_status = camp.get("effective_status")
            cfg_status = camp.get("configured_status")

            # Effective statuses vary; consider ACTIVE (and historical variants) as active
            effective_ok_statuses = {"ACTIVE", "CAMPAIGN_ACTIVE", "CAMPAIGN_GROUP_ACTIVE"}
            is_active = (eff_status in effective_ok_statuses) or (cfg_status in effective_ok_statuses)

            # Budgets (minor units)
            daily_budget = to_float(camp.get("daily_budget") or 0) / 100.0
            lifetime_budget = to_float(camp.get("lifetime_budget") or 0) / 100.0

            # Times (campaign-level CBO has stop_time, not end_time)
            start_time = parse_fb_time(camp.get("start_time"))
            end_time = parse_fb_time(camp.get("stop_time"))

            # Skip paused unless include_paused
            if (not include_paused) and (not is_active):
                continue

            # Skip no-budget campaigns unless include_paused=true
            if (daily_budget < min_daily_budget) and (lifetime_budget <= 0):
                if not include_paused:
                    continue

            # Spend today (campaign insights)
            ins_today = get_insights_single(campaign_id, date_preset="today")
            spend_today = to_float(ins_today.get("spend", 0.0))

            # Spend to date
            since = (start_time or (now - timedelta(days=90))).date().isoformat()
            until = now.astimezone(account_tz(tz_name)).date().isoformat()
            ins_cum = get_insights_single(campaign_id, **{"time_range[since]": since, "time_range[until]": until})
            spend_to_date = to_float(ins_cum.get("spend", 0.0))

            # Expectations
            expected_today_so_far = compute_daily_expected_so_far(daily_budget, now, tz_name) if daily_budget > 0 else None
            expected_to_date = compute_flight_expected(lifetime_budget, start_time, end_time, now) if lifetime_budget > 0 else None

            pacing_today_ratio = spend_today / expected_today_so_far if expected_today_so_far and expected_today_so_far > 0 else None
            pacing_flight_ratio = spend_to_date / expected_to_date if expected_to_date and expected_to_date > 0 else None

            # Neutralize "today" pacing for non-active so they don't show "behind"
            if not is_active:
                expected_today_so_far = None
                pacing_today_ratio = None

            # Remaining & recommendation (campaign)
            days_remaining = avg_daily_needed = None
            recommendation = None
            if lifetime_budget > 0 and end_time:
                days_remaining = max(1, math.ceil((end_time - now).total_seconds() / 86400))
                remaining = max(0.0, lifetime_budget - spend_to_date)
                avg_daily_needed = remaining / days_remaining
                if daily_budget > 0:
                    if avg_daily_needed > daily_budget * 1.1:
                        recommendation = f"Increase daily budget to ~{avg_daily_needed:.0f} to hit flight target."
                    elif avg_daily_needed < daily_budget * 0.9:
                        recommendation = f"Decrease daily budget to ~{avg_daily_needed:.0f} to avoid overspend."
                    else:
                        recommendation = "Daily budget looks aligned with flight target."

            item = {
                "id": campaign_id,
                "name": campaign_name,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "status": eff_status or cfg_status,
                "budget_type": "DAILY" if daily_budget > 0 else ("LIFETIME" if lifetime_budget > 0 else "NONE"),
                "daily_budget": daily_budget or None,
                "lifetime_budget": lifetime_budget or None,
                "start_time": iso(start_time),
                "end_time": iso(end_time),
                "spend_today": spend_today,
                "spend_to_date": spend_to_date,
                "expected_today_so_far": expected_today_so_far,
                "expected_to_date": expected_to_date,
                "pacing_today_ratio": pacing_today_ratio,
                "pacing_flight_ratio": pacing_flight_ratio,
                "pacing_today_status": classify_ratio(pacing_today_ratio) if pacing_today_ratio is not None else "n/a",
                "pacing_flight_status": classify_ratio(pacing_flight_ratio) if pacing_flight_ratio is not None else "n/a",
                "days_remaining": days_remaining,
                "avg_daily_needed": avg_daily_needed,
                "recommendation": recommendation,
                # Nice-to-haves when present
                "pacing_type": camp.get("pacing_type"),
                "spend_cap": to_float(camp.get("spend_cap") or 0) / 100.0 if camp.get("spend_cap") else None,
            }

            # Optional: 7-day series
            if include_series_7d:
                tzobj = account_tz(tz_name)
                today_local = now.astimezone(tzobj).date()
                since_7 = (today_local - timedelta(days=6)).isoformat()
                until_7 = today_local.isoformat()
                series_rows = get_insights_series(campaign_id, since_7, until_7)
                series = [{"date": r.get("date_start"), "spend": to_float(r.get("spend", 0.0))}
                          for r in series_rows if r.get("date_start")]
                item["spend_series_7d"] = series

            items_out.append(item)

    # Sorting by "worst pacing"
    if sort_by in ("flight", "today"):
        items_out.sort(key=lambda it: worst_key(it, sort_by, overspend_first))

    # Debug summary to help explain results/filters
    debug_summary = {
        "requested_limit": limit,
        "returned_count": len(items_out),
        "filters": {
            "include_paused": include_paused,
            "min_daily_budget": min_daily_budget,
            "campaign_filter": campaign_filter,
            "campaign_ids": campaign_ids,
            "name_filter": name_filter,
            "include_series_7d": include_series_7d,
            "sort_by": sort_by,
            "overspend_first": overspend_first,
            "include_campaign_overlay": include_campaign_overlay
        }
    }

    return {
        "generated_at": iso(now),
        "timezone": tz_name,
        "level": level_lc,
        "count": len(items_out),
        "items": items_out,
        "debug": debug_summary,
    }

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["servers"] = [{"url": "https://phd-ai-meta-pacing.onrender.com"}]  # <-- your Render URL
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
