"""Delivery engine for briefings and real-time alerts (FinalPRD §9.5 — M9).

Handles email (SMTP) and Webhook delivery channels for scheduled digests and
real-time rule alerts.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import time
import urllib.request
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from intelligence_os.store import Store
from intelligence_os.digest import build as build_digest

logger = logging.getLogger("intelligence_os.deliver")
logger.setLevel(logging.INFO)

# --- Configuration Loader ---
SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "noreply@intelligenceos.local")
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "True").lower() in ("true", "1")
SMTP_USE_SSL = os.environ.get("SMTP_USE_SSL", "False").lower() in ("true", "1")


def send_webhook(url: str, payload: dict) -> bool:
    """Send JSON payload via HTTP POST to a webhook target."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.status
            logger.info(f"Webhook delivered to {url}, status {status}")
            return 200 <= status < 300
    except Exception as e:
        logger.error(f"Webhook delivery failed to {url}: {e}")
        return False


def _telegram_token() -> str:
    # read at call time, not import: the settings pane writes it into os.environ
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")


def send_telegram(chat_id: str, text: str, keyframe: Optional[str] = None) -> bool:
    """Bot API. A keyframe rides along as the photo's caption rather than a
    second message — splitting the claim from its evidence is the one thing
    this product is against."""
    token = _telegram_token()
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set. Skipping Telegram delivery.")
        return False

    kf = Path(keyframe) if keyframe else None
    if kf is not None and not kf.is_file():
        logger.warning(f"Keyframe not found, sending text only: {keyframe}")
        kf = None

    # Telegram's own caps: 1024 chars on a caption, 4096 on a message
    text = text[:1024] if kf else text[:4096]
    base = f"https://api.telegram.org/bot{token}/"

    try:
        if kf is None:
            data = json.dumps({"chat_id": str(chat_id), "text": text,
                               "disable_web_page_preview": True}).encode()
            req = urllib.request.Request(base + "sendMessage", data=data,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
        else:
            boundary, body = _multipart({"chat_id": str(chat_id), "caption": text},
                                        kf.name, kf.read_bytes())
            req = urllib.request.Request(
                base + "sendPhoto", data=body, method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            ok = 200 <= r.status < 300
        logger.info(f"Telegram delivered to {chat_id} ({'photo' if kf else 'text'})")
        return ok
    except Exception as e:
        logger.error(f"Telegram delivery failed to {chat_id}: {e}")
        return False


def _multipart(fields: dict, filename: str, blob: bytes) -> tuple[str, bytes]:
    """Minimal multipart/form-data body. sendPhoto is the only caller, so this
    stays here rather than pulling in a HTTP library for one upload."""
    import uuid
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
             .encode() for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; '
                 f'filename="{filename}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
                 + blob + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(parts)


def send_email(to_email: str, subject: str, html_body: str, text_body: str, keyframe_paths: list[str]) -> bool:
    """Send HTML and text email with inline keyframe image attachments."""
    # Fallback to logs if SMTP host is placeholder or empty
    if not SMTP_HOST or SMTP_HOST == "placeholder":
        logger.warning(f"SMTP not configured. Skipping email to {to_email}. Subject: {subject}")
        logger.info(f"Plain text content:\n{text_body}")
        return True

    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = to_email

        msg_alternative = MIMEMultipart("alternative")
        msg.attach(msg_alternative)

        msg_alternative.attach(MIMEText(text_body, "plain", "utf-8"))
        msg_alternative.attach(MIMEText(html_body, "html", "utf-8"))

        attached_cids = set()
        for kf_path_str in keyframe_paths:
            if not kf_path_str:
                continue
            kf_path = Path(kf_path_str)
            if not kf_path.exists() or not kf_path.is_file():
                logger.warning(f"Keyframe attachment not found: {kf_path}")
                continue

            cid = kf_path.name
            if cid in attached_cids:
                continue

            try:
                with open(kf_path, "rb") as f:
                    img_data = f.read()
                mime_image = MIMEImage(img_data)
                mime_image.add_header("Content-ID", f"<{cid}>")
                mime_image.add_header("Content-Disposition", "inline", filename=cid)
                msg.attach(mime_image)
                attached_cids.add(cid)
            except Exception as e:
                logger.error(f"Failed to attach inline keyframe {kf_path_str}: {e}")

        # Dispatch via SMTP
        if SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)

        if SMTP_USE_TLS and not SMTP_USE_SSL:
            server.ehlo()
            server.starttls()
            server.ehlo()

        if SMTP_USER and SMTP_PASSWORD:
            server.login(SMTP_USER, SMTP_PASSWORD)

        server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        server.quit()
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send SMTP email to {to_email}: {e}")
        return False


def _esc(s) -> str:
    """Titles/labels are operator- and rule-authored, but they land in HTML, so
    escape them (report may be opened in a browser)."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _img_src(keyframe: Optional[str], embed: bool) -> Optional[str]:
    """The <img src>. Email inlines keyframes as attachments referenced by cid:,
    but a downloaded file has no attachments — its images must be self-contained,
    so embed them as base64 data URIs. Returns None if there's no usable frame."""
    if not keyframe:
        return None
    name = Path(keyframe).name
    if not embed:
        return f"cid:{name}"
    import base64
    from intelligence_os.config import FRAMES_DIR
    try:
        blob = (Path(FRAMES_DIR) / name).read_bytes()
    except OSError:
        return None                       # frame aged out of retention -> drop it
    return "data:image/jpeg;base64," + base64.b64encode(blob).decode()


def _kpi(label: str, value, accent: bool = False) -> str:
    cls = "kpi accent" if accent else "kpi"
    return (f'<div class="{cls}"><div class="kpi-n">{value}</div>'
            f'<div class="kpi-l">{_esc(label)}</div></div>')


def _histogram_html(hist: list, since: float, now: float) -> str:
    """The activity chart. `hist` is 30 buckets, each {count, type} where type is
    r(outine)/f(ired)/u(nusual) — already computed by digest.build; we only draw it.
    A bucket that saw a fired/unusual event is colored so the eye lands on it."""
    if not hist:
        return ""
    color = {"f": "#B02F26", "u": "#9A6B1E", "r": "#B8BBB2"}
    peak = max((b["count"] for b in hist), default=0) or 1
    bars = []
    for b in hist:
        # sqrt scale: real activity is spiky, so one busy bucket on a linear scale
        # flattens all the others to nothing. Empty buckets draw no bar (0px) —
        # the container's baseline carries the axis, not a row of stubs.
        h = round((b["count"] / peak) ** 0.5 * 52) if b["count"] else 0
        bars.append(f'<div class="bar" title="{b["count"]} obs" '
                    f'style="height:{h}px;background:{color.get(b["type"], "#B8BBB2")}"></div>')
    ax = lambda ts: time.strftime("%a %H:%M", time.localtime(ts))
    return (f'<div class="chart">{"".join(bars)}</div>'
            f'<div class="chart-ax"><span>{ax(since)}</span><span>peak {peak}/bucket</span>'
            f'<span>{ax(now)}</span></div>')


def render_html(d: dict, embed_images: bool = False) -> str:
    """Render the digest as a standalone report. `embed_images=True` inlines
    keyframes (for a downloaded, offline-readable file); the default keeps cid:
    references for email, where keyframes ride as inline attachments."""
    t = lambda ts: time.strftime("%a %d %b %H:%M", time.localtime(ts))
    r = d["routine"]
    hist = d.get("histogram") or []
    total_obs = sum(b["count"] for b in hist) if hist else r["observations"]
    n_fired, n_unusual = len(d["rule_fired"]), len(d["unusual"])

    html = ["""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #EFEEE9; color: #151815; padding: 20px; line-height: 1.5; margin: 0; }
    .container { max-width: 680px; margin: 0 auto; background: #FBFAF7; border: 1px solid #DFDDD5; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
    .header { border-bottom: 1px solid #DFDDD5; padding-bottom: 16px; margin-bottom: 20px; }
    .header h1 { font-size: 20px; margin: 0; font-weight: 600; }
    .header p { font-size: 13px; color: #585C57; margin: 4px 0 0 0; }
    .header .gen { font-size: 11px; color: #8B8F88; font-family: monospace; margin-top: 6px; }
    .kpis { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px; }
    .kpi { flex: 1 1 120px; background: #FFFFFF; border: 1px solid #DFDDD5; border-radius: 8px; padding: 12px 14px; }
    .kpi.accent { border-color: #E6C6C1; background: #FAEBE9; }
    .kpi-n { font-size: 24px; font-weight: 700; line-height: 1; }
    .kpi.accent .kpi-n { color: #B02F26; }
    .kpi-l { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #8B8F88; margin-top: 6px; }
    .chart { display: flex; align-items: flex-end; gap: 3px; height: 56px; margin: 18px 0 4px; padding: 6px 4px 0; background: #FFFFFF; border: 1px solid #DFDDD5; border-radius: 8px; }
    .chart .bar { flex: 1; border-radius: 2px 2px 0 0; min-width: 2px; }
    .chart-ax { display: flex; justify-content: space-between; font-size: 10px; color: #8B8F88; font-family: monospace; margin-top: 5px; }
    .sect { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; color: #8B8F88; margin: 24px 0 12px; border-bottom: 1px solid #DFDDD5; padding-bottom: 4px; }
    .sect span { color: #B8BBB2; }
    .card { display: flex; gap: 14px; align-items: flex-start; border: 1px solid #DFDDD5; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #FFFFFF; }
    .card.fired { border-color: #E6C6C1; background: #FAEBE9; }
    .card.unusual { border-color: #E8D9BC; background: #F7F1E4; }
    .card .body { flex: 1; min-width: 0; }
    .card h3 { font-size: 14px; margin: 0; font-weight: 600; }
    .card.fired h3 { color: #B02F26; }
    .card .meta { font-size: 11px; color: #585C57; font-family: monospace; margin-top: 4px; }
    .card .why { font-size: 12px; color: #585C57; margin: 6px 0 0 0; }
    .thumb { width: 92px; height: 92px; flex-shrink: 0; border-radius: 6px; object-fit: cover; border: 1px solid #DFDDD5; background: #EFEEE9; display: block; }
    .empty { font-size: 13px; color: #8B8F88; font-style: italic; margin-bottom: 12px; }
    .routine { background: #EFEEE9; border: 1px solid #DFDDD5; border-radius: 8px; padding: 12px; font-size: 13px; color: #585C57; text-align: center; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Intelligence OS digest</h1>
"""]
    html.append(f"<p>{t(d['since'])} &rarr; {t(d['now'])}</p>"
                f'<div class="gen">generated {t(d["now"])}</div></div>')

    # KPI strip — the at-a-glance answer before anyone scrolls
    html.append('<div class="kpis">')
    html.append(_kpi("Observations", total_obs))
    html.append(_kpi("Entities seen", r["entities"]))
    html.append(_kpi("Rules fired", n_fired, accent=n_fired > 0))
    html.append(_kpi("Unusual", n_unusual))
    html.append('</div>')
    html.append(_histogram_html(hist, d["since"], d["now"]))

    # Rule fired
    html.append(f'<div class="sect">Rule Fired <span>· {n_fired}</span></div>')
    if d["rule_fired"]:
        for item in d["rule_fired"]:
            src = _img_src(item.get("keyframe"), embed_images)
            thumb = f'<img class="thumb" src="{src}" alt="">' if src else ""
            loc = f" @ {_esc(item['location'])}" if item.get('location') else ""
            html.append(f'<div class="card fired">{thumb}<div class="body">'
                        f'<h3>{_esc(item["title"])}</h3>'
                        f'<div class="meta">{t(item["timestamp"])}{loc} &middot; '
                        f'{_esc(item["label"])}</div></div></div>')
    else:
        html.append('<p class="empty">No rules fired.</p>')

    # Unusual
    html.append(f'<div class="sect">Unusual Activity <span>· {n_unusual}</span></div>')
    if d["unusual"]:
        for item in d["unusual"]:
            src = _img_src(item.get("keyframe"), embed_images)
            thumb = f'<img class="thumb" src="{src}" alt="">' if src else ""
            loc = f" @ {_esc(item['location'])}" if item.get('location') else ""
            html.append(f'<div class="card unusual">{thumb}<div class="body">'
                        f'<h3>{_esc(item["label"])}</h3>'
                        f'<div class="meta">{t(item["timestamp"])}{loc}</div>'
                        f'<p class="why">{_esc(item["title"])}</p></div></div>')
    else:
        html.append('<p class="empty">Nothing broke pattern.</p>')

    # Routine
    html.append('<div class="sect">Routine Summary</div>')
    html.append(f'<div class="routine">{r["observations"]} routine observation(s) '
                f'from {r["entities"]} entit(y/ies)</div>')
    html.append("</div></body></html>")
    return "".join(html)


def render_csv(d: dict) -> str:
    """The digest as CSV. A spreadsheet can't hold the chart or the keyframes, so
    instead of a bare findings dump we ship three stacked sections: a SUMMARY block
    (the KPIs), an ACTIVITY block (the chart's per-bucket numbers), and FINDINGS
    (every item, with a keyframe filename to look up). Blank-line-separated sections
    open cleanly in Excel/Sheets."""
    import csv
    import io
    t = lambda ts: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    r = d["routine"]
    hist = d.get("histogram") or []
    total_obs = sum(b["count"] for b in hist) if hist else r["observations"]
    buf = io.StringIO()
    w = csv.writer(buf)

    w.writerow(["SUMMARY"])
    w.writerow(["metric", "value"])
    for k, v in (("period_start", t(d["since"])), ("period_end", t(d["now"])),
                 ("generated", t(d["now"])), ("total_observations", total_obs),
                 ("entities_seen", r["entities"]), ("rules_fired", len(d["rule_fired"])),
                 ("unusual_events", len(d["unusual"]))):
        w.writerow([k, v])
    w.writerow([])

    if hist:
        span = (d["now"] - d["since"]) / len(hist)
        band = {"r": "routine", "f": "rule_fired", "u": "unusual"}
        w.writerow(["ACTIVITY"])
        w.writerow(["bucket_start", "observations", "band"])
        for i, b in enumerate(hist):
            w.writerow([t(d["since"] + i * span), b["count"], band.get(b["type"], "routine")])
        w.writerow([])

    w.writerow(["FINDINGS"])
    w.writerow(["band", "title", "entity", "entity_id", "when", "location", "keyframe"])
    for band, items in (("rule_fired", d["rule_fired"]), ("unusual", d["unusual"])):
        for i in items:
            w.writerow([band, i["title"], i["label"], i["entity_id"], t(i["timestamp"]),
                        i.get("location") or "",
                        Path(i["keyframe"]).name if i.get("keyframe") else ""])
    return buf.getvalue()


def deliver_digest(store: Store, user: dict, since: float, now: float) -> bool:
    """Build the digest briefing for a user and send it via the configured sink."""
    d = build_digest(store, since=since, now=now)
    dest = user.get("delivery_destination") or user.get("email") or user.get("username")

    if user.get("delivery_sink") == "email":
        if not dest or "@" not in dest:
            logger.warning(f"No valid email destination for user {user['username']}")
            return False

        subject = f"Intelligence OS digest brief: {time.strftime('%a %d %b')}"
        from intelligence_os.digest import render_text
        text_body = render_text(d)
        html_body = render_html(d)

        # Collect keyframe paths
        kfs = []
        for i in d.get("rule_fired", []) + d.get("unusual", []):
            if i.get("keyframe"):
                kfs.append(i["keyframe"])

        return send_email(dest, subject, html_body, text_body, kfs)

    elif user.get("delivery_sink") == "webhook":
        if not dest or not dest.startswith(("http://", "https://")):
            logger.warning(f"No valid webhook destination URL for user {user['username']}")
            return False
        return send_webhook(dest, d)

    elif user.get("delivery_sink") == "telegram":
        if not dest:
            logger.warning(f"No Telegram chat id for user {user['username']}")
            return False
        from intelligence_os.digest import render_text
        # the lead keyframe: the digest's first item that has one
        kf = next((i.get("keyframe") for i in d.get("rule_fired", []) + d.get("unusual", [])
                   if i.get("keyframe")), None)
        return send_telegram(dest, render_text(d), kf)

    else:
        logger.info(f"Digest delivery in_app only for user {user['username']}. Briefing skipped.")
        return True


def deliver_realtime_event(store: Store, event) -> None:
    """Deliver a real-time rule alert to all users with schedule set to 'realtime'."""
    users = store.conn.execute(
        "SELECT * FROM users WHERE delivery_schedule = 'realtime'"
    ).fetchall()

    for u in users:
        u_dict = dict(u)
        dest = u_dict.get("delivery_destination") or u_dict.get("email") or u_dict.get("username")

        if u_dict.get("delivery_sink") == "email":
            if not dest or "@" not in dest:
                continue
            subject = f"[ALERT] Intelligence OS: {event.rule} fired"
            text_body = f"Rule fired: {event.rule}\nEntity: {event.entity_id}\nTimestamp: {time.ctime(event.timestamp)}\nLocation: {event.location_id}\n"

            html_body = f"""
            <html>
            <body>
              <div style="font-family: sans-serif; padding: 20px; border: 1px solid #E6C6C1; background: #FAEBE9; border-radius: 8px; max-width: 500px;">
                <h2 style="color: #B02F26; margin-top: 0;">Rule Fired Alert</h2>
                <p><b>Rule:</b> {event.rule}</p>
                <p><b>Entity:</b> {event.entity_id}</p>
                <p><b>Location:</b> {event.location_id}</p>
                <p><b>Time:</b> {time.ctime(event.timestamp)}</p>
                {f'<div style="margin-top: 12px;"><img src="cid:{Path(event.keyframe).name}" style="max-width: 100%; border-radius: 6px;"></div>' if event.keyframe else ''}
              </div>
            </body>
            </html>
            """
            send_email(dest, subject, html_body, text_body, [event.keyframe] if event.keyframe else [])

        elif u_dict.get("delivery_sink") == "webhook":
            if not dest or not dest.startswith(("http://", "https://")):
                continue
            payload = {
                "event": "rule_fired",
                "rule": event.rule,
                "entity_id": event.entity_id,
                "location_id": event.location_id,
                "observation_id": event.observation_id,
                "keyframe": event.keyframe,
                "timestamp": event.timestamp
            }
            send_webhook(dest, payload)

        elif u_dict.get("delivery_sink") == "telegram":
            if not dest:
                continue
            send_telegram(dest,
                          f"⚠ {event.rule}\n"
                          f"{event.entity_id} · {event.location_id or 'no zone'}\n"
                          f"{time.ctime(event.timestamp)}",
                          event.keyframe)


# --- Background Scheduler Loop ---
def run_scheduler_tick(store: Store, now: float) -> None:
    """Run one tick of the scheduled digest checking loop."""
    local_time = time.localtime(now)
    hour = local_time.tm_hour

    users = store.conn.execute(
        "SELECT * FROM users WHERE delivery_schedule IN ('daily', 'twice_daily')"
    ).fetchall()

    for u in users:
        u_dict = dict(u)
        sched = u_dict.get("delivery_schedule")
        last_t = u_dict.get("last_delivered_at") or 0.0

        due = False
        since = now - 24 * 3600

        if sched == "daily":
            # Fire at 08:00 AM local time
            if hour == 8 and (now - last_t) >= 12 * 3600:
                due = True
                since = now - 24 * 3600
        elif sched == "twice_daily":
            # Fire at 08:00 AM and 08:00 PM local time
            if hour in (8, 20) and (now - last_t) >= 6 * 3600:
                due = True
                since = now - 12 * 3600

        if due:
            logger.info(f"Running scheduled delivery ({sched}) for user: {u_dict['username']}")
            success = False
            try:
                success = deliver_digest(store, u_dict, since=since, now=now)
            except Exception as e:
                logger.error(f"Error during digest delivery for {u_dict['username']}: {e}")

            # Update delivery timestamp so we don't double fire in the same hour window
            with store.tx() as c:
                c.execute(
                    "UPDATE users SET last_delivered_at = ? WHERE user_id = ?",
                    (now, u_dict["user_id"])
                )


def start_scheduler_thread() -> None:
    """Start the scheduled delivery worker thread."""
    def loop():
        logger.info("Starting background delivery scheduler thread...")
        while True:
            time.sleep(60)
            store = Store()
            try:
                run_scheduler_tick(store, time.time())
            except Exception as e:
                logger.error(f"Scheduler tick failed: {e}")
            finally:
                store.close()

    import threading
    t = threading.Thread(target=loop, name="delivery-scheduler", daemon=True)
    t.start()
