# Deployment

Honest starting point: **this is a prototype, and its threat model is "a machine
you control, on a network you control."** It binds `127.0.0.1`, has no TLS, no
CSRF tokens and no rate limiting. None of that is a secret — it is in
[SECURITY.md](../SECURITY.md) — and it determines how you should run it.

Two supported paths: **[Docker](docker.md)** (shortest, and what most people
should use) or a virtualenv plus a process supervisor, documented below. The
lock-down advice on this page applies to both.

## Sizing

| | |
|---|---|
| CPU | The motion gate is cheap; YOLO is not. One camera on a modern laptop CPU is comfortable; four wants a GPU or a smaller model. |
| RAM | Models are loaded **once and shared across camera threads**, so memory grows with model choice, not camera count. Budget ~2 GB with identity enabled. |
| Disk | Keyframes, pruned at `raw_retention_days` (default 7). The database itself stays small — it stores a graph, not video. |
| Network | Only outbound to the Anthropic API, and only if you set a key. |

## Running it as a service

```bash
cd /opt/intelligence-os
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

`systemd`:

```ini
[Unit]
Description=Intelligence OS
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=intelligence
WorkingDirectory=/opt/intelligence-os
Environment=INTELLIGENCE_OS_DATA=/var/lib/intelligence-os
ExecStart=/opt/intelligence-os/.venv/bin/intelligence-os --port 8000
Restart=on-failure
RestartSec=10

# The process needs nothing outside its data directory.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/intelligence-os

[Install]
WantedBy=multi-user.target
```

The `intelligence-os` console script resolves its static assets against the
installed package, so the working directory doesn't have to be the repo — but
`config.yaml` is still read from the repo root (or `INTELLIGENCE_OS_CONFIG`).

## Exposing it beyond localhost

The server binds `127.0.0.1`. **Do not change that to `0.0.0.0` and call it
done.** Put a reverse proxy in front:

```nginx
server {
    listen 443 ssl;
    server_name cameras.internal.example;

    ssl_certificate     /etc/ssl/certs/your.crt;
    ssl_certificate_key /etc/ssl/private/your.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # The MJPEG stream is one long-lived response per viewer per camera.
        proxy_buffering off;
        proxy_read_timeout 24h;
    }
}
```

Then, before anyone else can reach it:

1. **Terminate TLS at the proxy.** Sessions are cookies; plain HTTP means
   sniffable sessions.
2. **Restrict by source** — VPN, Tailscale, or an allowlist. There is no rate
   limiting on the login endpoint.
3. **Consider proxy-level auth** in front of the app's own, if the audience is
   wider than a couple of trusted operators.

Do not put this directly on the public internet.

## Backups

Everything that matters is in the data directory:

```bash
systemctl stop intelligence-os        # WAL mode: a live copy can be inconsistent
tar czf backup-$(date +%F).tar.gz /var/lib/intelligence-os
systemctl start intelligence-os
```

Or hot-copy the database properly with `sqlite3 memory.db ".backup out.db"`,
which is WAL-safe, and rsync the frames directory separately.

`memory.db` contains face embeddings — biometric data. Treat the backup the way
you would treat the original: full-disk encryption at minimum. The database is
**not** encrypted at rest.

## Cameras

- RTSP over a **wired, isolated VLAN** if you can. Camera firmware is not
  something to trust on a flat network.
- Credentials live in `config.yaml`, which is gitignored. If you keep a
  deployment repo, keep the real file out of it — commit a redacted sample like
  [`examples/04_multi_camera/config.yaml`](../examples/04_multi_camera/config.yaml).
- A camera that refuses a connection at startup currently takes its thread down
  for the life of the process. Until retry-with-backoff lands, `Restart=on-failure`
  plus a health check on `/api/cameras` is the practical mitigation.

## Before you point this at real people

The short form. **[Responsible use](responsible-use.md)** is the long form, with
the jurisdiction-by-jurisdiction detail and the full checklist — read it before
a deployment anyone but you can see.

Not legal advice, but the questions any deployment should be able to answer:

- **Do the people in frame know?** Signage is a legal requirement in many
  jurisdictions and a decency requirement everywhere.
- **Is face matching genuinely necessary?** It ships off. Most rules, zones,
  alerts and timelines work without it. Turning it on makes this a biometric
  system, with the legal weight that carries (GDPR Art. 9 and equivalents).
- **How long do you keep it?** `raw_retention_days` for keyframes; the graph
  persists until deleted. Set a retention policy and configure it, rather than
  keeping everything by default.
- **How does someone get removed?** `python -m intelligence_os.operator delete
  <entity_id>` cascades to signatures, observations and relations. Know how to
  do it before you're asked to.
- **Who can see the dashboard?** Every signed-in operator can see the whole
  graph. There is no role separation yet; today, an account is full access.

This system is retrospective by design. It answers what happened. It is not a
safety interlock and must not be deployed as one.

**If you are deploying it for someone else, or as part of a product, also read
[licensing](licensing.md)** — YOLO arrives under AGPL-3.0, and hosting the
dashboard for other users engages its network clause.
