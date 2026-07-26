# Security Policy

Intelligence OS ingests camera footage and stores a memory graph of the people
in it. A vulnerability here is not a leaked to-do list — it is someone else's
movements. We take reports seriously and we would rather hear about a
maybe-issue than not.

## Supported versions

| Version | Supported |
|---|---|
| `main` | ✅ Fixes land here first |
| `0.1.x` | ✅ |
| < 0.1 | ❌ Pre-release, no backports |

The project is pre-1.0: the fix for a reported issue may ship as a breaking
change if that is the correct fix.

## Reporting a vulnerability

**Please do not open a public issue, PR, or Discussion for a security problem.**

Report it privately through GitHub:

> **[Open a private security advisory →](https://github.com/Infinex-Labs/Intelligence-OS/security/advisories/new)**
> (Repository → *Security* → *Advisories* → *Report a vulnerability*)

That channel is visible only to maintainers, and it gives us a private fork to
develop and review the fix in.

If you cannot use GitHub advisories, email the maintainers at
**[INSERT SECURITY EMAIL — fill this in before the repository goes public]**.

### What to include

The more of this you can give us, the faster the fix:

- What kind of issue it is (auth bypass, IDOR, path traversal, RCE, injection…).
- The file and, if you have it, the line — e.g. `intelligence_os/web.py:812`.
- Steps to reproduce, ideally a `curl` invocation or a short script.
- What an attacker gets out of it. "Any signed-in user can read another
  operator's threads" is much more actionable than "the endpoint looks wrong".
- Whether you are willing to be credited, and under what name.

### What to expect

| | |
|---|---|
| First response | within **3 business days** |
| Assessment and severity | within **7 days** |
| Fix or documented mitigation | target **30 days**, sooner for anything critical |
| Disclosure | coordinated — we publish the advisory once a fix is available |

We will keep you updated as the fix progresses, credit you in the advisory and
the changelog unless you ask us not to, and tell you plainly if we decide
something is not a vulnerability and why.

There is no paid bug bounty.

## Scope

**In scope** — anything in this repository:

- Authentication and session handling (`auth.py`, `web.py`).
- Authorization between users: reading, renaming, deleting or writing into
  another operator's threads, entities, rules or reports. Ownership is enforced
  as a `WHERE` clause in `store.py`; a bypass of that is in scope.
- Path traversal in the static and keyframe routes.
- SQL injection anywhere in `store.py`.
- Anything that leaks keyframes, RTSP credentials, or the `ANTHROPIC_API_KEY`.
- Remote code execution via the rules compiler or a config file.

**Out of scope:**

- Anything requiring physical access to the host or the camera network.
- Denial of service by pointing the pipeline at a huge or malformed video.
- Vulnerabilities in dependencies (YOLO, OpenCV, InsightFace) with no
  exploitable path through this codebase — report those upstream.
- Missing hardening on a deployment you exposed to the internet yourself (see
  below).

## Known limits you should design around

These are documented properties of the current design, not bugs — but they
determine how you should deploy it:

1. **Bind address.** The server listens on `127.0.0.1` by default, and that
   default is load-bearing: there is no TLS, no rate limiting, and no CSRF
   tokens. `--host` can widen it, and the Docker image sets `--host 0.0.0.0`
   because a container's network namespace is its boundary — but the published
   port there still goes to the host's loopback. Anywhere else, put it behind a
   reverse proxy that terminates TLS. Do not expose it directly to the internet.
2. **`config.yaml` holds credentials.** RTSP URLs commonly embed a username and
   password. The file is gitignored for that reason. Keep it that way.
3. **The database is not encrypted at rest.** `memory.db` and the retained
   keyframes are plain files. Use full-disk encryption if that matters.
4. **Face recognition is opt-in and stays that way.** `IdentityConfig.enabled`
   defaults to `false`. A change that turns biometric matching on by default
   would be treated as a security regression.
5. **Deletion is real.** `operator delete <entity_id>` cascades to signatures,
   observations and relations — that is the privacy-removal path, and it is
   meant to be irreversible.
