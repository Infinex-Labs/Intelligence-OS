# Responsible use

**Not legal advice.** The maintainers are not lawyers and this page is not a
substitute for one. It is what the project believes about how it should be used,
plus the questions a deployment ought to be able to answer. Where your
jurisdiction disagrees with this page, your jurisdiction wins.

**Who is responsible:** you are. If you run this software against real cameras
and real people, you are the data controller. The authors never see your
footage, your database, or your configuration, and cannot comply on your behalf.

---

## The short version

Intelligence OS is intended for **premises you are responsible for, with the
knowledge of the people who enter them.** That is the deployment the design
assumes and the one the defaults are tuned for.

It is not built for covert surveillance, for tracking a specific individual
without a lawful basis, or for anything a person's safety depends on. It is
retrospective by design: it tells you what happened, and it must never be
deployed as a safety interlock.

---

## The one switch that changes everything

```yaml
face_matching: false      # the default
```

With it **off**, the system processes personal data — video of identifiable
people — which is regulated, but ordinarily.

With it **on**, the system computes and stores face embeddings. In most legal
systems that is a different category entirely:

| | Off | On |
|---|---|---|
| EU/UK GDPR | Personal data (Art. 6 basis needed) | **Special category** biometric data — Art. 9 basis needed, and legitimate interest does not qualify |
| Illinois BIPA | Not engaged | Engaged. Written consent *before* collection, published retention schedule, **private right of action** with statutory damages |
| EU AI Act | Largely out of the biometric provisions | Squarely in them |
| Practical effect | Entities are stable within a session | Entities persist across days and cameras |

Turn it on deliberately, having decided you have a basis — not because a feature
looked useful. Most rules, zones, alerts and timelines work perfectly well
without it.

**Second-largest risk you could add: audio.** This system is video-only, and
that is not an accident. US wiretap statutes are considerably harsher than
video-surveillance law — roughly a dozen states require all-party consent to
record conversation, with criminal penalties and private rights of action, and
none of video law's "reasonable expectation" flexibility. Adding a microphone
would be the single largest increase in legal exposure this project could take.

---

## Before you point it at real people

A checklist, roughly in the order the questions get asked of you:

- [ ] **Do the people in frame know?** Signage at every entrance to the covered
      area. A legal requirement in much of the world and a decency requirement
      everywhere else.
- [ ] **Do you have a lawful basis** — and if face matching is on, an Art. 9
      basis or local biometric-consent equivalent?
- [ ] **Have you done a DPIA?** Under GDPR Art. 35, systematic monitoring of a
      publicly accessible area is a standing trigger. Do it before deployment,
      not after a complaint.
- [ ] **Is the camera placement lawful?** Bathrooms, changing rooms, medical and
      sleeping areas are criminal offences in most jurisdictions regardless of
      what your software does.
- [ ] **Are you covering anyone else's property or a public street?** If so, the
      "it's my house" exemption almost certainly does not apply — see below.
- [ ] **Have you set a retention period** rather than leaving the default?
      `raw_retention_days` prunes keyframes; the graph persists until deleted.
- [ ] **Do you know how to remove one person** on request, before someone asks?
      `python -m intelligence_os.operator delete <entity_id>` cascades to their
      signatures, observations and relations.
- [ ] **Can you answer a subject access request?** `operator inspect
      <entity_id>` and `/api/entity/<id>` are the tools; deciding what to
      disclose is yours.
- [ ] **Who can see the dashboard?** Every signed-in operator can see the whole
      graph. There are no roles yet.
- [ ] **Is it exposed to the network?** It binds `127.0.0.1`, has no TLS, no
      CSRF protection and no rate limiting. See [deployment](deployment.md).

### The domestic exemption is narrower than people think

GDPR exempts purely personal or household activity. The CJEU held in *Ryneš*
(C-212/13) that a home camera covering **any** part of a public space falls
outside that exemption — the homeowner becomes a full data controller.

A doorbell camera pointed at your own porch is usually fine. The same camera
catching the pavement and your neighbour's door is a different legal situation,
and UK county courts have awarded damages on exactly these facts.

---

## By jurisdiction

Non-exhaustive, and moving quickly. Confirm current status rather than trusting
this table.

### EU / UK

- **GDPR / UK GDPR** — lawful basis, transparency, minimisation, retention
  limits, subject rights. Face embeddings for identification are Art. 9 special
  category data.
- **DPIA** (Art. 35) — effectively mandatory here.
- **EU AI Act** — the relevant prohibitions and classifications:
  - Real-time remote biometric identification in publicly accessible spaces:
    **prohibited** for law enforcement, with narrow exceptions.
  - Post/retrospective remote biometric identification: **high-risk**, with
    conformity obligations attached.
  - Emotion recognition in workplaces and education: **prohibited**. (This
    project does not do it, and will not — see
    [what we will not build](#what-we-will-not-build).)
  - Building or expanding facial-recognition databases by **untargeted scraping**
    of images from the internet or CCTV footage: **prohibited** (Art. 5(1)(e)).
    A gallery built from your own premises for a defined purpose is a different
    thing from untargeted scraping — but if you are enrolling every face that
    passes an outward-facing camera, get advice before you scale it.
  - The free-and-open-source exemption (Art. 2(12)) does **not** rescue
    prohibited or high-risk uses.
- **UK** — DPA 2018, the Surveillance Camera Code, and ICO guidance on facial
  recognition, which is notably stricter than its guidance on plain CCTV.

### United States

- **Illinois BIPA** is the one that matters most: written consent before
  collecting face geometry, a published retention and destruction schedule, no
  sale or profit from biometric identifiers, and a **private right of action**
  with statutory damages. The 2024 amendment limited per-scan accrual but did
  not soften the core duties.
- **Texas (CUBI)** and **Washington** have comparable rules, enforced by the
  Attorney General rather than by private suit.
- **State privacy acts** (CCPA/CPRA and successors) treat biometric data as
  sensitive personal information with extra duties.
- **Employment** — New York requires notice of electronic monitoring; several
  states have their own rules; and monitoring that plausibly captures protected
  concerted activity raises NLRA issues.
- **No federal biometric statute** exists. Absence of a rule is not permission.

### India

The **Digital Personal Data Protection Act, 2023** brings notice-and-consent
obligations to personal data processing, with broad exemptions for the state.
Sector rules and workplace-surveillance norms apply on top. The regime is
younger than the EU's but no longer absent.

### Anywhere else

Assume something applies. Camera-in-workplace rules, notice requirements, and
biometric-specific statutes exist in most of Latin America, Canada (PIPEDA and
provincial equivalents), Australia, and much of Asia.

---

## What the software actually does

The honest technical description a DPIA needs, without marketing:

| | |
|---|---|
| **Captures** | Frames from cameras you configure. No audio, ever. |
| **Stores** | Keyframes (pruned at `raw_retention_days`, default 7) and a graph of entities, observations, zones and relations in SQLite. Not continuous video. |
| **Biometrics** | Only if you enable face matching. Then: L2-normalised face embedding vectors, capped per person, stored in `signatures`. |
| **Infers** | Presence, proximity, dwell, and repeated patterns ("habits"). Every inference retains pointers to the observations and keyframes that produced it. |
| **Does not infer** | Emotion, intent, demographics, health, or any protected characteristic. Not implemented, and on the will-not-build list. |
| **Sends externally** | Nothing, unless you set `ANTHROPIC_API_KEY` (keyframes and questions go to the Anthropic API) or configure a delivery channel (email/webhook/Telegram). No telemetry, ever. |
| **Deletes** | `cascade_delete` removes an entity's signatures, observations and relations together. It is intended to be irreversible. |
| **Accuracy** | Identity matching has a tunable threshold and a real error rate in both directions. See [tuning](tuning.md). Do not treat an entity match as proof of anything. |

That last row deserves emphasis. **Never use an output of this system as the
sole basis for an accusation, a disciplinary action, or a decision that affects
someone.** Face matching fails, and it does not fail uniformly across
demographic groups. The graph is correctable precisely because it is expected to
be wrong sometimes.

---

## What we will not build

These are positions, not scheduling notes. They do not move when someone asks
nicely:

- Real-time intervention or anything safety-critical
- Continuous video archival
- Emotion, intent, or demographic inference from faces
- Covert operation — anything that helps a deployment hide from the people it
  watches
- A hosted service holding other people's footage

Feature requests in these directions will be declined, and the reason is here
rather than in a maintainer's mood.

## What we will not help with

Issues, discussions and PRs asking for help tracking a specific individual
without a lawful basis, evading notice requirements, or operating covertly will
be closed. This is not a judgement about you; a maintainer who knowingly assists
a specific unlawful deployment has a very different legal position from one
publishing a general-purpose tool, and the project intends to stay in the
second category.

If you are unsure whether your use case is acceptable, ask in
[Discussions](https://github.com/Infinex-Labs/Intelligence-OS/discussions)
before building on it. Asking is not an admission of anything.

---

## For contributors

Two design rules follow from all of the above, and they are not negotiable in
review:

1. **Privacy defaults stay defaults.** A PR that turns face matching on by
   default, extends retention silently, weakens deletion, or makes any biometric
   feature the path of least resistance will be rejected. Adding capability is
   fine; changing what happens to someone who never read the docs is not.
2. **Every new claim carries provenance.** If your feature produces a new kind
   of inference, it must be traceable to the observations and keyframes behind
   it. An untraceable claim is unauditable, and an unauditable claim about a
   person is the thing this project exists to avoid.

See [CONTRIBUTING.md](../CONTRIBUTING.md).
