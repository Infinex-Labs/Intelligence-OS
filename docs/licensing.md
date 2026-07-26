# Licensing

**Not legal advice.** Get your own if you are making a commercial decision on
the strength of this page.

## This project

Apache License 2.0 ([LICENSE](../LICENSE)). Commercial use, modification,
distribution and private use are permitted, with an express patent grant and no
CLA. Contributions are accepted under the same licence.

## The complication: ultralytics is AGPL-3.0

Intelligence OS is Apache-2.0, but it does not run alone. `detect.py` imports
`ultralytics` for YOLO detection and ByteTrack tracking, and **ultralytics is
licensed AGPL-3.0** — as are the model weights that project distributes,
including `yolo26s.pt`.

This matters more than a typical transitive dependency for three reasons:

**1. Compatibility runs one way.** Apache-2.0 code may be combined *into* an
AGPL-3.0 work, but the resulting combination must be distributed under
AGPL-3.0. You cannot take Apache-2.0 code plus an AGPL-3.0 library and ship the
combination as Apache-2.0.

**2. Section 13 is engaged by design.** AGPL-3.0's distinguishing clause extends
the source-offer obligation to users who interact with the software **over a
network**. Intelligence OS is normally run as a web dashboard. If you host it
for anyone other than yourself, that is precisely the situation §13 was written
for.

**3. Ultralytics reads their own licence broadly.** Their published position is
that any application incorporating Ultralytics YOLO — including larger
applications and derivative works — must itself be open-sourced under
AGPL-3.0, and they sell a commercial licence specifically for organisations that
cannot do that.

### What that means for you

| You are | Position |
|---|---|
| Running it privately on your own machine | Fine. No distribution, no network users, no obligation triggered. |
| Self-hosting for your household or your own company's internal use | Generally fine, though "internal users over a network" is exactly the grey area §13 addresses. Low risk, non-zero. |
| Hosting it as a service for other people | AGPL-3.0 §13 applies. You must offer the complete corresponding source of what you are running, under AGPL-3.0. |
| Shipping it inside a commercial product | You need an Ultralytics commercial licence, or a different detector. |
| Forking and redistributing | Your combined work is AGPL-3.0, not Apache-2.0, for as long as ultralytics is a required dependency. |

### Ways out, if you need one

1. **Buy an Ultralytics Enterprise licence.** The intended path, and the
   simplest if the budget exists.
2. **Swap the detector for a permissively licensed one.** YOLOX (Apache-2.0),
   the original RT-DETR / D-FINE repositories (Apache-2.0), or torchvision's
   detection models (BSD) are all viable. **A pluggable detector backend** is a
   licensing feature at least as much as an architectural one, and it is the
   highest-value contribution anyone could make to this project's commercial
   usability.
3. **Accept AGPL-3.0 for your deployment** and publish your source. For a
   self-hosted, already-open project this often costs nothing.

Note that swapping the *code* is not enough on its own: weights published by
Ultralytics carry the same licence. A permissive detector needs permissive
weights too.

### Why the project is Apache-2.0 anyway

Because this repository's own source genuinely is Apache-2.0, and that is what
governs the contribution you make and the code you read. The AGPL obligation
arrives with a dependency, applies to the *combination*, and disappears the
moment a different detector is configured. Mislabelling the project's own source
as AGPL would misstate that.

The honest summary: **the code is Apache-2.0; a default deployment is an
AGPL-3.0 combined work.** Both statements are true, and the second is the one
that matters if you are selling something.

## Full dependency list

| Package | Licence | Required? |
|---|---|---|
| ultralytics | **AGPL-3.0** | Yes — detection and tracking |
| lap | BSD-2-Clause | Yes — ByteTrack's solver |
| opencv-python-headless | Apache-2.0 | Yes |
| numpy | BSD-3-Clause (bundles other permissive licences) | Yes |
| PyYAML | MIT | Yes |
| psutil | BSD-3-Clause | Yes |
| torch / torchvision | BSD-3-Clause | Transitive, via ultralytics |
| insightface | MIT | Optional — face identity |
| onnxruntime | MIT | Optional — face identity |
| anthropic | MIT | Optional — VLM and assistant |

Everything except ultralytics is permissive and imposes no reciprocal obligation
on your deployment.

Model weights carry their own terms independently of the code that loads them:
`yolo26s.pt` from Ultralytics is AGPL-3.0; InsightFace's `buffalo_l` is released
for non-commercial research use by its authors, which is a separate constraint
worth checking against your own use before enabling face identity commercially.

## Contributing

By opening a PR you confirm you have the right to submit the code under
Apache-2.0. Do not paste in code from an AGPL, GPL or otherwise reciprocally
licensed project — including code produced by an assistant that reproduced it —
because it would change the licence of this repository's own source, which is
the one thing currently keeping the situation above as simple as it is.
