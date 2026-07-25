# Assets

Brand and documentation assets. These are the copies the README and `docs/`
reference.

| File | Use |
|---|---|
| `logo.png` | Dark-background logo — the README hero, GitHub's dark theme. |
| `logo-light.png` | Light-background variant. |

The app serves its **own** copies from `intelligence_os/static/`, because those
ship inside the wheel and are loaded by the dashboard at runtime. The two sets
are intentionally separate: one is documentation, one is product. Update both
when the mark changes.

## Still missing

- **`demo.gif`** — 30–60 seconds showing the loop this project exists for: a
  question typed into the assistant, an answer, and the keyframes it cites.
  Nothing explains this system faster, and its absence is the biggest remaining
  gap in the README.
- **`architecture.png`** — the cascade diagram in
  [docs/architecture.md](../docs/architecture.md) is ASCII today, which is
  readable and diffable but not what anyone screenshots.
- **`screenshot-*.png`** — dashboard, timeline, assistant, in both themes.

Screenshots must be taken against seeded or synthetic data, never real footage
of real people.
