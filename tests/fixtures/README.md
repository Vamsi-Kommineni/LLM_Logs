# Fixtures

Provider response bodies used by the adapter tests. No test touches the network.

| Files | Origin |
| --- | --- |
| `groq_*.json` | Recorded from the live Groq API with `scripts/record_fixtures.py`, then scrubbed: identifiers, fingerprints and timestamps are replaced. Bodies only, never headers. |
| `openai_*.json`, `anthropic_*.json` | Written by hand from the public API references, because no key for these providers was available. The tests parse every one of them into the provider SDK's own response types, so a shape the SDK would reject fails the suite. Replace them with recorded bodies when you can. |

Model IDs appear here and nowhere else in `src/` or `tests/`.
