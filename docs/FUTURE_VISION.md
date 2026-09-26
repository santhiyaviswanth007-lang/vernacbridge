# Future Vision (NOT V1 scope)

This file exists so future-vision ideas get written down instead of either
being forgotten or accidentally implemented as if they were part of V1.

**V1 is intentionally scoped to:** a Tanglish-only Python package, built on
the existing ~4,000-row dataset, proving real language-processing
engineering capability as a college-level project. Phases 1–3 (Translation
Pipeline, Dataset Intelligence, Memory Layer) are complete and stable.
Phase 4+ continues within this same scope (context, further intelligence),
still using the existing dataset — no new data domains required.

**Before adding anything to this file as a "next phase," ask:** *is this
necessary for the current V1 Tanglish package, or is this a future vision
feature?* If it's the latter, it belongs here, not in the roadmap.

---

## Ideas explicitly deferred past V1

### Professional / formal communication data domain
Real gap identified while discussing V1's long-term motivation: the current
`career` domain dataset is entirely casual, first-person *emotional venting
about* work situations ("I feel anxious about my review") rather than
actual professional-register content someone would want translated (an
email, a resume line, a request to a manager). Corpus-matching can't
translate a register it has no labeled examples of.

**Not being built in V1.** If pursued later, it would mean a new set of
labeled rows (same schema: `tanglish_input, english_output, intent,
emotion, keywords, urgency, confidence`) covering emails, resumes,
interview messages, and workplace requests — ideally authored by someone
with direct experience of what a Tamil professional actually needs to say
in English, since that authenticity matters more than the schema mechanics.

### Multilingual expansion (Manglish, Hinglish, other variants)
Explicitly out of scope for V1 and any near-term version unless requested.
If ever pursued, the current schema would need a `language` field added
before mixed-language data is collected, to avoid cross-language false
matches in fuzzy search. Not needed now — noted only so it isn't a painful
retrofit if this direction is ever revisited.

### LOVE AI integration
VernacBridge is being built as an independent, standalone Python package.
It is not being designed around LOVE AI's needs, and no LOVE-specific code,
naming, or coupling belongs in this package. If a separate future project
wants to consume VernacBridge as a language-understanding layer, that's a
decision for that project to make against VernacBridge's public API as it
exists — not a reason to change VernacBridge itself now.

---

*Nothing in this file is scheduled. It's a parking lot, not a roadmap.*
