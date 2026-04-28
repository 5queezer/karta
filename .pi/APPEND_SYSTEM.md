## Karta memory

Karta memory tools may be available in this project.

Use `karta_search` before answering questions that may depend on previous project decisions, maintainer preferences, recurring bugs, architectural context, or other durable history.

Use `karta_ask` when the user asks a question that should be answered from stored project memory rather than only the current context.

Use `karta_add_note` only for stable, reusable information:

- user or project preferences
- architecture decisions
- recurring constraints
- bug root causes and fixes worth remembering
- important TODOs or follow-up context
- decisions that should survive across sessions

Do not store:

- secrets, credentials, tokens, or private keys
- raw logs or command output
- transient scratch work
- large code blocks
- speculative guesses
- every assistant response

Prefer concise, atomic notes with enough context to be useful later.
