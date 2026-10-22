# Developer Documentation Profile

## Audience And Scope

Write `README*.md` for users who want to build and run the application. Write
`docs/DEVELOPMENT*.md` for contributors who need the architecture, ownership,
runtime flow, invariants, validation commands, and troubleshooting context.
Keep implementation-only detail in the matching skill reference when it would
obscure the contributor workflow.

Use the Diátaxis distinction between tutorial, how-to, reference, and
explanation. A single page may contain more than one type, but keep each section
focused. Order the development guide as: quickstart/how-to, architecture
explanation, technical reference, then troubleshooting.

Sources:

- https://diataxis.fr/
- https://docs.github.com/en/contributing/style-guide-and-content-model/about-the-content-model
- https://developers.google.com/style/highlights

## Project Rules

- All documentation and code comments describe confirmed current branch state;
  only `plan.md` may contain planning or historical records. Do not include AI
  modification logs, temporary phase labels or superseded decisions. Verify
  requirements and implementation separately: an unconfirmed discrepancy is
  not a new business rule. State unresolved limitations without inventing intent.
- Treat code, CMake, tests, shader declarations, and generated target names as
  the source of truth. Verify names and behavior with `rg` before editing docs.
- Lead with the task or outcome. Put prerequisites before commands and expected
  results after them.
- Use sentence-case headings, active voice, and short paragraphs.
- Use numbered lists only when order, dependency, or chronology matters. Use
  round bullet lists for capabilities, alternatives, prerequisites, rules, and
  any other collection without a meaningful sequence.
- Use tables only for exact mappings or comparisons.
- Put file names, symbols, commands, environment variables, UI labels, and
  literal values in code formatting. Use descriptive link text.
- Keep commands runnable from the repository root. State platform/configuration
  assumptions and prefer the existing `build` directory.
- Separate current behavior, optional/debug behavior, and known limitations.
  Never describe planned work as implemented.
- During implementation, experimentation, A/B testing, or provisional review,
  do not add feature-specific behavior to `SKILL.md` or its references. Code,
  tests, and ordinary task documentation may change as required, but the skill
  must continue to describe only retained project conventions and stable
  behavior.
- Update the skill only after the feature has passed its relevant build, test,
  runtime, performance, or quality checks and the user or maintainer explicitly
  confirms that the feature will be kept. If the feature is reverted or remains
  provisional, leave it out of the skill.
- Delete obsolete paths, APIs, and workflows from the main guide. Git history
  is the archive; do not retain deprecated instructions unless users still need
  a migration path.
- Avoid duplicating volatile implementation detail. Link to the owning source
  or a focused reference, especially for shader formulas and binding contracts.
- Keep `DEVELOPMENT.md` and `DEVELOPMENT.zh-CN.md` structurally equivalent:
  same headings, commands, tables, defaults, limitations, and ownership claims.
- Do not add a “last updated” date. Validate facts instead of relying on a
  timestamp that can become stale.

## Review Checklist

- Confirm every documented path exists.
- Confirm every target, command, mode, default, and environment variable exists
  in code or CMake.
- Check ownership statements against headers and call sites.
- Remove references to deleted compatibility adapters and old class locations.
- Run documented build/test/quality commands when the environment supports them.
- Check local Markdown links and keep external links authoritative and minimal.
- Search both language versions for renamed symbols and stale numeric defaults.
