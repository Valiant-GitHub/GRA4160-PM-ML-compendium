# Phase 2 — Skills Inventory

Run on 2026-05-21. Setup Prompt v2, Phase 2.3.

## Locations checked

| Location | Result |
|---|---|
| `~/.claude/skills/` | Does not exist |
| `/mnt/skills/public`, `/private`, `/examples` | Do not exist (Windows host, no `/mnt` mount) |
| `./.claude/skills/` (project-local) | Does not exist |
| `~/.claude/plugins/marketplaces/.../skills/` | **Exists** — 28 plugin `SKILL.md` files found |

The three standard skill directories named in the setup prompt are all absent.
File-based skills instead live in the plugin **marketplace catalog** under
`~/.claude/plugins/marketplaces/claude-plugins-official/`. These were scanned and
each candidate's `SKILL.md` description read.

## Relevant skill found

### frontend-design — RELEVANT (theming)

- **Path:** `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/frontend-design/skills/frontend-design/SKILL.md`
- **Description (from SKILL.md):** "Create distinctive, production-grade frontend
  interfaces with high design quality… Generates creative, polished code that avoids
  generic AI aesthetics." Emphasizes CSS variables, distinctive typography, cohesive
  color systems, motion, and spatial composition.
- **How it will be used in the master-prompt build:** Guides the Quarto theme work in
  `theme/` — `custom.scss` color system (CSS custom properties), typography pairing,
  light/dark toggle styling, and overall non-generic visual identity for the study site.
  Note: this skill is not in the harness-active skill list (the `Skill` tool), so it
  will be applied by **reading and following its `SKILL.md` guidance manually** during
  the build rather than invoked as a slash command.

## Categories assessed — not found

- **PDF reading / text extraction:** **No PDF skill found.** Logged explicitly per
  setup prompt §2.3. The master-prompt build will rely entirely on **`pymupdf`
  (primary)** + **`pdfplumber` (fallback)** for the 10 lecture PDFs, ESL/ISL textbooks,
  exam PDFs, and guidelines. Not a blocker.
- **File reading / document handling:** No dedicated document-ingestion skill found.
  General source ingestion will use the harness's built-in `Read` tool plus the Python
  libraries above.

## Other skills present (not relevant to this build)

Messaging (discord, imessage, telegram), plugin/MCP tooling (plugin-dev family,
mcp-server-dev family, skill-creator), claude-md-management, hookify, math-olympiad,
session-report, cardputer/m5 maker skills, example/playground stubs. None apply to a
Quarto study-site build from PDF/notebook sources.

## Harness-active skills (via the Skill tool, listed at session start)

`update-config`, `keybindings-help`, `simplify`, `fewer-permission-prompts`, `loop`,
`schedule`, `claude-api`, `init`, `review`, `security-review`. None are PDF, document,
or frontend skills — none directly relevant to the master-prompt build.
