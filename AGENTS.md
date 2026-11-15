# AGENTS.md

Project guidelines for AI coding agents working on `manga2kindle`.

## Documentation & Planning

- Store all specification/design files in `docs/specs/` and implementation plan files in `docs/plans/` (e.g. `docs/specs/YYYY-MM-DD-<topic>-design.md`, `docs/plans/YYYY-MM-DD-<topic>-plan.md`).
- Keep `README.md` up to date whenever CLI commands, flags, installation steps, or system dependencies change.
- Keep `.gitignore` updated so virtual environments, build artifacts, temporary conversion directories, and sample `.cbz`/`.cbr`/`.kpf`/`.kfx` files are never committed.
- Use `superpowers` skills (`brainstorming`, `writing-plans`, `test-driven-development`, `systematic-debugging`, `verification-before-completion`) when appropriate.

## Engineering Principles

- **OOP:** Model pipeline stages and domain concepts (archives, pages, panels, device profiles, KPF/KFX compilers) as cohesive, single-responsibility classes with clear interfaces.
- **Typed Objects Over Plain Dicts:** Prefer structured objects (`Pydantic` models or `dataclass`es) over untyped `dict`s for domain records, intermediate pipeline state, and internal payloads; reserve plain `dict`s for dynamic lookup tables or serialization boundaries.
- **No Magic Strings:** Replace repeated, configuration, or protocol-significant string literals (e.g., metadata keys, Ion/KPF element types, statuses, archive extensions) with `Enum`s or named module constants.
- **KISS:** Prefer straightforward, readable implementations over clever abstractions.
- **DRY:** Extract shared helpers (e.g., cross-platform executable discovery, natural sorting, bounding-box math) instead of duplicating logic.
- **YAGNI:** Build only what is required for the current spec; do not add speculative features, unused config knobs, or premature plugin systems.

## Cross-Platform & Quality Standards

- Target Python 3.11+ and ensure full compatibility on both **macOS** and **Windows** using `pathlib.Path` exclusively for filesystem paths.
- Use `pytest` for unit and integration tests; verify changes by running tests before claiming completion.

## Git Workflow, Commits & Versioning

- **Branch Workflow (`main` > `develop` > feature branches):**
  - Maintain `main` as the stable release branch and `develop` as the integration branch.
  - Always branch feature/fix branches (e.g., `feat/<topic>`, `fix/<topic>`) off `develop` and merge them back into `develop`.
  - Never merge feature branches directly into `main`; only `develop` merges into `main`.
- **Conventional Commits:** Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, with optional scope like `feat(cli): ...`) for all commit messages.
- **No Co-Author Trailers:** Never include co-authoring trailers or messages (such as `Co-Authored-By:`) in commit messages.
- **Semantic Versioning:** Tag releases on `main` using Semantic Versioning (`vMAJOR.MINOR.PATCH`, e.g., `v0.1.0`), aligned with `pyproject.toml` version bumps.
