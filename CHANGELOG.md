# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning:
[SemVer](https://semver.org/).

## [1.1.0] — 2026-11-12

### Added

- **An Agent Skill and a Claude Code plugin** (`/plugin marketplace add zch6a/ai-env-doctor`). Ask the assistant
  why PyTorch does not use the GPU, why downloads fail, or which local model fits: it runs ai-env-doctor in the
  project, reads `--json` by finding id and level (never by the translated titles), explains what is wrong, and shows
  the fix commands before running any. A test runs every command the skill gives on a synthetic machine, offline,
  and checks the fields the skill tells the assistant to read.
- This changelog.

## [1.0.0] — 2026-11-11

The first stable release. `--json` is covered by [docs/json.md](docs/json.md) (schema 1). What changed on the way
there, round by round, is in [POLISH_LOG.md](POLISH_LOG.md).
