# Development Rules

## 0. MCP Servers

This project integrates multiple MCP (Model Context Protocol) servers for AI-assisted development across the streaming infrastructure stack. See `MCP-SERVERS.md` for:
- Setup guide for all 6 MCP servers (Confluent, Kafka, PostgreSQL, MinIO, Grafana, Prometheus)
- Per-client configuration (Cursor, Claude Desktop, VS Code)
- Usage examples and troubleshooting
- Pre-configured JSON files in `mcp-config/`

## 1. Specification Adherence

**NEVER deviate from plans/specs without asking user first.**

- Follow specification document exactly: `docs/superpowers/specs/2026-05-06-ca-dqstream-architecture-design.md` (V1.9-Simplified)
- Ask user before changing: algorithms, architecture, libraries, data flows
- No "executive decisions" - even if you think it's better, ask first
- Spec is Final Boss reviewed - deviations introduce untested risks

## 2. Implementation Quality

**NEVER make "lite" versions to finish tasks quickly.**

- Implement complete, production-quality code with all error handling
- Apply ALL V1.9 bug fixes:
  - Broadcast State `.clear()` before `.put()`
  - Watermark `.withIdleness(Duration.ofSeconds(30))`
  - MurmurHash3 (not MD5)
  - AggregateFunction pattern (not ProcessWindowFunction alone)
  - asyncache (not manual lock)
- Write tests for all components (unit, integration, performance)
- No shortcuts: no skipped error handling, no hardcoded values, no "happy path only"

## 3. Project Structure

**Test files**: `/nfs/interns/dacthinh/repos/brainstorm_the/test/`
```
test/
├── unit/           # Unit tests for each component
├── integration/    # Integration tests for layer connections
├── performance/    # Throughput tests
└── fixtures/       # Test data
```

**Source code**: `/nfs/interns/dacthinh/repos/brainstorm_the/src/`
```
src/
├── layers/         # 4 processing layers
├── operators/      # Flink operators
├── models/         # ML models
├── utils/          # Helper functions
├── config/         # Configuration
└── main.py         # Entry point
```

**Rules**:
- Only production code in `src/`
- All tests in `test/`
- No mixing test files into `src/`
- No temporary scripts in `src/`

## 4. Documentation

**Minimize markdown documents. If needed, put in `docs/superpowers/specs/`**

**Write markdown only for**:
- Specifications: `docs/superpowers/specs/`
- Implementation plans: `docs/superpowers/plans/`

**Do NOT write markdown for**:
- Progress reports → use git commits
- Debugging notes → use code comments
- TODO lists → use inline TODOs
- Random notes → use scratch files

**Use code documentation instead**:
```python
def generate_trip_id(record):
    """Generate unique ID for NYC Taxi trip.
    
    Uses MurmurHash3 for 10-20x faster hashing than MD5.
    """
    
# CRITICAL: Clear Broadcast State before put() - V1.9 Bug #1 fix
broadcast_state.clear()
```

## Anti-Patterns

```python
# ❌ WRONG: Skip AggregateFunction
stream.window(...).process(ProcessWindowFunction())

# ✅ CORRECT: Follow spec
stream.window(...).aggregate(AggregateFunction(), ProcessWindowFunction())
```

```
# ❌ WRONG structure
src/layer1.py
src/test_layer1.py

# ✅ CORRECT structure
src/layer1.py
test/unit/test_layer1.py
```

## Pre-Commit Checklist

- [ ] Code follows spec exactly (no deviations)
- [ ] Tests written and passing
- [ ] Files in correct directories
- [ ] No unnecessary .md documents
- [ ] User approved any deviations

**When in doubt → ASK USER first.**
