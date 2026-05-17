# Skill Routing: When to Use Which Domain Skill

All skills are invoked from `research-from-paper\.cursor\skills\` — **NO external skills**.

## Routing Principle

When you encounter a domain-specific task during research, search the skills library for the right tool. Read the SKILL.md of the relevant skill before starting — it contains workflows, common issues, and production-ready code examples.

---

## Complete Routing Map (verified skills only)

### Literature & Research

| Task | Skill | Location |
|---|---|---|
| Find paper by DOI/URL | paper-lookup | `14-literature-research/paper-lookup/` |
| Systematic literature search | literature-review | `13-research-ideation/literature-review/` |
| Research trend analysis | trend-analyst | `05-research-analysis/trend-analyst/` |
| Competitive landscape | competitive-analyst | `05-research-analysis/competitive-analyst/` |
| Data research | data-researcher | `05-research-analysis/data-researcher/` |
| Search specialist | search-specialist | `05-research-analysis/search-specialist/` |

### Research Ideation & Analysis

| Task | Skill | Location |
|---|---|---|
| Structured ideation | brainstorming-research-ideas | `13-research-ideation/brainstorming-research-ideas/` |
| Creative thinking | creative-thinking-for-research | `13-research-ideation/creative-thinking-for-research/` |
| Discovery process | discovery-process | `13-research-ideation/discovery-process/` |
| Critical thinking | scientific-critical-thinking | `13-research-ideation/scientific-critical-thinking/` |
| Peer review | peer-review | `13-research-ideation/peer-review/` |
| Scientific brainstorming | scientific-brainstorming | `13-research-ideation/scientific-brainstorming/` |

### Statistics & Evaluation

| Task | Skill | Location |
|---|---|---|
| Statistical analysis | statistical-analysis | `00-general-skills/statistical-analysis/` |
| Benchmark evaluation | evaluation | `07-ml/evaluation/` |
| Data quality eval | dataquality-evaluator | `04-evaluation/dataquality-evaluator/` |

### Machine Learning

| Task | Skill | Location |
|---|---|---|
| Scikit-learn models | scikit-learn | `07-ml/scikit-learn/` |
| ML engineering | machine-learning-engineer | `07-ml/machine-learning-engineer/` |
| Data science | data-scientist | `07-ml/data-scientist/` |
| ML training recipes | ml-training-recipes | `08-ml-optimization/ml-training-recipes/` |
| MLops | mlops | `07-ml/mlops/` |
| Figure generation | matplotlib | `07-ml/matplotlib/` |

### Data Engineering

| Task | Skill | Location |
|---|---|---|
| Data pipeline engineering | data-engineer | `02-data-engineering/data-engineer/` |
| SQL optimization | sql-pro | `02-data-engineering/sql-pro/` |
| Database optimization | database-optimizer | `02-data-engineering/database-optimizer/` |
| PostgreSQL | postgres-pro | `02-data-engineering/postgres-pro/` |
| Ray data processing | ray-data | `03-data-processing/ray-data/` |
| Data curation | nemo-curator | `03-data-processing/nemo-curator/` |

### System Design

| Task | Skill | Location |
|---|---|---|
| System architecture | system-architecture | `11-system-design/system-architecture/` |
| LLM architecture | llm-architect | `11-system-design/llm-architect/` |
| Data architecture | data-architect | `11-system-design/data-architect/` |
| Architecture review | architect-reviewer | `11-system-design/architect-reviewer/` |
| Code review | code-reviewer | `11-system-design/code-reviewer/` |

### Engineering & DevOps

| Task | Skill | Location |
|---|---|---|
| Python engineering | python-pro | `06-infrastructure-devops/python-pro/` |
| DevOps | devops-engineer | `06-infrastructure-devops/devops-engineer/` |
| Docker | docker-expert | `06-infrastructure-devops/docker-expert/` |
| Platform engineering | platform-engineer | `06-infrastructure-devops/platform-engineer/` |
| Database admin | database-administrator | `06-infrastructure-devops/database-administrator/` |
| Refactoring | refactoring-specialist | `06-infrastructure-devops/refactoring-specialist/` |
| Code review | code-reviewer | `10-engineering/code-reviewer/` |
| Architect review | architect-reviewer | `10-engineering/architect-reviewer/` |
| Engineering refactoring | refactoring-specialist | `10-engineering/refactoring-specialist/` |

### Experiment Tracking

| Task | Skill | Location |
|---|---|---|
| Weights & Biases | weights-and-biases | `09-mlops/weights-and-biases/` |
| MLflow | mlflow | `09-mlops/mlflow/` |
| TensorBoard | tensorboard | `09-mlops/tensorboard/` |
| SwanLab | swanlab | `09-mlops/swanlab/` |

### Paper Writing

| Task | Skill | Location |
|---|---|---|
| ML paper writing | ml-paper-writing | `12-ml-paper-writing/ml-paper-writing/` |
| Systems paper writing | systems-paper-writing | `12-ml-paper-writing/systems-paper-writing/` |
| Academic paper | academic-paper | `academic-paper/` |
| Academic plotting | academic-plotting | `12-ml-paper-writing/academic-plotting/` |
| Conference talks | presenting-conference-talks | `12-ml-paper-writing/presenting-conference-talks/` |

### General

| Task | Skill | Location |
|---|---|---|
| LaTeX documents | latex-document | `00-general-skills/latex-document/` |
| Scientific writing | scientific-writing | `00-general-skills/scientific-writing/` |
| Academic pipeline | academic-pipeline | `academic-pipeline/` |
| Academic reviewer | academic-paper-reviewer | `academic-paper-reviewer/` |
| Deep research | deep-research | `deep-research/` |

---

## Common Research Workflows

### "I need to evaluate a benchmark"

1. Use `evaluation` skill for standard evaluation workflows
2. Use `dataquality-evaluator` for data quality metrics
3. Track with `mlflow` or `weights-and-biases`

### "I need to process large-scale data"

1. Use `ray-data` for distributed data processing
2. Use `nemo-curator` for data curation and filtering

### "I need to write a research paper"

1. Use `ml-paper-writing` for ML/NLP venues (NeurIPS, ICML, ICLR, ACL)
2. Use `systems-paper-writing` for systems venues (OSDI, NSDI, ASPLOS, SOSP)
3. Generate figures with `matplotlib` or `academic-plotting`
4. Use `latex-document` for compilation

---

## Finding Skills

If you're not sure which skill to use, check `references/skill-routing.md` in the research-from-paper root directory for the complete phase-by-phase mapping, or search by keyword in the skills directory.
