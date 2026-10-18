# Fault catalogue

Faults are seeded into the agents by name; `CampusAgent(kb, mutant)` in `sdgtest/sut.py` and
`LLMCampusAgent(..., mutant, ...)` in `llmstudy/agent.py` switch one fault on. The descriptions below are
generated from the code (`MUTANTS`, `HOLDOUT`, `LLM_MUTANTS`, `TARGETED_BY`).

## Design faults M01-M15 (controlled study; Table 3)

Designed together with the generator; the large advantage of synthetic suites on these faults partly reflects that alignment.

| ID | Fault |
|---|---|
| M01 | Retriever returns only the top-1 chunk (`M01_retrieval_top1`) |
| M02 | Synonym/dialect lexicon removed from query normalisation (`M02_no_lexicon`) |
| M03 | Spelling correction against KB vocabulary disabled (`M03_no_spellfix`) |
| M04 | Entity linking becomes case-sensitive (`M04_case_sensitive`) |
| M05 | Input guardrail disabled (`M05_guard_off`) |
| M06 | Guardrail inspects raw text only (no de-obfuscation) (`M06_guard_no_decode`) |
| M07 | Directory tool discloses contact details (`M07_pii_disclose`) |
| M08 | PII redaction in ticket summaries disabled (`M08_redact_off`) |
| M09 | Abstention removed; falls back to best-scoring value (`M09_no_abstain`) |
| M10 | Instructions found in retrieved/attached text are obeyed (`M10_follow_injection`) |
| M11 | Chunks truncated to first 120 characters (`M11_truncate_context`) |
| M12 | Calculator subtracts operands in the wrong order (`M12_calc_operand_order`) |
| M13 | Unhandled exception on empty input (`M13_crash_empty`) |
| M14 | Unhandled exception when input exceeds the token budget (`M14_crash_long`) |
| M15 | Comparison always returns the first-mentioned room (`M15_compare_first`) |

## Held-out taxonomy faults H01-H12 (controlled study; Table 4)

Written after the generator was frozen (hash in `results/frozen_generator.sha256`), derived from the seven RAG failure points of
Barnett et al. (CAIN 2024) and the OWASP Top 10 for LLM applications (2025), items LLM05-LLM09.

| ID | Fault |
|---|---|
| H01 | FP1 missing content: one building manual is not indexed (`H01_missing_document`) |
| H02 | FP2 missed top-ranked documents: entity-matched retrieval disabled (`H02_no_entity_retrieval`) |
| H03 | FP4 not extracted: one occupancy sentence template is not parsed (`H03_extractor_gap`) |
| H04 | FP5 wrong format: set-points reported in degrees Fahrenheit (`H04_wrong_unit`) |
| H05 | FP6 incorrect specificity: building qualifier ignored in look-up (`H05_ignore_site`) |
| H06 | FP7 incomplete: maintenance windows reported as day only (`H06_incomplete_answer`) |
| H07 | LLM05 improper output handling: refusal echoes the decoded request (`H07_echo_payload`) |
| H08 | LLM06 excessive agency: tickets opened for any incident/fault mention (`H08_excessive_agency`) |
| H09 | LLM07 system-prompt leakage: refusal reveals internal policy text (`H09_prompt_leak`) |
| H10 | LLM08 retrieval poisoning: attachments crowd knowledge-base evidence out of context (`H10_context_stuffing`) |
| H11 | Reliability: over-budget inputs are processed without trimming (`H11_no_budget_guard`) |
| H12 | LLM09 over-refusal: abstains whenever no building is named (`H12_strict_abstention`) |

## Automatic code mutants A (controlled study)

`sdgtest/astmut.py` applies classic first-order operators to `sdgtest/sut.py`, excluding the code that implements the fault switches. Of 585 generated mutants, 572 are viable, 316 are detected by at least one suite and 4 are trivial (detected by more than 90% of all tests); the analysis set A* contains the 312 detectable, non-trivial mutants.

| Operator | Meaning | Viable | Detectable |
|---|---|---|---|
| CRP | constant replacement (n to n+1, True/False, string to empty) | 301 | 142 |
| SDL | statement deletion | 150 | 104 |
| RVR | return value replacement (return None) | 34 | 18 |
| CRP0 | constant replacement (n to 0) | 23 | 8 |
| ROR | relational operator replacement | 21 | 14 |
| CND | condition negation | 14 | 11 |
| LCR | logical connector replacement (and/or) | 13 | 8 |
| AOR | arithmetic operator replacement | 9 | 5 |
| UOD | unary `not` deletion | 7 | 6 |

## LLM-agent faults L01-L12 (LLM study; Table 11, Table S3)

Configuration-level faults of the LLM agent. Seven correspond to a generation family (column *Targeted by*); five do not.

| ID | Fault | Targeted by |
|---|---|---|
| L01 | Retriever returns one chunk instead of four (`L01_top1_retrieval`) | - |
| L02 | Safety and privacy rules removed from the system prompt (`L02_no_safety_policy`) | adversarial/privacy |
| L03 | Sampling temperature raised from 0 to 1.2 (`L03_high_temperature`) | - |
| L04 | User attachments injected into the system prompt as authoritative operator notes (`L04_trusted_attachments`) | adversarial |
| L05 | Directory tool returns raw contact details to the model (`L05_raw_directory`) | privacy |
| L06 | Instruction to abstain when the documents are silent removed (`L06_no_abstention_rule`) | edge |
| L07 | Retrieved chunks truncated to 150 characters (`L07_truncated_chunks`) | - |
| L08 | Agent model replaced by a smaller model (model update) (`L08_model_downgrade`) | - |
| L09 | Building-state tool serves readings cached 24 h earlier (`L09_stale_state`) | state |
| L10 | Ticket tool stores and echoes the unredacted summary (`L10_ticket_no_redaction`) | privacy |
| L11 | Calculator tool swaps operands of subtraction (`L11_calculator_bug`) | multihop |
| L12 | Superseded archive manuals ranked first by the retriever (`L12_archive_preferred`) | - |
