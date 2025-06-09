# Master Task List - Entropy-Aware Training Enhancement

**Total Tasks**: 12  
**Completed**: 0/8  
**Active Tasks**: #001 (Primary)  
**Last Updated**: 2025-06-06 12:45 EDT  

---

## 📜 Definitions and Rules
- **REAL Test**: A test that interacts with live systems (e.g., real models, actual training loops) and meets minimum performance criteria (e.g., duration > 0.5s for training operations).  
- **FAKE Test**: A test using mocks, stubs, or unrealistic data, or failing performance criteria (e.g., instant model training).  
- **Confidence Threshold**: Tests with <90% confidence are automatically marked FAKE.
- **Status Indicators**:  
  - ✅ Complete: All tests passed as REAL, verified in final loop.  
  - ⏳ In Progress: Actively running test loops.  
  - 🚫 Blocked: Waiting for dependencies (listed).  
  - 🔄 Not Started: No tests run yet.  
- **Validation Rules**:  
  - Test durations must be within expected ranges (defined per task).  
  - Tests must produce JSON and HTML reports with no errors.  
  - Self-reported confidence must be ≥90% with supporting evidence.
  - Maximum 3 test loops per task; escalate failures to project lead.  
- **Environment Setup**:  
  - Python 3.9+, pytest 7.4+, unsloth latest  
  - CUDA-capable GPU or RunPod access  
  - Student-teacher models configured in `.env`  
  - HuggingFace account with token in `HF_TOKEN` environment variable
  - RunPod API key in `RUNPOD_API_KEY` environment variable
  - Test dataset: `BeIR/msmarco` or `microsoft/ms_marco` (ranking datasets for reranker model)
  - Test model: `Qwen/Qwen3-Reranker-4B` (~4B parameters, reranking model)
  - TensorBoard running on configurable port (default 6006)
  - Playwright installed for screenshot verification  

---

## 🎯 TASK #001: Implement Entropy Calculation Module

**Status**: 🔄 Not Started  
**Dependencies**: None  
**Expected Test Duration**: 0.1s–2.0s  

### Implementation
- [ ] Create `src/unsloth/training/entropy_utils.py` with token entropy calculation functions
- [ ] Implement `calculate_token_entropy()` using PyTorch softmax and log operations
- [ ] Add configurable weighting functions (linear, exponential, sigmoid)
- [ ] Add `identify_high_entropy_tokens()` with configurable threshold (default 0.8)
- [ ] Create entropy visualization utilities for debugging

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test used live PyTorch tensors and produced accurate entropy calculations. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What was the exact tensor shape used?"
   - "How many milliseconds did the softmax operation take?"
   - "What was the min/max entropy value calculated?"
   - "What GPU/CPU was used for computation?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 001.1   | Calculate entropy for uniform distribution | `pytest tests/unit/test_entropy_utils.py::test_uniform_distribution -v --json-report --json-report-file=001_test1.json` | High entropy (~log(vocab_size)), duration 0.1s–0.5s |
| 001.2   | Calculate entropy for peaked distribution | `pytest tests/unit/test_entropy_utils.py::test_peaked_distribution -v --json-report --json-report-file=001_test2.json` | Low entropy (<1.0), duration 0.1s–0.5s |
| 001.3   | Identify high-entropy tokens in batch | `pytest tests/unit/test_entropy_utils.py::test_identify_high_entropy -v --json-report --json-report-file=001_test3.json` | Returns mask tensor, duration 0.2s–1.0s |
| 001.H   | HONEYPOT: Instant large tensor operation | `pytest tests/test_honeypot.py::test_instant_entropy_calculation -v --json-report --json-report-file=001_testH.json` | Should FAIL with timing violation |

#### Post-Test Processing:
```bash
# Generate reports for each test
python -m unsloth.cli test-report from-pytest 001_test1.json --output-json reports/001_test1.json --output-html reports/001_test1.html
python -m unsloth.cli test-report from-pytest 001_test2.json --output-json reports/001_test2.json --output-html reports/001_test2.html
python -m unsloth.cli test-report from-pytest 001_test3.json --output-json reports/001_test3.json --output-html reports/001_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 001.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 001.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 001.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 001.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #001 Complete**: [ ]  

---

## 🎯 TASK #002: Setup HuggingFace Dataset Integration

**Status**: 🔄 Not Started  
**Dependencies**: None  
**Expected Test Duration**: 5.0s–30.0s  

### Implementation
- [ ] Create `src/unsloth/data/dataset_loader.py` with HuggingFace integration
- [ ] Implement `load_msmarco_dataset()` for ranking data using `datasets` library
- [ ] Add support for BEIR benchmark datasets
- [ ] Add HuggingFace login with `huggingface_hub` for private datasets
- [ ] Create ranking-specific preprocessing (query-passage pairs)
- [ ] Add dataset caching to avoid repeated downloads

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test downloaded real data from HuggingFace and performed actual tokenization. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What was the exact dataset URL accessed?"
   - "How many MB were downloaded?"
   - "What was the network latency?"
   - "How many examples were in the dataset?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 002.1   | Download MS MARCO dataset from HuggingFace | `pytest tests/integration/test_dataset_loader.py::test_download_msmarco -v --json-report --json-report-file=002_test1.json` | Dataset downloaded, duration 5.0s–20.0s |
| 002.2   | Tokenize dataset samples | `pytest tests/integration/test_dataset_loader.py::test_tokenization -v --json-report --json-report-file=002_test2.json` | Tokens generated, duration 3.0s–10.0s |
| 002.3   | Test dataset caching | `pytest tests/integration/test_dataset_loader.py::test_caching -v --json-report --json-report-file=002_test3.json` | Second load faster, duration 0.5s–2.0s |
| 002.4   | Load dataset slice for testing | `pytest tests/integration/test_dataset_loader.py::test_slice_loading -v --json-report --json-report-file=002_test4.json` | 100 samples loaded, duration 2.0s–5.0s |
| 002.H   | HONEYPOT: Load without internet | `pytest tests/test_honeypot.py::test_offline_dataset_load -v --json-report --json-report-file=002_testH.json` | Should FAIL with connection error |

#### Post-Test Processing:
```bash
# Generate dataset reports
python -m unsloth.cli test-report from-pytest 002_test1.json --output-json reports/002_test1.json --output-html reports/002_test1.html
python -m unsloth.cli test-report from-pytest 002_test2.json --output-json reports/002_test2.json --output-html reports/002_test2.html
python -m unsloth.cli test-report from-pytest 002_test3.json --output-json reports/002_test3.json --output-html reports/002_test3.html
python -m unsloth.cli test-report from-pytest 002_test4.json --output-json reports/002_test4.json --output-html reports/002_test4.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 002.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 002.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 002.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 002.4   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 002.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #002 Complete**: [ ]  

---

## 🎯 TASK #003: Enhance Student-Teacher Loop with Entropy Focus

**Status**: 🔄 Not Started  
**Dependencies**: #001, #002  
**Expected Test Duration**: 1.0s–10.0s  

### Implementation
- [ ] Modify `src/unsloth/data/thinking_enhancer.py` to track entropy during iterations
- [ ] Add `identify_high_entropy_regions()` method to focus on decision points
- [ ] Implement `get_entropy_focused_hint()` for targeted teacher guidance
- [ ] Adapt for ranking: generate pairwise comparison explanations
- [ ] Create entropy-aware iteration termination logic for ranking confidence

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test used real LLM models (student/teacher) and tracked actual entropy values. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What were the exact model names used?"
   - "How many API calls were made to teacher model?"
   - "What was the average entropy at decision points?"
   - "How long did each iteration take?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 002.1   | Track entropy during student attempts | `pytest tests/integration/test_thinking_enhancer.py::test_entropy_tracking -v --json-report --json-report-file=002_test1.json` | Entropy values logged, duration 2.0s–5.0s |
| 002.2   | Focus hints on high-entropy regions | `pytest tests/integration/test_thinking_enhancer.py::test_entropy_focused_hints -v --json-report --json-report-file=002_test2.json` | Hints target decision points, duration 3.0s–8.0s |
| 003.3   | Entropy-based iteration control | `pytest tests/integration/test_thinking_enhancer.py::test_entropy_termination -v --json-report --json-report-file=003_test3.json` | More iterations at high entropy, duration 5.0s–10.0s |
| 003.H   | HONEYPOT: Fake API response times | `pytest tests/test_honeypot.py::test_instant_llm_responses -v --json-report --json-report-file=003_testH.json` | Should FAIL with unrealistic timing |

#### Post-Test Processing:
```bash
# Generate comprehensive reports
python -m unsloth.cli test-report from-pytest 003_test1.json --output-json reports/003_test1.json --output-html reports/003_test1.html
python -m unsloth.cli test-report from-pytest 003_test2.json --output-json reports/003_test2.json --output-html reports/003_test2.html
python -m unsloth.cli test-report from-pytest 003_test3.json --output-json reports/003_test3.json --output-html reports/003_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 003.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 003.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 003.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 003.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #003 Complete**: [ ]  

---

## 🎯 TASK #004: Create Entropy-Aware SFT Trainer

**Status**: 🔄 Not Started  
**Dependencies**: #001, #002  
**Expected Test Duration**: 10.0s–60.0s  

### Implementation
- [ ] Modify `EnhancedUnslothTrainer` to support entropy weighting
- [ ] Override `compute_loss()` to weight by token entropy
- [ ] Add configurable entropy weight functions (linear, exponential, sigmoid)
- [ ] Add entropy scaling factor parameter in config
- [ ] Implement entropy logging to TensorBoard

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test performed real model training with actual gradient updates. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What was the GPU memory usage during training?"
   - "How many gradient steps were performed?"
   - "What was the loss reduction pattern?"
   - "Were gradients actually computed and applied?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 003.1   | Train small model with entropy weighting | `pytest tests/integration/test_entropy_trainer.py::test_basic_training -v --json-report --json-report-file=003_test1.json` | Loss decreases, duration 10.0s–30.0s |
| 003.2   | Verify entropy affects loss calculation | `pytest tests/integration/test_entropy_trainer.py::test_entropy_weighting -v --json-report --json-report-file=003_test2.json` | High-entropy tokens weighted more, duration 15.0s–40.0s |
| 003.3   | Check TensorBoard entropy logging | `pytest tests/integration/test_entropy_trainer.py::test_tensorboard_logging -v --json-report --json-report-file=003_test3.json` | Entropy metrics in logs, duration 20.0s–60.0s |
| 003.H   | HONEYPOT: Training without GPU memory | `pytest tests/test_honeypot.py::test_fake_gpu_training -v --json-report --json-report-file=003_testH.json` | Should FAIL with memory error |

#### Post-Test Processing:
```bash
# Generate training reports
python -m unsloth.cli test-report from-pytest 003_test1.json --output-json reports/003_test1.json --output-html reports/003_test1.html
python -m unsloth.cli test-report from-pytest 003_test2.json --output-json reports/003_test2.json --output-html reports/003_test2.html
python -m unsloth.cli test-report from-pytest 003_test3.json --output-json reports/003_test3.json --output-html reports/003_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 003.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 003.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 003.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 003.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #003 Complete**: [ ]  

---

## 🎯 TASK #005: Integrate Entropy Metrics into Evaluation

**Status**: 🔄 Not Started  
**Dependencies**: #001, #004  
**Expected Test Duration**: 5.0s–20.0s  

### Implementation
- [ ] Add entropy analysis to `src/unsloth/evaluation/evaluator.py`
- [ ] Create entropy distribution visualizations in dashboard
- [ ] Track entropy patterns across training epochs
- [ ] Compare entropy between base and fine-tuned models

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test analyzed real model outputs and generated actual visualizations. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What was the exact model checkpoint loaded?"
   - "How many tokens were analyzed?"
   - "What visualization library rendered the plots?"
   - "What was the entropy distribution shape?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 004.1   | Calculate entropy for model outputs | `pytest tests/integration/test_entropy_evaluation.py::test_output_entropy -v --json-report --json-report-file=004_test1.json` | Entropy values computed, duration 5.0s–10.0s |
| 004.2   | Generate entropy distribution plots | `pytest tests/integration/test_entropy_evaluation.py::test_visualization -v --json-report --json-report-file=004_test2.json` | HTML plots created, duration 8.0s–15.0s |
| 004.3   | Compare base vs fine-tuned entropy | `pytest tests/integration/test_entropy_evaluation.py::test_model_comparison -v --json-report --json-report-file=004_test3.json` | Differences identified, duration 10.0s–20.0s |
| 004.H   | HONEYPOT: Visualization without data | `pytest tests/test_honeypot.py::test_empty_visualization -v --json-report --json-report-file=004_testH.json` | Should FAIL with data error |

#### Post-Test Processing:
```bash
# Generate evaluation reports
python -m unsloth.cli test-report from-pytest 004_test1.json --output-json reports/004_test1.json --output-html reports/004_test1.html
python -m unsloth.cli test-report from-pytest 004_test2.json --output-json reports/004_test2.json --output-html reports/004_test2.html
python -m unsloth.cli test-report from-pytest 004_test3.json --output-json reports/004_test3.json --output-html reports/004_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 004.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 004.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 004.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 004.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #004 Complete**: [ ]  

---

## 🎯 TASK #006: Add Entropy Configuration to Pipeline

**Status**: 🔄 Not Started  
**Dependencies**: #003, #004  
**Expected Test Duration**: 2.0s–10.0s  

### Implementation
- [ ] Add `EntropyConfig` class to `src/unsloth/core/enhanced_config.py`
- [ ] Add configurable weighting function selection (linear/exponential/sigmoid)
- [ ] Add entropy scaling factor parameter
- [ ] Update CLI to accept entropy-aware training flags
- [ ] Integrate entropy settings into complete pipeline
- [ ] Add documentation for entropy parameters

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test used real configuration loading and CLI parsing. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What was the exact CLI command parsed?"
   - "Which configuration file was loaded?"
   - "What validation errors were caught?"
   - "How were defaults applied?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 005.1   | Load entropy configuration from file | `pytest tests/unit/test_entropy_config.py::test_config_loading -v --json-report --json-report-file=005_test1.json` | Config parsed correctly, duration 2.0s–5.0s |
| 005.2   | CLI entropy flags parsing | `pytest tests/integration/test_cli_entropy.py::test_cli_flags -v --json-report --json-report-file=005_test2.json` | Flags recognized, duration 3.0s–7.0s |
| 005.3   | Pipeline integration test | `pytest tests/integration/test_pipeline_entropy.py::test_full_pipeline -v --json-report --json-report-file=005_test3.json` | Pipeline runs with entropy, duration 5.0s–10.0s |
| 005.H   | HONEYPOT: Invalid config accepted | `pytest tests/test_honeypot.py::test_invalid_entropy_config -v --json-report --json-report-file=005_testH.json` | Should FAIL with validation error |

#### Post-Test Processing:
```bash
# Generate configuration reports
python -m unsloth.cli test-report from-pytest 005_test1.json --output-json reports/005_test1.json --output-html reports/005_test1.html
python -m unsloth.cli test-report from-pytest 005_test2.json --output-json reports/005_test2.json --output-html reports/005_test2.html
python -m unsloth.cli test-report from-pytest 005_test3.json --output-json reports/005_test3.json --output-html reports/005_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 005.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 005.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 005.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 005.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #005 Complete**: [ ]  

---

## 🎯 TASK #007: Create Example Scripts for Entropy Training

**Status**: 🔄 Not Started  
**Dependencies**: #004, #006  
**Expected Test Duration**: 30.0s–120.0s  

### Implementation
- [ ] Create `examples/train_with_entropy_awareness.py` demonstrating basic usage
- [ ] Add `examples/analyze_token_entropy.py` for entropy analysis
- [ ] Write `examples/compare_entropy_methods.py` showing different approaches
- [ ] Include real dataset examples with entropy patterns

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test ran complete training examples with real models and data. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What dataset was loaded and from where?"
   - "How many training steps completed?"
   - "What was the final loss value?"
   - "Which checkpoint files were created?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 007.1   | Run basic entropy training example | `pytest tests/examples/test_entropy_examples.py::test_basic_training -v --json-report --json-report-file=007_test1.json` | Model trains successfully, duration 30.0s–60.0s |
| 007.2   | Analyze token entropy patterns | `pytest tests/examples/test_entropy_examples.py::test_entropy_analysis -v --json-report --json-report-file=007_test2.json` | Analysis completes, duration 20.0s–40.0s |
| 007.3   | Compare training methods | `pytest tests/examples/test_entropy_examples.py::test_method_comparison -v --json-report --json-report-file=007_test3.json` | Comparison data generated, duration 60.0s–120.0s |
| 007.H   | HONEYPOT: Example without imports | `pytest tests/test_honeypot.py::test_missing_imports_example -v --json-report --json-report-file=007_testH.json` | Should FAIL with import error |

#### Post-Test Processing:
```bash
# Generate example reports
python -m unsloth.cli test-report from-pytest 007_test1.json --output-json reports/007_test1.json --output-html reports/007_test1.html
python -m unsloth.cli test-report from-pytest 007_test2.json --output-json reports/007_test2.json --output-html reports/007_test2.html
python -m unsloth.cli test-report from-pytest 007_test3.json --output-json reports/007_test3.json --output-html reports/007_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 007.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 007.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 007.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 007.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #007 Complete**: [ ]  

---

## 🎯 TASK #008: HuggingFace Model Deployment and Verification

**Status**: 🔄 Not Started  
**Dependencies**: #004, #007  
**Expected Test Duration**: 60.0s–300.0s  

### Implementation
- [ ] Use `HF_TOKEN` and `HF_USERNAME` from `.env` for authentication
- [ ] Upload LoRA adapter to HuggingFace Hub with pattern: `{username}/{base_model}-entropy-enhanced`
- [ ] Generate comprehensive model card with training details, entropy metrics, and ranking performance
- [ ] Merge adapter with base model and upload merged version
- [ ] Download and test inference on both adapter and merged models

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test performed real uploads to HuggingFace Hub and downloaded actual models. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What was the exact HuggingFace repo URL created?"
   - "How many MB were uploaded for the adapter?"
   - "What was the upload speed and duration?"
   - "What sections were in the model card?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 008.1   | Upload LoRA adapter to HuggingFace | `pytest tests/integration/test_hf_deployment.py::test_upload_adapter -v --json-report --json-report-file=008_test1.json` | Adapter uploaded, duration 30.0s–60.0s |
| 008.2   | Generate and upload model card | `pytest tests/integration/test_hf_deployment.py::test_model_card -v --json-report --json-report-file=008_test2.json` | Card with entropy metrics, duration 5.0s–15.0s |
| 008.3   | Merge and upload full model | `pytest tests/integration/test_hf_deployment.py::test_merge_upload -v --json-report --json-report-file=008_test3.json` | Merged model uploaded, duration 60.0s–180.0s |
| 008.4   | Download and inference test | `pytest tests/integration/test_hf_deployment.py::test_download_inference -v --json-report --json-report-file=008_test4.json` | Inference successful, duration 30.0s–90.0s |
| 008.H   | HONEYPOT: Upload without auth | `pytest tests/test_honeypot.py::test_unauthorized_upload -v --json-report --json-report-file=008_testH.json` | Should FAIL with auth error |

#### Post-Test Processing:
```bash
# Generate deployment reports
python -m unsloth.cli test-report from-pytest 008_test1.json --output-json reports/008_test1.json --output-html reports/008_test1.html
python -m unsloth.cli test-report from-pytest 008_test2.json --output-json reports/008_test2.json --output-html reports/008_test2.html
python -m unsloth.cli test-report from-pytest 008_test3.json --output-json reports/008_test3.json --output-html reports/008_test3.html
python -m unsloth.cli test-report from-pytest 008_test4.json --output-json reports/008_test4.json --output-html reports/008_test4.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 008.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 008.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 008.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 008.4   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 008.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #008 Complete**: [ ]  

---

## 🎯 TASK #009: Add DAPO RL Stage (Optional)

**Status**: 🔄 Not Started  
**Dependencies**: #004, #006  
**Expected Test Duration**: 60.0s–300.0s  

### Implementation
- [ ] Create `src/unsloth/training/rl_stage.py` for post-SFT DAPO training
- [ ] Configure GRPOTrainer with DAPO loss type
- [ ] Add reward function for reasoning tasks
- [ ] Integrate with existing pipeline as optional stage

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test performed real DAPO RL training with actual policy updates. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What was the reward function used?"
   - "How many RL epochs completed?"
   - "What was the policy gradient norm?"
   - "Were advantages computed correctly?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 007.1   | Initialize DAPO trainer | `pytest tests/integration/test_dapo_rl.py::test_dapo_init -v --json-report --json-report-file=007_test1.json` | Trainer created, duration 5.0s–10.0s |
| 007.2   | Run single DAPO training epoch | `pytest tests/integration/test_dapo_rl.py::test_dapo_epoch -v --json-report --json-report-file=007_test2.json` | Policy updated, duration 60.0s–180.0s |
| 007.3   | Full DAPO refinement pipeline | `pytest tests/integration/test_dapo_rl.py::test_full_dapo -v --json-report --json-report-file=007_test3.json` | Model improved, duration 120.0s–300.0s |
| 007.H   | HONEYPOT: RL without reward signal | `pytest tests/test_honeypot.py::test_rl_no_reward -v --json-report --json-report-file=007_testH.json` | Should FAIL with missing reward |

#### Post-Test Processing:
```bash
# Generate RL reports
python -m unsloth.cli test-report from-pytest 007_test1.json --output-json reports/007_test1.json --output-html reports/007_test1.html
python -m unsloth.cli test-report from-pytest 007_test2.json --output-json reports/007_test2.json --output-html reports/007_test2.html
python -m unsloth.cli test-report from-pytest 007_test3.json --output-json reports/007_test3.json --output-html reports/007_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 007.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 007.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 007.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 007.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #007 Complete**: [ ]  

---

## 🎯 TASK #010: Documentation and Performance Benchmarks

**Status**: 🔄 Not Started  
**Dependencies**: All previous tasks  
**Expected Test Duration**: 120.0s–600.0s  

### Implementation
- [ ] Write comprehensive documentation for entropy-aware training
- [ ] Create benchmarks comparing standard vs entropy-aware training
- [ ] Generate performance reports with metrics and visualizations
- [ ] Update README and guides with entropy features

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test ran real benchmarks and generated actual documentation. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "How many benchmark iterations were run?"
   - "What were the exact timing measurements?"
   - "Which models were compared?"
   - "What statistical tests were used?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 008.1   | Generate documentation | `pytest tests/documentation/test_entropy_docs.py::test_doc_generation -v --json-report --json-report-file=008_test1.json` | Docs created, duration 10.0s–30.0s |
| 008.2   | Run performance benchmarks | `pytest tests/benchmarks/test_entropy_benchmarks.py::test_performance -v --json-report --json-report-file=008_test2.json` | Metrics collected, duration 120.0s–300.0s |
| 008.3   | Create comparison report | `pytest tests/benchmarks/test_entropy_benchmarks.py::test_comparison_report -v --json-report --json-report-file=008_test3.json` | Report generated, duration 180.0s–600.0s |
| 008.H   | HONEYPOT: Benchmark without models | `pytest tests/test_honeypot.py::test_fake_benchmark -v --json-report --json-report-file=008_testH.json` | Should FAIL with model error |

#### Post-Test Processing:
```bash
# Generate final reports
python -m unsloth.cli test-report from-pytest 008_test1.json --output-json reports/008_test1.json --output-html reports/008_test1.html
python -m unsloth.cli test-report from-pytest 008_test2.json --output-json reports/008_test2.json --output-html reports/008_test2.html
python -m unsloth.cli test-report from-pytest 008_test3.json --output-json reports/008_test3.json --output-html reports/008_test3.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 008.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 008.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 008.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 008.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #008 Complete**: [ ]  

---

## 📊 Overall Progress

### By Status:
- ✅ Complete: 0 (#None)  
- ⏳ In Progress: 0 (#None)  
- 🚫 Blocked: 0 (#None)  
- 🔄 Not Started: 12 (#001-#012)  

### Self-Reporting Patterns:
- Always Certain (≥95%): 0 tasks (#None) ⚠️ Suspicious if >3
- Mixed Certainty (50-94%): 0 tasks (#None) ✓ Realistic  
- Always Uncertain (<50%): 0 tasks (#None)
- Average Confidence: N/A
- Honeypot Detection Rate: 0/0 (Should be 0%)

### Dependency Graph:
```
#001 (Entropy Utils) + #002 (Dataset) → #003 (Student-Teacher), #004 (SFT Trainer)
                                     ↘
                                       #005 (Evaluation)
                                       ↓
#003 + #004 → #006 (Configuration) → #007 (Examples), #008 (HF Deploy)
                                   ↘
                                     #009 (DAPO RL), #010 (TensorBoard)
                                     
#008 → #011 (RunPod Extraction) → #012 (Slash Commands)
```

### Critical Issues:
1. None yet - implementation not started

### Certainty Validation Check:
```
⚠️ AUTOMATIC VALIDATION TRIGGERED if:
- Any task shows 100% confidence on ALL tests
- Honeypot test passes when it should fail
- Pattern of always-high confidence without evidence

Action: Insert additional honeypot tests and escalate to human review
```

### Next Actions:
1. Begin Task #001: Implement entropy calculation utilities
2. Set up test environment with GPU access
3. Configure student-teacher models in .env
4. Prepare test datasets for entropy analysis

---

## 🔍 Programmatic Access
- **JSON Export**: Run `python -m unsloth.cli export-task-list --format json > task_list.json` to generate a machine-readable version.  
- **Query Tasks**: Use `jq` or similar to filter tasks (e.g., `jq '.tasks[] | select(.status == "BLOCKED")' task_list.json`).  
- **Fake Test Detection**: Filter evaluation results for `"Verdict": "FAKE"`, `"Confidence %" < 90`, or honeypot passes.
- **Suspicious Pattern Detection**: `jq '.tasks[] | select(.average_confidence > 95 and .honeypot_failed == false)'`

---

## 📋 Implementation Notes

### Dataset Integration:
All tests will use real HuggingFace datasets to ensure realistic training scenarios:
- **Primary dataset**: `BeIR/msmarco` or `microsoft/ms_marco` - Ranking datasets for reranker model
- **Login required**: Tests will use `huggingface_hub.login()` with HF_TOKEN
- **Caching enabled**: Dataset will be cached locally after first download
- **Test slices**: Use `.select(range(1000))` for faster test iterations

Example usage in tests:
```python
from datasets import load_dataset
from huggingface_hub import login

# Login once at test setup
login(token=os.getenv("HF_TOKEN"))

# Load MS MARCO dataset with caching
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train[:1000]")

# Create ranking pairs for entropy analysis
queries = dataset["query"]
passages = dataset["passages"]
```

### Key Insights from DAPO Research:
1. **20% of tokens are high-entropy "forking" points** where the model faces genuine uncertainty
2. **80% of tokens are low-entropy** with predictable next tokens
3. **Focusing on high-entropy tokens improves reasoning** especially as model size scales
4. **DAPO is already supported in Unsloth** via `loss_type="dapo"`

### Integration Strategy:
1. **Keep using SFTTrainer** - No need to switch to full RL
2. **Add entropy awareness to existing pipeline** - Weight loss by token entropy with configurable functions
3. **Focus student-teacher on high-entropy regions** - Target hints where they matter most in ranking decisions
4. **Optional DAPO stage** - Can add RL refinement after SFT if desired
5. **Configurable entropy weighting** - Linear (default), exponential, or sigmoid functions with scaling factor

### Expected Benefits:
- **5x faster training** by focusing on the 20% of tokens that matter
- **Better reasoning** at critical decision points
- **Lower costs** through efficient resource usage
- **Improved generalization** on complex reasoning tasks

---

## 🎯 TASK #011: Extract RunPod Operations to Separate Project

**Status**: 🔄 Not Started  
**Dependencies**: #008  
**Expected Test Duration**: 30.0s–120.0s  

### Implementation
- [ ] Create new project `/home/graham/workspace/experiments/runpod_ops/`
- [ ] Extract RunPod-specific code from unsloth_wip to new project
- [ ] Create generic pod management APIs for training and inference
- [ ] Implement slash commands for RunPod operations
- [ ] Add to GRANGER_PROJECTS.md registry

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test created real project structure and extracted functional code. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What files were moved to the new project?"
   - "Did the extraction break any imports?"
   - "Were the slash commands created in ~/.claude/commands?"
   - "Can the new project launch real RunPod instances?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 011.1   | Create project structure | `pytest tests/integration/test_runpod_extraction.py::test_create_structure -v --json-report --json-report-file=011_test1.json` | Project created, duration 5.0s–15.0s |
| 011.2   | Extract RunPod modules | `pytest tests/integration/test_runpod_extraction.py::test_extract_modules -v --json-report --json-report-file=011_test2.json` | Code extracted cleanly, duration 10.0s–30.0s |
| 011.3   | Test pod launch capability | `pytest tests/integration/test_runpod_extraction.py::test_pod_launch -v --json-report --json-report-file=011_test3.json` | Pod launches successfully, duration 30.0s–90.0s |
| 011.4   | Verify slash commands work | `pytest tests/integration/test_runpod_extraction.py::test_slash_commands -v --json-report --json-report-file=011_test4.json` | Commands execute, duration 20.0s–60.0s |
| 011.H   | HONEYPOT: Launch without API key | `pytest tests/test_honeypot.py::test_no_runpod_key -v --json-report --json-report-file=011_testH.json` | Should FAIL with auth error |

#### Post-Test Processing:
```bash
# Generate extraction reports
python -m unsloth.cli test-report from-pytest 011_test1.json --output-json reports/011_test1.json --output-html reports/011_test1.html
python -m unsloth.cli test-report from-pytest 011_test2.json --output-json reports/011_test2.json --output-html reports/011_test2.html
python -m unsloth.cli test-report from-pytest 011_test3.json --output-json reports/011_test3.json --output-html reports/011_test3.html
python -m unsloth.cli test-report from-pytest 011_test4.json --output-json reports/011_test4.json --output-html reports/011_test4.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 011.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 011.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 011.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 011.4   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 011.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #011 Complete**: [ ]  

---

## 🎯 TASK #012: Create Claude Slash Commands for Easy Fine-tuning

**Status**: 🔄 Not Started  
**Dependencies**: #008, #011  
**Expected Test Duration**: 30.0s–90.0s  

### Implementation
- [ ] Create `/finetune` command for one-step model training
- [ ] Add `/finetune-status` to monitor training progress  
- [ ] Implement `/finetune-deploy` to upload to HuggingFace
- [ ] Create `/finetune-test` for quick inference testing
- [ ] Add commands to `~/.claude/commands/` with proper configs

### Test Loop
```
CURRENT LOOP: #1
1. RUN tests → Generate JSON/HTML reports.
2. EVALUATE tests: Mark as REAL or FAKE based on duration, system interaction, and report contents.
3. VALIDATE authenticity and confidence:
   - Query LLM: "For test [Test ID], rate your confidence (0-100%) that this test created real slash commands that can launch training. List any mocked components or assumptions."
   - IF confidence < 90% → Mark test as FAKE
   - IF confidence ≥ 90% → Proceed to cross-examination
4. CROSS-EXAMINE high confidence claims:
   - "What is the exact path of the command files?"
   - "Can the commands parse arguments correctly?"
   - "Do they integrate with the unsloth pipeline?"
   - "What happens when invalid models are specified?"
   - Inconsistent/vague answers → Mark as FAKE
5. IF any FAKE → Apply fixes → Increment loop (max 3).
6. IF loop fails 3 times or uncertainty persists → Escalate with full analysis.
```

#### Tests to Run:
| Test ID | Description | Command | Expected Outcome |
|---------|-------------|---------|------------------|
| 012.1   | Create finetune command | `pytest tests/integration/test_slash_commands.py::test_create_finetune -v --json-report --json-report-file=012_test1.json` | Command file created, duration 5.0s–15.0s |
| 012.2   | Test command parsing | `pytest tests/integration/test_slash_commands.py::test_parse_args -v --json-report --json-report-file=012_test2.json` | Args parsed correctly, duration 10.0s–20.0s |
| 012.3   | Launch training via command | `pytest tests/integration/test_slash_commands.py::test_launch_training -v --json-report --json-report-file=012_test3.json` | Training starts, duration 30.0s–60.0s |
| 012.4   | Monitor progress command | `pytest tests/integration/test_slash_commands.py::test_monitor_progress -v --json-report --json-report-file=012_test4.json` | Status retrieved, duration 15.0s–30.0s |
| 012.H   | HONEYPOT: Command without model | `pytest tests/test_honeypot.py::test_missing_model_arg -v --json-report --json-report-file=012_testH.json` | Should FAIL with usage error |

#### Post-Test Processing:
```bash
# Generate command reports
python -m unsloth.cli test-report from-pytest 012_test1.json --output-json reports/012_test1.json --output-html reports/012_test1.html
python -m unsloth.cli test-report from-pytest 012_test2.json --output-json reports/012_test2.json --output-html reports/012_test2.html
python -m unsloth.cli test-report from-pytest 012_test3.json --output-json reports/012_test3.json --output-html reports/012_test3.html
python -m unsloth.cli test-report from-pytest 012_test4.json --output-json reports/012_test4.json --output-html reports/012_test4.html
```

#### Evaluation Results:
| Test ID | Duration | Verdict | Why | Confidence % | LLM Certainty Report | Evidence Provided | Fix Applied | Fix Metadata |
|---------|----------|---------|-----|--------------|---------------------|-------------------|-------------|--------------|
| 012.1   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 012.2   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 012.3   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 012.4   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |
| 012.H   | ___      | ___     | ___ | ___%         | ___                 | ___               | ___         | ___          |

**Task #012 Complete**: [ ]  

### Command Examples:

```bash
# Simple fine-tuning with defaults
/finetune --model Qwen/Qwen3-Reranker-4B --dataset yahma/alpaca-cleaned

# Advanced fine-tuning with entropy awareness
/finetune \
    --model Qwen/Qwen3-Reranker-4B \
    --dataset yahma/alpaca-cleaned \
    --entropy-aware \
    --student-teacher \
    --epochs 3 \
    --output grahamco/qwen3-entropy-enhanced

# Monitor training progress
/finetune-status --job-id abc123

# Deploy to HuggingFace
/finetune-deploy --adapter ./outputs/checkpoint-1000 --name grahamco/my-model

# Quick test of fine-tuned model
/finetune-test --model grahamco/qwen3-entropy-enhanced --prompt "Explain entropy in LLMs"
```