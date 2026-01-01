---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
  - _bmad-output/project-context.md
  - docs/README.md
---

# vectormesh - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for vectormesh, decomposing the requirements from the PRD, UX Design if it exists, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

FR1: Implement `TextVectorizer` wrapper for HuggingFace models with uniform interface and auto-detection.
FR2: Implement `RegexVectorizer` for creating binary vectors from pattern matches.
FR3: Implement `VectorCache` with chunk-level storage (2DTensor) using HuggingFace Datasets (Parquet/Arrow).
FR4: Implement `VectorMeshComponent` base class with Pydantic validation (v2+).
FR5: Implement Typed Tensor system (`OneDTensor`, `TwoDTensor`, `ThreeDTensor`) using `jaxtyping` + `beartype`.
FR6: Implement Combinators: `Serial`, `Parallel`, `Branch` (Trax-style architecture).
FR7: Implement syntactic sugar `>>` for linear sequencing (mapping to `Serial`).
FR8: Implement `concat` and `stack` connectors with shape validation.
FR9: Implement `VectorCache` persistence: atomic creation, content-hash versioning, `.vmcache` directory structure.
FR10: Implement `DataLoader` that handles variable chunk sizes, automatic padding, and attention masks.
FR11: Implement Parameter-free aggregation strategies (`mean`, `max` pooling).
FR12: Implement robust device management (auto-detect GPU/MPS/CPU) with user override.
FR13: Implement Model Zoo registry (`src/vectormesh/zoo`) with metadata for the 10 supported models.
FR14: Support `hyperopt` integration for search spaces (replacing Ray Tune for Windows compatibility).
FR15: Implement Visualization utilities (`visualize.py`) for component graphs/shapes.
FR16: Implement basic Gating components (`Gate`, `GateSkip`, `ContextGate`).
FR17: Implement VectorMesh MCP Server to expose vectorization tools to AI agents.

### NonFunctional Requirements

NFR1: **Type Safety**: All components must enforce input/output contracts at definition time (not just runtime).
NFR2: **Hardware Democratization**: Experimentation (composition/aggregation) must run efficiently on CPU (80% use case).
NFR3: **Educational UX**: Error messages must be custom (`VectorMeshError`), actionable, and provide `hint` and `fix` suggestions.
NFR4: **Performance**: Zero-copy loading of cached vectors (memory-mapping via HF Datasets).
NFR5: **Code Quality**: 100% type coverage, strict linting (Ruff), Google-style docstrings.
NFR6: **Security**: Safe model loading (`weights_only=True`) to prevent arbitrary code execution.
NFR7: **Usability**: Interactive error feedback guiding the user to the correct shapes/types.

### Additional Requirements

AR1: **Project Structure**: Strict `src/vectormesh` layout managed by `uv`.
AR2: **Testing**: Unit tests must mock network (no real HF calls); Integration tests use `vcrpy`. Tests must mirror `src/` structure.
AR3: **Caching Protocol**: `.vmcache` must be a directory of Parquet files (HF Dataset format) for robust cross-platform sharing.
AR4: **Implementation Pattern**: Functional configuration (frozen Pydantic models). State changes return new objects.
AR5: **Composition Syntax**: `>>` operator must compile to `Serial` container; no implicit chaining.
AR6: **Anti-Pattern (Shapes)**: No implicit broadcasting; strict shape validation using explicit ops (einops).
AR7: **Anti-Pattern (Paths)**: No `os.path`; strict usage of `pathlib`.
AR8: **Naming**: PascalCase for components/types, snake_case for methods/vars.
AR9: **Documentation**: All public APIs must have Google-style docstrings with `Args`, `Returns`, and `Shapes`.
AR10: **MCP Framework**: Use `fastmcp` for building the MCP server (simpler/faster than `mcp` SDK).

### FR Coverage Map

FR1 (TextVectorizer): Epic 1 (Core Tooling)
FR2 (RegexVectorizer): Epic 3 (Extensible Vectorization)
FR3 (VectorCache): Epic 1 (Core Tooling)
FR4 (Component Base): Epic 1 (Core Tooling)
FR5 (Typed Tensors): Epic 1 (Core Tooling)
FR6 (Combinators): Epic 2 (Advanced Composition)
FR7 (>> Syntax): Epic 2 (Advanced Composition)
FR8 (Connectors): Epic 2 (Advanced Composition)
FR9 (Cache persistence): Epic 1 (Core Tooling)
FR10 (DataLoader): Epic 3 (Extensible Vectorization)
FR11 (Aggregation): Epic 1 (Core Tooling)
FR12 (Device Mgmt): Epic 1 (Core Tooling)
FR13 (Model Zoo): Epic 4 (Model Zoo)
FR14 (Hyperopt): Epic 4 (Model Zoo)
FR15 (Visualization): Epic 2 (Advanced Composition)
FR16 (Gating): Epic 2 (Advanced Composition)
FR17 (MCP Server): Epic 1 (Core Tooling)

## Epic List

### Epic 1: Core Vector Integration & Tooling
**Goal:** Establish the foundational vector processing pipeline, enabling basic text-to-vector transformation, caching, and simple composition. This allows users to start experimenting immediately.
**FRs covered:** FR1, FR3, FR4, FR5, FR9, FR11, FR12, FR17, AR1, AR3, AR4, AR6, AR7, AR8, AR10

### Story 1.1: Core Types & Component Base (Foundation)
As a developer,
I want a strict type system and component base class,
So that I catch shape errors early and enforce configuration validation.

**Acceptance Criteria:**

**Given** a class inheriting from `VectorMeshComponent`
**When** it is instantiated with invalid configuration types
**Then** Pydantic v2 raises a validation error immediately
**And** the configuration is immutable (frozen)

**Given** a method decorated with `@jaxtyping.jaxtyped`
**When** I pass a tensor of the wrong shape (e.g., 2D instead of 1D)
**Then** `beartype` raises a detailed `TypeCheckError` describing the shape mismatch

**Given** a usage of `OneDTensor`, `TwoDTensor`, or `ThreeDTensor`
**When** I inspect the type hints
**Then** they resolve to the correct `Float[Tensor, ...]` jaxtyping definition

---

### Story 1.2: HuggingFace TextVectorizer
As a researcher,
I want to convert text to vectors using HuggingFace models without managing the model lifecycle,
So that I can focus on the vectors, not the infrastructure.

**Acceptance Criteria:**

**Given** a `TextVectorizer` initialized with a supported model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
**When** I call it with a list of strings
**Then** it automatically downloads the model (if missing)
**And** moves it to the correct device (MPS on Mac, CUDA if available, else CPU)
**And** returns a typed `OneDTensor` (batch, dim) of embeddings

**Given** an invalid model string
**When** I initialize `TextVectorizer`
**Then** it raises a descriptive `VectorMeshError` (not a generic HF error)

---

### Story 1.3: VectorCache Persistence & Storage
As a user,
I want to cache embeddings for my dataset (e.g., `assets/train.jsonl`),
So that I don't re-compute expensive vectors during every experiment.

**Acceptance Criteria:**

**Given** a list of texts loaded from `assets/train.jsonl`
**When** I call `VectorCache.create(texts, vectorizer, name="my_cache")`
**Then** it processes the texts in batches
**And** saves the results as a partitioned Parquet dataset in `.vmcache/my_cache`
**And** the directory contains `embeddings` (Arrow Array3D) and `metadata.json`

**Given** an interrupted cache creation process (e.g. SIGKILL)
**When** I check the `.vmcache` directory
**Then** the target directory is clean (does not exist or is empty), no partial corruption

---

### Story 1.4: Parameter-Free Aggregation
As a data scientist,
I want to aggregate cached 2D chunks into 1D vectors using simple strategies,
So that I can train simple classifiers (Linear, MLP) quickly.

**Acceptance Criteria:**

**Given** a loaded `VectorCache` containing 2D chunks (batch, chunks, dim)
**When** I call `.aggregate(strategy="mean")`
**Then** it returns a `OneDTensor` (batch, dim) representing the average vector per document

**Given** the same cache
**When** I call `.aggregate(strategy="max")`
**Then** it returns the max-pooled representation
**And** shape validation ensures the output is compatible with standard Linear layers

---

### Story 1.5: MCP Server (Basic)
As an AI Agent user,
I want a `fastmcp` server exposing vectorization tools,
So that my agents can vectorize text and query the cache.

**Scope:** `vectorize_text`

**Acceptance Criteria:**

**Given** the `vectormesh` MCP server running (built with `fastmcp`)
**When** an agent calls the `vectorize_text` tool
**Then** it returns the vector representation as a JSON array of floats

**Given** the MCP server
**When** I query `list_tools`
**Then** I see `vectorize_text` available

### Epic 2: Advanced Composition & Architecture
**Goal:** Enable users to build complex, branching architectures and combine multiple vector signals for sophisticated processing. This unlocks the true power of "vector mesh" composition.
**FRs covered:** FR6, FR7, FR8, FR15, FR16, AR5, AR6

### Story 2.1: Combinators (Serial/Parallel)
As a machine learning engineer,
I want explicit `Serial` and `Parallel` containers,
So that I can define complex topologies clearly without relying on fragile list chaining.

**Acceptance Criteria:**

**Given** a list of components `[A, B]`
**When** I wrap them in `Serial([A, B])`
**Then** data flows sequentially A -> B
**And** shapes are validated definition-time (static analysis/mypy) and runtime (beartype)

**Given** a list of branches `[branch1, branch2]`
**When** I wrap them in `Parallel([branch1, branch2])`
**Then** input is broadcast to both branches
**And** the output is a tuple/list of results from each branch (not automatically concatenated yet)

---

### Story 2.2: Syntactic Sugar (>> Operator)
As a developer,
I want to use `>>` to chain components,
So that my code reads like a data flow pipeline.

**Acceptance Criteria:**

**Given** two components `comp1` and `comp2`
**When** I write `pipeline = comp1 >> comp2`
**Then** `pipeline` is an instance of `Serial` containing `[comp1, comp2]`
**And** shape compatibility is checked immediately

**Given** a `Serial` pipeline
**When** I use `>>` to add another component
**Then** it returns a new `Serial` (flattened if possible)
**And** the original objects remain uniform (immutable config preference)

---

### Story 2.3: Connectors (Concat/Stack)
As a model architect,
I want typed `concat` and `stack` connectors,
So that I can merge parallel branches into a single tensor for downstream processing.

**Acceptance Criteria:**

**Given** the output of a `Parallel` branch (list of tensors)
**When** I apply `GlobalConcat(dim=-1)`
**Then** it returns a single tensor with concatenated features
**And** it validates that all other dimensions match (e.g., batch size)

**Given** a `Serial` pipeline ending in `Parallel`
**When** I append `>> GlobalConcat()`
**Then** the final output is a single tensor

---

### Story 2.4: Component Visualization
As a user,
I want to visualize my component graph,
So that I can verify the topology and tensor shapes.

**Acceptance Criteria:**

**Given** a complex `Serial` pipeline
**When** I call `visualize(pipeline)`
**Then** it prints a clear ASCII or text representation of the flow
**And** it shows the input/output shapes between layers (e.g., `(B, 768) -> [Serial] -> (B, 128)`)

---

### Story 2.5: Gating Mechanisms
As an advanced user,
I want gating components (`Gate`, `GateSkip`),
So that I can dynamically route data or apply residual connections based on vector content.

**Acceptance Criteria:**

**Given** a "Main" path and a "Skip" path
**When** I use `GateSkip(main=..., skip=...)`
**Then** the outputs are combined (typically summed)
**And** shape validation ensures `main` and `skip` outputs are compatible/broadcastable

**Given** a `Gate` component
**When** I provide a routing tensor/score
**Then** it modulates the signal accordingly (e.g., soft gating or hard routing)

### Epic 3: Extensible Vectorization & Data
**Goal:** Expand vectorization capabilities beyond basic text to include regex, custom logic, and efficiently handle large/variable datasets for real-world use cases.
**FRs covered:** FR2, FR10, NFR4

### Story 3.1: RegexVectorizer
As a feature engineer,
I want to create binary vectors based on regex pattern matches,
So that I can extract explicit features (e.g., "contains_email", "is_capitalized") alongside semantic embeddings.

**Acceptance Criteria:**

**Given** a dictionary of patterns `{"email": r"...", "upper": r"[A-Z]+"}`
**When** I initialize `RegexVectorizer(patterns)`
**Then** it validates the regexes are valid python re patterns (raises specifically `VectorMeshError` on invalid syntax)

**Given** a list of strings
**When** I call `vectorizer.vectorize(texts)`
**Then** it returns a `OneDTensor` of shape `(batch, num_patterns)`
**And** values are 1.0 (match) or 0.0 (no match)

---

### Story 3.2: Smart DataLoader
As a trainer,
I want a DataLoader that handles variable chunk counts,
So that I can batch documents together even if they split into different numbers of chunks.

**Acceptance Criteria:**

**Given** a dataset where Document A has 5 chunks and Document B has 3 chunks
**When** I create a batch of size 2
**Then** the `DataLoader` automatically pads Document B to 5 chunks (using a special padding vector)
**And** returns an `attention_mask` indicating which chunks are real vs padded

**Given** a very large dataset
**When** I iterate over the DataLoader
**Then** it streams data efficiently without loading everything into RAM (leveraging HF Datasets lazy loading)

---

### Story 3.3: Custom Base Aggregator
As a researcher,
I want a base class for Aggregators,
So that I can implement custom aggregation logic (e.g., Attention-based pooling) that is compatible with the rest of the pipeline.

**Acceptance Criteria:**

**Given** a custom class inheriting from `BaseAggregator`
**When** I implement the `forward` method
**Then** the base class enforces strict type hints for input (Batch, Chunks, Dim) and output (Batch, Dim)

**Given** my custom aggregator
**When** I plug it into a `Serial` pipeline
**Then** it works seamlessly with `VectorCache` output

---

### Story 3.4: Dataset Validation
As a user,
I want to validate my input data schema,
So that I don't run a long batch job only to fail halfway through due to a missing column.

**Acceptance Criteria:**

**Given** an input file (jsonl or parquet)
**When** I attempt to load it
**Then** the system checks for required columns (e.g., "text" or "id")
**And** raises a helpful error if the schema is invalid

**Given** a dataset with `None` values in the text column
**When** validation runs
**Then** it flags specific rows with null values

### Epic 4: Model Zoo & Ecosystem
**Goal:** Provide a curated, reliable library of models and utilities for rigorous experimentation and optimization. This ensures users have high-quality building blocks.
**FRs covered:** FR13, FR14, NFR1, NFR2, NFR3, NFR6, AR2, AR9

### Story 4.1: Model Zoo Registry
As a user,
I want to load models by simple enum names (e.g., `Model.MINILM`),
So that I don't have to remember long HuggingFace ID strings or worry about typos.

**Acceptance Criteria:**

**Given** the `Zoo` module
**When** I access `Model.BERT_TINY`
**Then** it returns the correct HF string "prajjwal1/bert-tiny"
**And** provides metadata like `dim=128` and `max_seq_len=512`

**Given** a request for `Model.ROBERTA_BASE`
**When** I instantiate a `TextVectorizer` with it
**Then** it loads correctly without me pasting the path

---

### Story 4.2: Hyperopt Integration
As a machine learning engineer,
I want to define search spaces for my pipeline parameters,
So that I can optimize performance using `hyperopt`.

**Acceptance Criteria:**

**Given** a pipeline with parameters (e.g., aggregation strategy)
**When** I define a search space using `vector_mesh.opt.choice(["mean", "max"])`
**Then** it is compatible with standard `hyperopt` trials

**Given** a search trial
**When** the optimizer suggests a config
**Then** I can instantiate the pipeline with that config using the `VectorMeshComponent` frozen configuration system

---

### Story 4.3: Educational Error System
As a student/learner,
I want error messages that tell me what went wrong and how to fix it,
So that I learn the concepts instead of just fixing bugs.

**Acceptance Criteria:**

**Given** a shape mismatch error (e.g., passing 1D tensor to 2D expected input)
**When** `VectorMeshError` is raised
**Then** the output contains a `Hint:` section explaining the concept
**And** a `Fix:` section suggesting likely code changes (e.g., "Try using .unsqueeze(0)")

**Given** a code block
**When** I run it
**Then** I see custom colored output for errors (using `rich` or standard ANSI codes) for better readability

---

### Story 4.4: Production Guardrails & Testing
As a maintainer,
I want to ensure the library is safe and reliable,
So that I can trust it in production or classroom environments.

**Acceptance Criteria:**

**Given** any model loading call
**When** it executes
**Then** it strictly enforces `weights_only=True` to prevent pickle attacks

**Given** a malicious pickle file
**When** loaded via `TextVectorizer`
**Then** it raises `UnpicklingError` or safety violation (caught by safety filter)

**Given** the test suite
**When** I run integration tests
**Then** `vcrpy` records/replays network interactions so tests pass offline


<!-- Repeat for each epic in epics_list (N = 1, 2, 3...) -->

## Epic {{N}}: {{epic_title_N}}

{{epic_goal_N}}

<!-- Repeat for each story (M = 1, 2, 3...) within epic N -->

### Story {{N}}.{{M}}: {{story_title_N_M}}

As a {{user_type}},
I want {{capability}},
So that {{value_benefit}}.

**Acceptance Criteria:**

<!-- for each AC on this story -->

**Given** {{precondition}}
**When** {{action}}
**Then** {{expected_outcome}}
**And** {{additional_criteria}}

<!-- End story repeat -->
