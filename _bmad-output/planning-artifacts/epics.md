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

FR1: Implement `TwoDVectorizer` and `ThreeDVectorizer` for HuggingFace models with uniform interface and auto-detection.
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

FR1 (TwoDVectorizer/ThreeDVectorizer): Epic 1 (Core Tooling)
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

### Story 1.2: HuggingFace TwoDVectorizer & ThreeDVectorizer
As a researcher,
I want to convert text to vectors using HuggingFace models without managing the model lifecycle,
So that I can focus on the vectors, not the infrastructure.

**Acceptance Criteria:**

**Given** a `TwoDVectorizer` initialized with a sentence-transformer model (e.g., "sentence-transformers/all-MiniLM-L6-v2")
**When** I call it with a list of strings
**Then** it automatically downloads the model (if missing)
**And** moves it to the correct device (MPS on Mac, CUDA if available, else CPU)
**And** returns a typed `TwoDTensor` (batch, dim) of embeddings

**Given** a `ThreeDVectorizer` initialized with a raw transformer model
**When** I call it with a list of strings
**Then** it chunks long texts automatically
**And** returns a typed `ThreeDTensor` (batch, chunks, dim) of embeddings

**Given** an invalid model string
**When** I initialize either vectorizer
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

**📋 Story Integration Dependencies:**
- **Story 2.1**: Foundation - implements `Serial` and `Parallel` combinators with 2D/3D awareness
- **Story 2.2**: Builds on 2.1 - `>>` operator compiles to `Serial` containers
- **Story 2.3**: Builds on 2.1 - `GlobalConcat` consumes `Parallel` tuple outputs
- **Story 2.4**: Builds on 2.1 - visualizes combinator structures and tensor flows
- **Story 2.5**: Builds on 2.1 - gating integrates with combinator framework

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

**Integration Note:** Depends on Story 2.1 `Serial` combinator implementation. The `>>` operator compiles to `Serial` containers created in Story 2.1.

**Acceptance Criteria:**

**Given** two components `TwoDVectorizer()` and `MeanProcessor()`
**When** I write `pipeline = vectorizer >> processor`
**Then** `pipeline` is an instance of `Serial` containing `[TwoDVectorizer(), MeanProcessor()]`
**And** shape compatibility is checked immediately with 2D/3D awareness

**Given** a `Serial` pipeline from Story 2.1
**When** I use `>>` to add another component
**Then** it returns a new `Serial` (flattened if possible)
**And** the original objects remain immutable (frozen config from Story 2.1)

**Given** mixed 2D/3D components
**When** I write `ThreeDVectorizer("model") >> MeanAggregator() >> FinalProcessor()`
**Then** it creates a `Serial` with proper 3D→2D→2D shape flow validation

---

### Story 2.3: Connectors (Concat/Stack)
As a model architect,
I want typed `concat` and `stack` connectors,
So that I can merge parallel branches into a single tensor for downstream processing.

**Integration Note:** Depends on Story 2.1 `Parallel` combinator implementation. `GlobalConcat` must consume tuple outputs from `Parallel` containers and handle all 2D/3D combinations defined in Story 2.1.

**Acceptance Criteria:**

**Given** a `Parallel` output tuple from Story 2.1 `(TwoDTensor, TwoDTensor)`
**When** I apply `GlobalConcat(dim=-1)`
**Then** it returns a single `TwoDTensor` with concatenated features `[batch, combined_dim]`
**And** it validates that batch dimensions match

**Given** a mixed `Parallel` output `(TwoDTensor, ThreeDTensor)` from Story 2.1
**When** I apply `GlobalConcat(dim=-1)`
**Then** it raises an educational `VectorMeshError` explaining 2D/3D incompatibility
**And** suggests using aggregation first: "Hint: Use MeanAggregator() on 3D branch before concatenation"

**Given** a `Serial` pipeline ending in `Parallel` from Story 2.1
**When** I append `>> GlobalConcat(dim=-1)`
**Then** the final output is a single tensor with proper dimension handling

**Given** a normalized `Parallel` output `(TwoDTensor, TwoDTensor)` from `[TwoDVectorizer, Serial([ThreeDVectorizer, MeanAggregator])]`
**When** I apply `GlobalConcat(dim=-1)`
**Then** it concatenates successfully because aggregation normalized dimensions

---

### Story 2.4: Gating Mechanisms (Basic)
As an advanced user,
I want basic gating components (`Skip` and `Gate`),
So that I can add residual connections and signal modulation without complex magic.

**Integration Note:** Depends on Story 2.1 combinator framework. Components inherit from `VectorMeshComponent` with frozen Pydantic pattern. No auto-magic features - explicit, educational design.

**Acceptance Criteria:**

**Given** a main component path
**When** I use `Skip(main=MyLayer())`
**Then** it computes `LayerNorm(input + main(input))` with add+norm pattern
**And** validates shapes match (or uses manual projection if provided)
**And** raises educational errors on mismatches

**Given** a component to gate with a router function
**When** I use `Gate(component=MyLayer(), router=my_router_fn)`
**Then** it computes `router(input) * component(input)`
**And** router is always required (no default pass-through)
**And** integrates seamlessly with Serial/Parallel from Story 2.1

**Given** Skip or Gate in combinators
**When** I compose complex pipelines
**Then** they maintain proper 2D/3D tensor type tracking
**And** follow VectorMeshComponent pattern (frozen=True)

---

### Story 2.5: Complex Gating & MoE
As an advanced user,
I want complex gating mechanisms (Highway networks, MoE, learnable gates),
So that I can build sophisticated routing and mixture-of-experts architectures.

**Integration Note:** Builds on Story 2.4 basic gating. Adds learnable components, sparse routing, and advanced patterns from research literature.

**Acceptance Criteria:**

**Given** a transform component
**When** I use `Highway(transform=MyLayer(), gate_fn=learned_gate)`
**Then** it computes `G * transform(input) + (1-G) * input` with learned gating
**And** implements Highway network pattern from literature

**Given** multiple expert components
**When** I use `MoE(experts=[E1(), E2(), E3()], router=my_router, top_k=2)`
**Then** it routes to top-k experts based on learned routing
**And** combines outputs with routing weights
**And** supports load balancing

**Given** a component with learnable gate
**When** I use `LearnableGate(component=MyLayer(), gate_network=GateNet())`
**Then** gradients flow through both component and gate network
**And** integrates with PyTorch autograd

---

### Story 2.6: Component Visualization
As a user,
I want to visualize my component graph,
So that I can verify the topology and tensor shapes.

**Integration Note:** Depends on Story 2.1 `Serial` and `Parallel` combinators, Story 2.4 basic gating (Skip/Gate), and Story 2.5 complex gating (Highway/MoE). Must visualize nested structures and 2D/3D tensor flows with mathematical notation.

**Acceptance Criteria:**

**Given** a `Serial` pipeline from Story 2.1
**When** I call `visualize(pipeline)`
**Then** it prints a clear representation with mathematical notation showing: `TwoDVectorizer → MeanProcessor → ℝ^{B×768} → ℝ^{B×128}`

**Given** a `Parallel` structure from Story 2.1
**When** I call `visualize(parallel_pipeline)`
**Then** it shows branching topology with mathematical symbols:
```
Input: Σ*
├── TwoDVectorizer("model1") → ℝ^{B×384}
└── TwoDVectorizer("model2") → ℝ^{B×768}
Output: (ℝ^{B×384}, ℝ^{B×768})
```

**Given** nested combinators from Story 2.1 AC4c with gating from Stories 2.4 and 2.5
**When** I visualize the complex multi-level structure
**Then** it shows the complete hierarchy including skip connections, gates, Highway networks, and MoE with proper 2D/3D dimension flow tracking

### Epic 3: Extensible Vectorization & Data
**Goal:** Expand vectorization capabilities beyond basic text to include regex, custom logic, and efficiently handle large/variable datasets for real-world use cases.
**FRs covered:** FR2, FR10, NFR4

### Story 3.1: RegexVectorizer
As a feature engineer,
I want to create binary vectors based on regex pattern matches,
So that I can extract explicit features. Some examples of what i would like to find is:
- "artikel 265 Boek 3 van het Burgerlijk Wetboek" should be detected, another variantion is "artikel 7:2 Burgerlijk Wetboek"
- "artikel 7:26 lid 3 van het Burgerlijk Wetboek" can also be a variation
- "artikelen 6:251 en 6:252 Burgerlijk Wetboek"  should detect both 6:251 and 6:252
- "artikel 55 Wet Bodembescherming" is interesting as well, because it is a "wet"
- but "artikel 6.5 en 31.5 van de koopovereenkomst" should NOT be detected; this is not "wetboek" but koopovereenkomst.

And it should be clear that i want to find artikelen, regardless of the formulation (6:252 or 265 boek 3, both variations should detect just the number) but obviously searching for [0-9];[0-9]+ wont be good enough, because we need to find also the mention of the wetboek, etc.

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
**When** I instantiate a `TwoDVectorizer` or `ThreeDVectorizer` with it
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
**When** loaded via `TwoDVectorizer` or `ThreeDVectorizer`
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
