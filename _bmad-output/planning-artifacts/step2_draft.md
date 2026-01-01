## Epic List

### Epic 1: Core Vector Integration & Tooling
**Goal:** Establish the foundational vector processing pipeline, enabling basic text-to-vector transformation, caching, and simple composition. This allows users to start experimenting immediately.
**FRs covered:** FR1, FR3, FR4, FR5, FR9, FR11, FR12, FR17, AR1, AR3, AR4, AR6, AR7, AR8, AR10

### Epic 2: Advanced Composition & Architecture
**Goal:** Enable users to build complex, branching architectures and combine multiple vector signals for sophisticated processing. This unlocks the true power of "vector mesh" composition.
**FRs covered:** FR6, FR7, FR8, FR15, FR16, AR5, AR6

### Epic 3: Extensible Vectorization & Data
**Goal:** Expand vectorization capabilities beyond basic text to include regex, custom logic, and efficiently handle large/variable datasets for real-world use cases.
**FRs covered:** FR2, FR10, NFR4

### Epic 4: Model Zoo & Ecosystem
**Goal:** Provide a curated, reliable library of models and utilities for rigorous experimentation and optimization. This ensures users have high-quality building blocks.
**FRs covered:** FR13, FR14, NFR1, NFR2, NFR3, NFR6, AR2, AR9

---

### FR Coverage Map

FR1 (TextVectorizer): Epic 1 (Core Tooling)
FR2 (RegexVectorizer): Epic 3 (Extensible Vectorization)
FR3 (VectorCache): Epic 1 (Core Tooling)
FR4 (Component Base): Epic 1 (Core Tooling)
FR5 (Typed Tensors): Epic 1 (Core Tooling)
FR6 (Combinators): Epic 2 (Advanced Composition)
FR7 (>> Syntax): Epic 2 (Advanced Composition)
FR8 (Connectors): Epic 2 (Advanced Composition)
FR9 (Cache Persistence): Epic 1 (Core Tooling)
FR10 (DataLoader): Epic 3 (Extensible Vectorization)
FR11 (Aggregation): Epic 1 (Core Tooling)
FR12 (Device Mgmt): Epic 1 (Core Tooling)
FR13 (Model Zoo): Epic 4 (Model Zoo)
FR14 (Hyperopt): Epic 4 (Model Zoo)
FR15 (Visualization): Epic 2 (Advanced Composition)
FR16 (Gating): Epic 2 (Advanced Composition)
FR17 (MCP Server): Epic 1 (Core Tooling)
