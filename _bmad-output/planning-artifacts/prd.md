---
stepsCompleted: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
inputDocuments:
  - docs/README.md
  - /Users/rgrouls/code/MADS/exercises/akte-classifier/README.md
  - /Users/rgrouls/code/MADS/exercises/akte-classifier/src/akte_classifier/utils/tensor.py
  - /Users/rgrouls/code/MADS/exercises/akte-classifier/src/akte_classifier/models/neural.py
  - /Users/rgrouls/code/MADS/exercises/akte-classifier/src/akte_classifier/models/regex.py
  - /Users/rgrouls/code/MADS/exercises/akte-classifier/src/akte_classifier/trainer.py
workflowType: 'prd'
lastStep: 10
documentCounts:
  briefCount: 0
  researchCount: 0
  brainstormingCount: 0
  projectDocsCount: 7
referenceProject:
  path: /Users/rgrouls/code/MADS/exercises/akte-classifier
  purpose: Reference implementation - learning from what worked and what didn't
  conceptsThatWorked:
    - Basic idea of modular vectorizers
    - Smart device detection (GPU/MPS/CPU)
    - Hash-based cache concept (though implementation was messy)
  issuesAndPitfalls:
    - Cache versioning was messy and needs cleaner implementation
    - Auto-detect pooling didn't work 100% (should use HF MCP instead)
    - models/neural.py too focused on single-model usage
    - No type validation (should use Pydantic)
    - Mixed use of os and pathlib
    - Lack of strict design principles
    - No linting enforcement
  designPrinciples:
    - SRP (Single Responsibility Principle)
    - Open-Closed Principle
    - Use Pydantic for all type validation and input/output contracts
    - Always use pathlib (never os for paths)
    - Enforce linting standards
    - Use Hugging Face MCP for model discovery and validation
  coreVision:
    - SDK handles efficiency and speed
    - Users focus on playing with vectors
    - Clear separation: vectorization → connectors → custom architectures
    - Pydantic-enforced expectations about model inputs/outputs
    - Test-first approach with validated HF models
    - Build-your-own-model with flexible layer composition
    - Hyperparameter tuning capabilities
---

# Product Requirements Document - vectormesh

**Author:** raoul
**Date:** 2025-12-31


  ## Executive Summary                                                        
                                                                              
  **VectorMesh** is a Python SDK that makes vector-space composition a first- 
  class programming primitive for machine learning. It treats pre-trained     
  HuggingFace models not as endpoints for transfer learning, but as composable
  functions that transform data through high-dimensional spaces.              
                                                                              
  ### The Fundamental Problem                                                 
                                                                              
  Deep learning is fundamentally about vector transformations, but current    
  tools create unnecessary friction:                                          
                                                                              
  1. **Wrong abstraction level**: We treat models as opaque boxes instead of  
  composable functions                                                        
  2. **HuggingFace complexity**: Students must learn each model's quirks,     
  transfer learning patterns, and optimization tricks - only ~2% discover     
  caching on their own                                                        
  3. **Missing caching**: Nobody thinks to cache text embeddings, so every    
  experiment re-computes everything (wasting hours)                           
  4. **Generic tensors**: PyTorch uses untyped Tensor - students connect CNNs 
  to NNs and get cryptic runtime errors about 4D→2D shape mismatches          
  5. **Architectural patterns are hard**: Skip connections, parallel layers,  
  gates require manual wiring                                                 
  6. **GPU inequality**: 80% of students lack local GPUs, making              
  experimentation prohibitively slow                                          
                                                                              
  The root issue: **We lack composable primitives for vector-space            
  experimentation with intelligent caching.**                                 
                                                                              
  ### The Solution: Composable Vector Primitives with Chunk-Level Caching     
                                                                              
  **HuggingFace Abstraction:**                                                
                                                                              
  • Uniform interface: all models become text → vectors regardless of         
  architecture                                                                
  • **Chunk-level caching**: Store raw embeddings as 2DTensor (not aggregated),
  enabling both parameter-free and learned aggregation                        
  • **One cache per model**: Professor builds on GPU once, students experiment
  infinitely on CPU                                                           
  • Smart defaults: pooling strategy, max length, device management handled   
  automatically                                                               
  • Students focus on composition, not model internals                        
                                                                              
  **Typed Tensor System:**                                                    
                                                                              
    from vectormesh import TextVectorizer, concat, stack                      
                                                                              
    bert = TextVectorizer("bert-base")      # → 1DTensor[768]                 
    regex = RegexVectorizer(patterns)       # → 1DTensor[50]                  
                                                                              
    # Concatenate → 1DTensor[818] → feed to NN                                
    features_1d = concat(bert(texts), regex(texts))                           
                                                                              
    # Stack → 2DTensor[2, 768] → feed to CNN or attention                     
    features_2d = stack(bert(texts), distilbert(texts))                       
                                                                              
  **Different shapes unlock different architectures:**                        
                                                                              
  • 1DTensor → Dense neural networks                                          
  • 2DTensor → CNNs, cross-attention between embeddings                       
  • 3DTensor → Video/sequence models, transformers over embeddings            
                                                                              
  **Type errors at definition time, not runtime:**                            
                                                                              
    # Student mistake caught immediately:                                     
    cnn_layer = CNN2D(...)                                                    
    dense_input = bert(text)  # 1DTensor[768]                                 
                                                                              
    pipeline = dense_input >> cnn_layer                                       
    # ❌ TypeError: Cannot connect 1DTensor to CNN2D expecting 2DTensor       
    # ✅ Error shows: "Expected 2DTensor[*, 768], got 1DTensor[768]"          
                                                                              
  **Chunk-Level Caching Architecture:**                                       
                                                                              
  For long documents or models with limited context windows (512 tokens),     
  VectorMesh caches raw chunks, not aggregated embeddings:                    
                                                                              
    # Cache stores 2DTensor chunks (one cache per model)                      
    cache_structure = {                                                       
        "embeddings": Tensor[N, n_chunks, dim],  # Raw chunks!                
        "masks": Tensor[N, n_chunks],            # Padding masks              
        "metadata": {"model": "mpnet", "chunk_size": 512}                     
    }                                                                         
                                                                              
    # Students experiment with aggregation WITHOUT re-computing               
    chunks = cache.get_chunks()  # 2DTensor[N, 24, 768]                       
                                                                              
    # Path 1: Parameter-free (instant)                                        
    features = chunks.mean(dim=1)  # → 1DTensor[N, 768]                       
                                                                              
    # Path 2: Learned aggregation (trainable!)                                
    class MyAttention(nn.Module):                                             
        def forward(self, chunks, mask):                                      
            # Student builds this using what they learned:                    
            # attention, dropout, batch norm, etc.                            
            ...                                                               
            return aggregated  # → 1DTensor[N, 768]                           
                                                                              
  **DataLoader with Automatic Padding:**                                      
                                                                              
    # Variable chunk sizes (doc1: 10 chunks, doc2: 24 chunks)                 
    # VectorMesh pads to max and provides attention masks                     
    loader = cache.to_dataloader(batch_size=32, pad_to_max=True)              
                                                                              
    for batch in loader:                                                      
        chunks = batch.embeddings    # [32, 24, 768] (padded)                 
        mask = batch.attention_mask  # [32, 24] (1=real, 0=padding)           
                                                                              
        # Student's aggregator respects padding                               
        features = my_aggregator(chunks, mask)                                
                                                                              
  **Architectural Components as First-Class Citizens:**                       
                                                                              
    from vectormesh import SkipConnection, ParallelBranch, Gate               
                                                                              
    # Skip connection (like ResNet)                                           
    skip = SkipConnection(                                                    
        transform=dense_layers,                                               
        residual=identity                                                     
    )                                                                         
                                                                              
    # Parallel branches (like GoogLeNet Inception)                            
    parallel = ParallelBranch([                                               
        bert_branch,      # 1DTensor[768]                                     
        tfidf_branch,     # 1DTensor[5000]                                    
        regex_branch      # 1DTensor[50]                                      
    ])  # → combines to 1DTensor[5818]                                        
                                                                              
    # Gate (mixture of experts style)                                         
    gated = Gate(                                                             
        router=attention_layer,                                               
        experts=[bert, roberta, distilbert]                                   
    )                                                                         
                                                                              
  **All components have typed inputs/outputs:**                               
                                                                              
  • Dimension mismatches caught at graph construction time                    
  • Pydantic contracts: expects: 1DTensor[768], got: 2DTensor[512, 768]       
  • Clear error messages guide composition                                    
                                                                              
  **Curated Model Library (10 Models):**                                      
                                                                              
  VectorMesh includes tested models with known characteristics:               
                                                                              
  **Large Context (8k-32k tokens):**                                          
                                                                              
  1. **BAAI/bge-multilingual-gemma2** (8k, multilingual, 9.2B params)         
  2. **intfloat/e5-mistral-7b-instruct** (32k, English, 7B params)            
  3. **Qwen/Qwen3-Embedding-8B** (32k, multilingual, 8B params)               
  4. **Qwen/Qwen3-Embedding-0.6B** (32k, multilingual, 600M params - **CPU-   
  friendly!**)                                                                
                                                                              
  **Standard Context with Auto-Chunk (512 tokens):**                          
  5. **sentence-transformers/paraphrase-multilingual-mpnet-base-v2** (Dutch,  
  249M                                                                        
  downloads)                                                                  
  6. **sentence-transformers/LaBSE** (109 languages, 470M params)             
  7. **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (fastest,
  117M                                                                        
  params)                                                                     
  8. **sentence-transformers/distiluse-base-multilingual-cased-v2** (distilled,
  134M params)                                                                
                                                                              
  All validated via HuggingFace MCP for Dutch support, context windows, and   
  pooling strategies.                                                         
                                                                              
  ### What Makes This Special                                                 
                                                                              
  **"Compose vectors like LEGO, cache chunks for infinite experimentation"**  
                                                                              
  1. **HuggingFace without the headaches**: Students don't learn 50 different 
  model APIs - they learn "models are functions that return typed vectors"    
  2. **Chunk-level caching**: Cache raw 2DTensor embeddings once, experiment  
  with aggregation strategies infinitely                                      
  3. **Professor builds, students experiment**: Professor computes on GPU → 3 
  models × 20k docs = 530MB cache → 30 students download → experiment with 45+
  combinations on CPU                                                         
  4. **Typed tensors prevent bugs**: Cannot connect 2DTensor to Linear layer  
  expecting 1DTensor - errors at definition time, not after 30-minute training
  run                                                                         
  5. **Architectural patterns**: Skip connections, parallel branches, gates - 
  all "just work" with type checking                                          
  6. **Visual composition diagrams**: See your architecture as a category     
  theory diagram - composition becomes visual and mathematical                
  7. **Parameter-free vs learned aggregation**: Students learn "When should I 
  aggregate before or after learning?" through experimentation                
                                                                              
  **Key pedagogical insight:**                                                
                                                                              
  • Tensor shape determines architectural possibilities - 2DTensor means "I   
  can learn aggregation," 1DTensor means "features are fixed"                 
  • Chunk-level caching separates infrastructure (embedding computation) from 
  experimentation (aggregation + architecture)                                
  • 80% of students without GPUs can hypertune across models because professor
  pre-computes caches                                                         
                                                                              
  **Key technical insight:**                                                  
  Composition often beats fine-tuning - combining signals (BERT + regex + TF- 
  IDF)                                                                        
  outperforms fine-tuning a single model, especially with limited data.       
  VectorMesh makes this composition-first approach natural and fast.          
                                                                              
  ### Target Users                                                            
                                                                              
  • **Master students** learning ML who understand theory but aren't strong   
  software engineers - need tools that handle efficiency and type safety      
  • **ML practitioners** rapidly prototyping architectures at the vector level
  • **Researchers** exploring novel composition patterns without              
  infrastructure overhead                                                     
  • **Instructors** teaching 100+ students/year who need shareable caches and 
  reproducible experiments                                                    
                                                                              
  ### Core Capabilities                                                       
                                                                              
  • **Uniform HuggingFace interface** with chunk-level caching (2DTensor per  
  model), pooling, device management                                          
  • **Collaborative caching system**:                                         
      • One cache file per model (e.g., legal_2024_mpnet.vmcache)             
      • Export/import .vmcache files (professor → GPU → students)             
      • Content-hash versioning                                               
      • Chunk-level storage enables learned aggregation                       
      • DataLoader with automatic padding for variable chunk sizes            
  • **Typed tensor system**: 1DTensor, 2DTensor, 3DTensor with shape inference
  • **Composable vectorizers**: text embeddings (10 curated HF models), regex,
  TF-IDF, custom transformations                                              
  • **Architectural components**:                                             
      • concat, stack for combining vectors                                   
      • SkipConnection (ResNet-style)                                         
      • ParallelBranch (Inception-style)                                      
      • Gate (mixture-of-experts style)                                       
      • Students build custom aggregators (Attention, CNN, LSTM)              
      • Custom connectors (open-closed principle)                             
  • **Aggregation strategies**:                                               
      • Parameter-free: mean, max pooling (instant experimentation)           
      • Learned: students build Attention, CNN, LSTM aggregators with PyTorch 
      • Both work with chunk-level caches                                     
  • **Composition visualization**:                                            
      • Diagrammatic rendering (ASCII, SVG, HTML)                             
      • Category theory inspired notation                                     
      • Type flow visualization                                               
      • Commutative diagram validation                                        
  • **Hyperparameter tuning integration**:                                    
      • Ray Tune support with search spaces for models + aggregation +        
      architecture                                                            
      • Hyperopt compatibility                                                
      • Example search spaces including model selection, aggregation strategy,
      NN architecture                                                         
      • Works with cached vectors for fast iteration                          
  • **Production-grade quality**:                                             
      • Pydantic contracts catch errors early                                 
      • pathlib for all paths                                                 
      • Linting enforcement                                                   
      • uv package management                                                 
      • Battle-tested patterns from akte-classifier                           
                                                                              
                                                                              
  **Student Examples (Quickstart):**                                          
                                                                              
    # Example 1: HF + Regex → Concat → NN (like akte-classifier but cached)   
    bert_cache = VectorCache.load("legal_bert.vmcache")                       
    features = concat(                                                        
        bert_cache.aggregate("mean"),  # 1DTensor[N, 768]                     
        regex_vec(texts)                # 1DTensor[N, 50]                     
    )  # → 1DTensor[N, 818]                                                   
                                                                              
    # Example 2: Chunks → 2DTensor → CNN                                      
    chunks = cache.get_chunks()  # 2DTensor[N, 24, 768]                       
    # Reshape and run 2D convolution...                                       
                                                                              
    # Example 3: Ray Tune hypertuning across models and aggregation           
    search_space = {                                                          
        "model": tune.choice(["mpnet", "bge", "qwen"]),                       
        "aggregation": tune.choice(["mean", "attention", "cnn"]),             
        "hidden_dim": tune.choice([128, 256, 512]),                           
        "lr": tune.loguniform(1e-4, 1e-2)                                     
    }                                                                         
                                                                              
  **Differentiation:**                                                        
                                                                              
  • **vs LangChain/Haystack**: They focus on LLM chains/RAG. VectorMesh       
  focuses on typed vector composition with chunk-level caching.               
  • **vs PyTorch nn.Sequential**: VectorMesh operates at vector level (after  
  embedding), not layer level. Typed tensors catch shape errors early. Chunk- 
  level caching enables aggregation experimentation.                          
  • **vs HuggingFace Transformers**: Transformers is a model zoo. VectorMesh  
  is a composition framework with intelligent caching that makes the zoo      
  composable and experimentable for students.                                 
                                                                              
  ### Why Now: Second-System Clarity                                          
                                                                              
  VectorMesh rebuilds akte-classifier with lessons learned:                   
                                                                              
  • **akte-classifier problem**: Grew organically around one dataset, became  
  rigid and unfocused, ~98% of students wouldn't discover caching             
  • **VectorMesh approach**: Built from first principles with vector          
  composition and chunk-level caching as core abstractions                    
  • **Result**: Clean architecture that works for both learning (exploration  
  with cached chunks) and production (type safety and rigor)                  
                                                                              
  **akte-classifier insights applied:**                                       
                                                                              
  • Cache versioning was messy → VectorMesh uses one cache per model with     
  content hashing                                                             
  • Auto-detect pooling failed → VectorMesh uses HF MCP for validation        
  • Fixed architectures prevented experimentation → VectorMesh makes          
  architectural components first-class                                        
  • No type safety → VectorMesh uses Pydantic throughout                      
  • Mixed os/pathlib → VectorMesh always uses pathlib                         
                                                                              
  ## Project Classification                                                   
                                                                              
  **Technical Type:** Developer Tool (Python SDK)                             
  **Domain:** Scientific Computing (ML/AI) with Educational Focus             
  **Complexity:** Medium                                                      
  **Project Context:** Greenfield                                             
  **Package Management:** uv                                                  
  **Core Framework:** PyTorch + HuggingFace Transformers                      
  **Target Environment:** 80% students without local GPU (CPU + shared caches)
                                                                              
  **Classification Rationale:**                                               
                                                                              
  • SDK providing mathematical primitives for vector composition in the Python
  ecosystem                                                                   
  • Scientific computing domain (ML/AI research and application)              
  • Educational + Production use cases unified through clean abstractions and 
  intelligent caching                                                         
  • Medium complexity: requires understanding ML workflows, chunk-level       
  caching, function composition, type systems                                 
  • Chunk-level caching democratizes experimentation (professor GPU → student 
  CPU)                                                                        
                                                                              
  **The Core Value Triangle:**                                                
                                                                              
            Typed Safety                                                      
           (Pydantic types)                                                   
                /\                                                            
               /  \                                                           
              /    \                                                          
             /      \                                                         
       Chunk-Level -------- HF Abstraction                                    
       Caching              (Uniform API +                                    
       (2DTensor storage)    10 models)                                       
                                                                              
  All three together create the unique value proposition: students experiment
  with vector composition using production-grade tools, learning both the
  mathematics and the engineering.


## Success Criteria

### User Success

**For Students:**
- Focus shifts from infrastructure to composition: "What if I combine BERT + regex + TF-IDF?"
- Experiments that took hours (re-embedding every iteration) now take seconds (load cached chunks)
- Type errors caught at definition time, not after 30-minute training runs
- Common patterns (skip connections, parallel branches) "just work" with clear composition syntax
- Learning outcomes: students understand when aggregation is learned vs parameter-free

**For Instructors:**
- Pre-compute caches on GPU once → share with 30+ students → all experiment on CPU
- Reproducible experiments via shared .vmcache files
- Students focus on architectural decisions, not HuggingFace API quirks
- Reduced support burden: Pydantic catches 80% of composition mistakes early

**For Hardware Democratization:**
- 80% of students without local GPUs can hypertune across 10 models + aggregation strategies
- Cache sizes: ~17-53MB per model for 20k documents (shareable via USB/cloud)
- Professor computes embeddings → students experiment infinitely on laptops

### Creator Success (raoul as Instructor)

**Educational Effectiveness:**
- Students learn "composition beats fine-tuning" through experimentation, not lectures
- Type system teaches architectural constraints (1DTensor → NN, 2DTensor → CNN) implicitly
- Chunk-level caching teaches separation of concerns (embedding computation vs aggregation)
- 100+ students/year can run hypertuning experiments without queuing for GPU cluster

**Technical Quality:**
- akte-classifier lessons applied: Pydantic types, pathlib, clean architecture
- Production-ready patterns from day one (linting, type contracts, device management)
- Second-system clarity: focus on vector composition + chunk-level caching as core abstractions

### Technical Success Indicators

**Chunk-Level Caching:**
- One cache per model with content-hash versioning
- 2DTensor storage enables both parameter-free (mean/max) and learned (Attention/CNN) aggregation
- DataLoader automatically pads variable chunk sizes with attention masks
- Export/import .vmcache files work across machines

**Type Safety:**
- Cannot connect 2DTensor to Linear layer expecting 1DTensor (caught at definition time)
- Clear error messages: "Expected 1DTensor[768], got 2DTensor[24, 768]"
- Pydantic contracts validate all vectorizer inputs/outputs

**HuggingFace Integration:**
- Uniform interface: 10 curated models tested via HF MCP
- Pooling strategies auto-detected (no manual configuration)
- Device management (GPU/MPS/CPU) handled transparently
- Models ranging from 117M (MiniLM, CPU-friendly) to 9.2B params (bge-gemma2, large context)

**Clean Architecture:**
- SRP: vectorizers, connectors, aggregators, caches as separate concerns
- Open-Closed: users can add custom regexes, aggregators, connectors
- pathlib for all file operations (never os)
- Linting passes with no warnings
- uv package management (not pip)

### Measurable Outcomes

VectorMesh success is qualitative rather than formal metrics. Indicators include:

**Student Adoption:**
- Students load caches and run experiments within 10 minutes of starting
- Students successfully combine 3+ vectorizers without asking for help
- Students debug composition errors using type error messages alone
- Students build custom aggregators (Attention, CNN) using cached chunks

**Instructor Efficiency:**
- Professor builds 3 model caches once → 30 students experiment for semester
- Reduced "how do I cache embeddings?" questions (0% vs ~2% discovery rate previously)
- Students submit Ray Tune hypertuning results across 10+ models without GPU access

**Technical Robustness:**
- Cache content hashes prevent stale data issues (no versioning bugs like akte-classifier)
- Pydantic catches composition errors before training starts
- Students don't hit "4D→2D tensor mismatch" runtime errors during training

**Composition as Natural Pattern:**
- Students think in terms of "concat this + stack that" rather than "build monolithic model"
- Students experiment with skip connections and parallel branches as standard practice
- Students understand "aggregation is a design decision" through experimentation

### Product Scope

**MVP (Minimum Viable Product):**

*Essential for Students:*
- Typed tensor system (1DTensor, 2DTensor, 3DTensor) with Pydantic contracts
- Chunk-level caching with 2DTensor storage (one cache per model)
- 4 HuggingFace models tested and curated:
  - sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (Dutch, auto-chunk)
  - Qwen/Qwen3-Embedding-0.6B (32k context, CPU-friendly)
  - sentence-transformers/LaBSE (109 languages)
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (fastest)
- Basic vectorizers: TextVectorizer (HF wrapper), RegexVectorizer, TFIDFVectorizer
- Core connectors: concat, stack
- Parameter-free aggregation: mean, max pooling for chunks
- DataLoader with automatic padding and attention masks
- Demo scripts showing complete workflows:
  1. Load cache from .vmcache file
  2. Add models (regex vectorizer + HF model)
  3. Hypertune and train (Ray Tune integration)
  4. Classify data and get metrics (F1, precision, recall)
- Quickstart examples (HF + regex, chunks → CNN, Ray Tune)
- Documentation covering core concepts
- Clear error messages guiding composition

*Technical Foundation:*
- Device management (GPU/MPS/CPU auto-detection)
- Content-hash based cache versioning
- Export/import .vmcache functionality
- pathlib for all file operations
- uv package management setup
- Basic linting configuration

**Growth Phase:**

*Expanded Model Library:*
- All 10 curated HF models with Dutch support tested and documented
- BAAI/bge-multilingual-gemma2 (8k context)
- intfloat/e5-mistral-7b-instruct (32k context, English)
- Qwen/Qwen3-Embedding-8B (32k context, 8B params)
- Additional sentence-transformers models

*Architectural Components:*
- SkipConnection (ResNet-style)
- ParallelBranch (Inception-style multi-path)
- Gate (mixture-of-experts routing)
- Students can build custom connectors (open-closed principle)

*Learned Aggregation:*
- Students implement Attention, CNN, LSTM aggregators for chunks
- Examples showing aggregation strategy comparison (parameter-free vs learned)
- Training utilities for learned aggregators

*Visualization:*
- ASCII/SVG/HTML composition diagrams
- Type flow visualization showing tensor shapes through pipeline
- Basic category theory notation for composition

*Hypertuning Integration:*
- Ray Tune examples with search spaces
- Hyperopt compatibility
- Example search spaces: model selection + aggregation + NN architecture
- Best practices for hypertuning with cached vectors

**Vision (Future Expansion):**

*Category Theory Visualization:*
- Commutative diagram validation
- Visual composition with drag-and-drop
- Mathematical notation for composition patterns

*Community Features:*
- Shared cache repository (students download professor's caches)
- Model performance benchmarks on common datasets
- Student-contributed aggregator examples

*Advanced Tooling:*
- Profiling tools showing bottlenecks in composition
- Automatic composition optimization suggestions
- Integration with experiment tracking (MLflow, Weights & Biases)

**Explicit Non-Goals:**

- **Not a fine-tuning framework**: VectorMesh uses pre-trained models as-is, composition over fine-tuning
- **Not a RAG framework**: No retrieval, document stores, or LLM chains (see LangChain/Haystack)
- **Not an AutoML platform**: Hypertuning support, but students control architecture decisions
- **Not model training from scratch**: Focus on pre-trained HuggingFace models as composable functions
- **Not production deployment tooling**: Educational focus, though production-grade code quality
- **Not a GUI/visual editor**: Code-first approach, though composition visualization for understanding

### Acceptance Criteria

**Students can successfully:**
1. Load a .vmcache file shared by professor and inspect metadata
2. Combine 3 vectorizers (e.g., BERT + regex + TF-IDF) using concat/stack
3. Get clear type error when connecting 2DTensor to 1DTensor-expecting layer
4. Experiment with parameter-free aggregation (mean vs max) without re-embedding
5. Build custom Attention aggregator using cached chunks
6. Run Ray Tune hypertuning across models + aggregation strategies on CPU
7. Export trained model and cache for reproducible results

**Instructors can successfully:**
1. Build caches for 3 HF models on GPU for 20k documents
2. Export .vmcache files and share with students (USB/cloud)
3. Verify students reproduce experiments using shared caches
4. See reduced support questions about HuggingFace API quirks and caching

**Technical validation:**
1. All 4 MVP models tested via HF MCP (pooling, context windows validated)
2. Pydantic contracts catch composition errors at definition time (not runtime)
3. Cache content hashing prevents stale data issues
4. DataLoader correctly pads variable chunk sizes with attention masks
5. Device management auto-detects GPU/MPS/CPU correctly
6. Linting passes with no warnings
7. All file operations use pathlib (zero os.path usage)
8. Package management works via uv (not pip)

**Qualitative indicators:**
- Students express "I finally understand when to aggregate early vs late"
- Students naturally think in composition terms ("concat this + stack that")
- Zero questions about "how do I cache embeddings?"
- Students submit hypertuning results showing 10+ model combinations tested


## User Journeys

### Journey 1: Sophie - Complete Classification Workflow (Happy Path)

Sophie is a master student in the legal text classification course with 20,000 documents to classify into 160 lawful facts. She understands the theory but dreads the infrastructure setup. On day one, she downloads three .vmcache files from the course portal (legal_mpnet.vmcache, legal_qwen.vmcache, legal_labse.vmcache) - about 150MB total.

Within 10 minutes, she's running her first experiment:

```python
from vectormesh import VectorCache, RegexVectorizer, concat
from vectormesh.connectors import Sequential
import torch.nn as nn

# Load pre-computed embeddings (instant)
mpnet_cache = VectorCache.load("legal_mpnet.vmcache")
features_mpnet = mpnet_cache.aggregate("mean")  # 1DTensor[N, 768]

# Add legal article patterns
regex = RegexVectorizer(patterns=["art\\.\\s*\\d+\\.\\d+"])
features_regex = regex(texts)  # 1DTensor[N, 50]

# Combine features
features = concat(features_mpnet, features_regex)  # 1DTensor[N, 818]

# Build simple classifier
model = Sequential([
    nn.Linear(818, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 160)
])
```

The code works on the first try. Sophie experiments with different combinations - adding the Qwen cache, trying max pooling instead of mean, adjusting the architecture. With Ray Tune, she hyperparameters across 15 combinations overnight on her laptop (no GPU needed - just loading cached vectors).

By week's end, she submits results showing F1=0.82 with mpnet+regex+two-layer NN. The breakthrough: she spent 90% of her time understanding composition strategies, not fighting infrastructure.

**This journey reveals requirements for:**
- VectorCache.load() with instant metadata inspection
- Simple aggregation: .aggregate("mean"), .aggregate("max")
- concat() for 1DTensor combination
- Clear type inference (1DTensor[768] + 1DTensor[50] → 1DTensor[818])
- Ray Tune integration with cached vectors
- Classification metrics (F1, precision, recall)

### Journey 2: Marcus - Building Custom CNN Aggregator (Extension)

Marcus reads about learned aggregation and wants to try CNN over chunks instead of simple mean pooling. He's nervous about getting tensor shapes wrong - in previous courses, he'd train for 30 minutes only to hit dimension mismatches.

He starts by examining the cached chunks:

```python
from vectormesh import VectorCache

cache = VectorCache.load("legal_mpnet.vmcache")
chunks = cache.get_chunks()  # 2DTensor[N, n_chunks, 768]
print(chunks.shape)  # "2DTensor[20000, 24, 768]"
```

The SDK documentation shows a clear extension pattern - inherit from `Aggregator` base class:

```python
from vectormesh.aggregators import Aggregator
from vectormesh.types import TwoDTensor, OneDTensor

class CNNAggregator(Aggregator):
    input_type = TwoDTensor[..., 768]  # Expects 2DTensor with last dim 768
    output_type = OneDTensor[..., 512]  # Produces 1DTensor with dim 512

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(768, 512, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, chunks, mask=None):
        # chunks: [batch, n_chunks, 768]
        x = chunks.transpose(1, 2)  # [batch, 768, n_chunks]
        x = self.conv(x)            # [batch, 512, n_chunks]
        x = self.pool(x)            # [batch, 512, 1]
        return x.squeeze(-1)        # [batch, 512] → 1DTensor
```

When Marcus connects it to his pipeline, VectorMesh validates types automatically:

```python
cnn_agg = CNNAggregator()
features = cnn_agg(chunks)  # ✅ Type check passes: 2DTensor[N,24,768] → 1DTensor[N,512]

# Later, tries to connect to Linear expecting 768 dims
classifier = nn.Linear(768, 160)
model = features >> classifier
# ❌ TypeError: Cannot connect 1DTensor[512] to Linear expecting 1DTensor[768]
# Hint: Output dimension is 512, but Linear expects 768. Update Linear(512, 160)?
```

The error message tells him exactly what's wrong and suggests the fix. Marcus adjusts the Linear layer and his model trains successfully. He's learned how aggregation works by experimenting, not debugging cryptic tensor errors.

**This journey reveals requirements for:**
- Base class pattern for extensibility (Aggregator, Vectorizer, Connector)
- Type annotations: input_type, output_type with Pydantic validation
- .get_chunks() API returning 2DTensor
- Type inference during composition (>> operator)
- Clear error messages showing expected vs actual dimensions with hints
- Documentation showing extension patterns

### Journey 3: Anna - Learning from Type Errors (Type Safety)

Anna understands embeddings conceptually but struggles with PyTorch shapes. She's trying to use cached chunks directly with a classifier:

```python
from vectormesh import VectorCache
import torch.nn as nn

cache = VectorCache.load("legal_mpnet.vmcache")
chunks = cache.get_chunks()  # 2DTensor[20000, 24, 768]

# She wants to classify directly
classifier = nn.Linear(768, 160)
model = chunks >> classifier
```

VectorMesh catches this immediately:

```
❌ TypeError: Cannot connect 2DTensor to Linear layer expecting 1DTensor

Expected: 1DTensor[..., 768]
Got:      2DTensor[..., 24, 768]

The Linear layer expects a 1-dimensional feature vector per sample, but you
provided a 2-dimensional tensor (chunks).

You need to aggregate chunks first. Try one of these:

  # Parameter-free aggregation (instant)
  features = chunks.mean(dim=1)  # Average across chunks → 1DTensor[N, 768]
  features = chunks.max(dim=1)   # Max pooling → 1DTensor[N, 768]

  # Learned aggregation (trainable)
  from vectormesh.aggregators import AttentionAggregator
  agg = AttentionAggregator(dim=768)
  features = agg(chunks)  # → 1DTensor[N, 768]
```

Anna reads the error, understands she needs aggregation, and fixes it:

```python
features = chunks.mean(dim=1)  # 1DTensor[20000, 768]
model = features >> classifier  # ✅ Type check passes
```

Later, she experiments with learned aggregation using the AttentionAggregator example from the docs. The error didn't just tell her what failed - it taught her a core concept about chunk-level representations.

**This journey reveals requirements for:**
- Type validation at composition time (not runtime)
- Rich error messages with context (expected vs actual)
- Suggested fixes in error messages
- Educational error messages that teach concepts
- Examples showing parameter-free vs learned aggregation
- dim parameter for aggregation operations

### Journey 4: Professor Raoul - Preparing Course Materials (Instructor)

Three weeks before the semester starts, Raoul needs to prepare caches for 100 students. He has GPU access and 20,000 legal documents. He selects 3 models: mpnet (best general performance), LaBSE (multilingual baseline), and Qwen3-0.6B (CPU-friendly for students without GPUs).

```python
from vectormesh import TextVectorizer, VectorCache

# Load texts once
texts = load_legal_corpus()  # 20,000 documents

# Build caches (runs on GPU automatically)
models = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "Qwen/Qwen3-Embedding-0.6B"
]

for model_name in models:
    vectorizer = TextVectorizer(model_name, auto_chunk=True, chunk_size=512)
    cache = VectorCache.create(
        texts=texts,
        vectorizer=vectorizer,
        name=f"legal_2024_{model_name.split('/')[-1]}"
    )
    # Processing: 100% |████████| 20000/20000 [15:23<00:00]
    # Cached 20,000 documents (chunks: 2DTensor[20000, 24, 768])
    # Cache size: 53.2 MB

    cache.export(f"legal_2024_{model_name.split('/')[-1]}.vmcache")
```

After 45 minutes, he has three .vmcache files (total 156MB). He uploads them to the course portal with metadata:

```python
cache.inspect()
# Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# Documents: 20,000
# Chunks: 2DTensor[20000, 24, 768] (avg 24 chunks/doc)
# Chunk size: 512 tokens
# Content hash: a7f3c9e2... (versioning)
# Created: 2024-08-15
# Size: 53.2 MB
```

Students download these files and can immediately experiment - no GPU needed, no recomputation. When a student asks "why 24 chunks?", Raoul points them to the metadata showing avg document length and 512 token chunks. When another student's results don't match the baseline, they compare content hashes and discover they're using an old cache version.

Throughout the semester, students submit 450+ experiments combining these 3 caches with regex patterns and different aggregation strategies. Zero questions about "how do I cache embeddings?" - the infrastructure just works.

**This journey reveals requirements for:**
- TextVectorizer with auto_chunk parameter
- VectorCache.create() API with progress bars
- Export/import .vmcache files with versioning
- .inspect() showing detailed metadata
- Content-hash versioning for cache validation
- Chunk statistics (avg chunks/doc, chunk size)
- Clear file size reporting
- GPU auto-detection and usage

### Journey Requirements Summary

These four journeys reveal the core capabilities VectorMesh must deliver:

**Caching Infrastructure:**
- VectorCache.load() and .create() with progress tracking
- Export/import .vmcache files with content-hash versioning
- .inspect() showing metadata (model, chunks, size, hash)
- .get_chunks() returning 2DTensor for advanced aggregation
- .aggregate("mean"|"max") for parameter-free aggregation
- Chunk-level storage (2DTensor[N, n_chunks, dim])

**Type System:**
- Typed tensors: 1DTensor, 2DTensor with dimension annotations
- Type validation at composition time (>> operator or .connect())
- Rich error messages with expected vs actual types
- Suggested fixes and educational hints in errors
- Type inference for composition chains

**Extensibility:**
- Base classes: Aggregator, Vectorizer, Connector
- input_type and output_type annotations with Pydantic
- Clear extension patterns in documentation
- Custom models integrate seamlessly with type checking

**Vectorizers:**
- TextVectorizer with auto_chunk parameter
- RegexVectorizer with pattern configuration
- concat() for 1DTensor combination
- Device management (GPU/MPS/CPU auto-detection)

**Training & Evaluation:**
- Ray Tune integration with cached vectors
- Standard metrics (F1, precision, recall)
- Sequential connector for building pipelines
- PyTorch nn.Module compatibility

**Documentation & DX:**
- Quickstart examples showing complete workflows
- Extension examples (custom aggregators, connectors)
- Error messages that teach concepts
- .inspect() for debugging and verification


## Innovation & Novel Patterns

### Detected Innovation Areas

**1. New Paradigm: Composition Over Fine-Tuning**

VectorMesh challenges the dominant paradigm in deep learning that pre-trained models are endpoints for transfer learning. Instead, it treats models as **composable functions** in a vector-space pipeline:

- **Traditional approach**: Fine-tune BERT for task → train for hours → deploy single model
- **VectorMesh approach**: BERT + regex + TF-IDF → compose signals → train small classifier on top

This shifts the abstraction level from "which model should I fine-tune?" to "which vector representations should I combine?"

**Key insight**: With limited data (20k documents), composition often beats fine-tuning. Combining multiple weak signals (BERT embeddings + legal article regex matches + TF-IDF) outperforms fine-tuning a single strong model.

**2. Chunk-Level Caching Architecture**

The most novel technical innovation: caching raw chunks as 2DTensor[N, n_chunks, dim] instead of aggregated embeddings as 1DTensor[N, dim].

**Why this matters:**
- **Standard caching**: Cache mean-pooled embeddings → locked into one aggregation strategy
- **VectorMesh caching**: Cache raw chunks → experiment with mean, max, Attention, CNN, LSTM without re-computing
- **Enables learning**: Students discover "when should I aggregate early (parameter-free) vs late (learned)?" through experimentation

**No precedent found**: HuggingFace users who discover caching (~2%) cache aggregated embeddings. Caching raw chunks for aggregation experimentation appears unexplored.

**3. Typed Tensor System for Machine Learning**

PyTorch uses generic `Tensor` - students connect CNNs to linear layers and get cryptic `RuntimeError: size mismatch` after 30-minute training runs.

VectorMesh introduces typed tensors (1DTensor, 2DTensor, 3DTensor) with Pydantic contracts:
- Errors at **composition time**, not runtime
- Educational error messages: "Cannot connect 2DTensor to Linear expecting 1DTensor. You need to aggregate chunks first. Try: features = chunks.mean(dim=1)"
- Type system teaches architectural constraints implicitly (1DTensor → NN, 2DTensor → CNN)

**Novel for ML**: Type-driven development is standard in software engineering but rare in ML pipelines.

**4. Hardware Democratization Through Shareable Caches**

80% of students lack local GPUs, creating inequality in experimentation capability.

VectorMesh's innovation: **Professor computes once (GPU) → students experiment infinitely (CPU)**
- 3 models × 20k docs = 156MB total cache size
- Students download .vmcache files → hypertune across 10+ model combinations on laptops
- Zero GPU queue time, zero re-computation, zero infrastructure friction

**Novel workflow**: Separates embedding computation (infrastructure) from experimentation (pedagogy). Students focus on composition strategies rather than fighting GPU access.

**5. Category Theory-Inspired Composition Primitives**

VectorMesh makes architectural patterns (skip connections, parallel branches, gates) first-class citizens with composition operators:
- `>>` operator for sequential composition with type checking
- `concat()`, `stack()` for tensor combination
- `SkipConnection`, `ParallelBranch`, `Gate` as reusable components

**Vision (Growth Phase)**: Commutative diagrams showing composition as mathematical structures. Students see architectures as category theory morphisms, not just code.

**Unexplored territory**: Bringing category theory abstraction to ML composition at this level.

### Market Context & Competitive Landscape

**Existing Tools:**

1. **HuggingFace Transformers**: Model zoo with 200k+ models
   - Problem: Complex APIs, no caching patterns, students discover caching ~2% of time
   - VectorMesh position: Makes HuggingFace composable with intelligent caching

2. **PyTorch nn.Sequential**: Sequential layer composition
   - Problem: Layer-level abstraction, generic tensors, runtime errors
   - VectorMesh position: Vector-level abstraction, typed tensors, definition-time errors

3. **LangChain/Haystack**: LLM chains and RAG frameworks
   - Problem: Focus on LLM orchestration, not vector composition
   - VectorMesh position: Focus on typed vector composition with chunk-level caching for experimentation

4. **Sentence-Transformers**: Excellent embedding library
   - Problem: No composition primitives, no type safety, no chunk-level caching
   - VectorMesh position: Builds on Sentence-Transformers with composition + types + caching

**Competitive Gap:**

No tool combines:
- HuggingFace model abstraction + chunk-level caching + typed tensors + composition primitives + hardware democratization

Closest: Students manually build caching (2%), use raw PyTorch (no type safety), fine-tune single models (miss composition benefits).

**Market Timing:**

- Pre-trained models are commoditized (200k+ on HuggingFace)
- GPU access is still scarce (80% students lack local GPU)
- Educational ML focuses on model training, not composition strategies
- Composition-first approach aligns with efficiency trends (smaller models, smarter combinations)

### Validation Approach

**Technical Validation (MVP):**

1. **Chunk-level caching delivers value:**
   - Measure: Students run 10+ aggregation strategy experiments without re-embedding
   - Success: 90%+ of students experiment with mean vs max vs learned aggregation
   - Metric: Zero "how do I cache embeddings?" questions (vs ~98% not discovering it)

2. **Type system catches errors early:**
   - Measure: Students hit 2DTensor→Linear errors at definition time, not runtime
   - Success: Zero 4D→2D tensor mismatch errors during training
   - Metric: 80%+ of students debug composition using error messages alone

3. **Composition beats fine-tuning (with limited data):**
   - Measure: 3-model composition (BERT+regex+TF-IDF) vs single fine-tuned BERT
   - Success: Composition achieves higher F1 on legal text classification (20k docs, 160 classes)
   - Baseline: akte-classifier results as reference

4. **Hardware democratization works:**
   - Measure: 30 students run Ray Tune hypertuning across 10+ model combinations on CPU
   - Success: All students submit hypertuning results (no GPU inequality)
   - Metric: Cache sizes stay under 200MB for shareability

**Pedagogical Validation (Growth Phase):**

5. **Students understand aggregation strategy trade-offs:**
   - Measure: Student reflections on "when to aggregate early vs late"
   - Success: 70%+ students articulate parameter-free vs learned aggregation trade-offs
   - Method: End-of-course survey or final project reports

6. **Composition becomes natural:**
   - Measure: Students think in terms of "concat this + stack that"
   - Success: Student-designed architectures use 3+ composition primitives
   - Method: Code review of final projects

### Risk Mitigation

**Risk 1: Chunk-level caching complexity doesn't justify benefits**

- **Fallback**: Degrade to aggregated embedding caching (standard approach)
- **Mitigation**: MVP validates chunk caching with student experiments before Growth Phase
- **Early signal**: If students never experiment with aggregation strategies, simplify to 1DTensor caching

**Risk 2: Type system feels heavyweight for Python users**

- **Fallback**: Make types optional (gradual typing approach)
- **Mitigation**: Error messages must be educational, not annoying
- **Early signal**: Student feedback on "helpful" vs "annoying" type errors

**Risk 3: Composition doesn't beat fine-tuning (students get worse results)**

- **Fallback**: VectorMesh still valuable for experimentation speed and type safety
- **Mitigation**: Provide baseline results showing composition advantages
- **Early signal**: akte-classifier already shows composition works (BERT+regex better than BERT alone)

**Risk 4: HuggingFace API changes break abstractions**

- **Fallback**: Pin to stable Transformers versions, use HF MCP for validation
- **Mitigation**: 10 curated models tested thoroughly, not 200k untested models
- **Early signal**: Test suite validates all 10 models before each release

**Risk 5: Students can't share caches (too large, infrastructure issues)**

- **Fallback**: Provide cloud storage or university network share
- **Mitigation**: Keep cache sizes under 200MB total (3 models × ~50MB each)
- **Early signal**: Monitor actual cache sizes in MVP, optimize if needed

**Risk 6: Category theory abstraction confuses rather than clarifies**

- **Fallback**: Keep category theory as optional visualization, not core API
- **Mitigation**: Vision phase only, not MVP - validate student interest first
- **Early signal**: Student feedback on composition diagrams in Growth Phase


## Developer Tool Specific Requirements

### Project-Type Overview

VectorMesh is a **Python SDK** for vector-space composition with chunk-level caching. As a developer tool, it must prioritize:
- **Clean API surface**: Intuitive imports, consistent naming, discoverable methods (ambitious but supported by MCP)
- **Type safety**: Full type hints, Pydantic contracts, educational error messages
- **Developer experience**: Fast feedback loops, MCP-powered exploration, focused zoo examples
- **Educational focus**: Students learn by doing with clear, isolated pattern examples

### Technical Architecture Considerations

**Language & Runtime:**
- **Python 3.12+** as minimum required version (no backward compatibility)
- Full type hints throughout codebase (PEP 484)
- **Type stubs (.pyi files) REQUIRED for MVP** - enables IDE autocomplete and validates type safety pitch
- Pin to PyTorch 2.x, refactor when PyTorch 3.0 stabilizes

**Package Management:**
- **uv** as default and recommended (enforced in CLAUDE.md)
- `pip install vectormesh[core]` - base installation
- `pip install vectormesh[full]` - includes hyperopt for hypertuning
- **No conda support** - reduces maintenance burden

**Dependency Structure:**
```toml
[project]
dependencies = [
    "torch>=2.0",
    "transformers>=4.30",
    "pydantic>=2.0",
    "tqdm>=4.65",
]

[project.optional-dependencies]
full = [
    "hyperopt>=0.2.7",  # Windows-friendly, TPE algorithm
]
dev = [
    "pytest>=7.0",
    "pyright>=1.1.0",  # Faster than mypy, strict mode
    "ruff>=0.1.0",
]
```

**IDE Integration:**
- Type hints + type stubs enable autocomplete in VS Code, PyCharm, Jupyter
- Pydantic models provide runtime validation + IDE support
- **pyright with strict mode** validates type flow through `>>` composition operator (faster than mypy)

**Jupyter Notebook Support:**
- **tqdm progress bars** during cache creation (works out of box)
- Optional: `rich` library for pretty `.inspect()` tables
- Keep it simple - no notebook magic commands needed

### Documentation Architecture (MCP-First Approach)

**VectorMesh MCP Server on Centralized VM:**

Instructor hosts VectorMesh MCP documentation on university VM for all students:

**Student Setup:**
```json
// One-time Claude Desktop config
{
  "mcpServers": {
    "vectormesh-docs": {
      "url": "https://vectormesh.university.edu/mcp",
      "description": "VectorMesh SDK Documentation & Zoo"
    }
  }
}
```

**Student Workflow:**
```
Student → Claude: "How do I cache chunks in VectorMesh?"
Claude → VM MCP: Queries VectorMesh documentation
VM MCP → Claude: Returns API ref + code example + zoo pattern
Claude → Student: Explains with official docs context
```

**VM Deployment Benefits:**
- Zero student setup friction (just configure URL)
- Instructor controls documentation version
- Usage analytics reveal confusing concepts
- Instant doc updates to all students
- Scales to 100+ students querying concurrently

**MCP Versioning Strategy:**
- **Always serve latest VectorMesh docs** (students on v0.2 → upgrade to v0.3)
- Educational context favors "always forward" over version fragmentation
- Breaking changes documented clearly with upgrade guides

**MCP Server Contents:**

1. **API Reference**: All classes, methods, parameters, return types
2. **Code Examples**: Indexed by use case (caching, composition, aggregation, hypertuning)
3. **Conceptual Guides**: Chunk-level caching, type system, composition patterns
4. **Troubleshooting**: Common errors with educational fixes
5. **Model Catalog**: 10 curated HF models with stats
6. **Zoo** (minimal, focused, one-concept-per-example):

**Aggregator Zoo** (show learned aggregation patterns):
- `WeightedMeanAggregator` - Simplest learned aggregation (weighted average)
- `AttentionAggregator` - Standard self-attention over chunks
- `CNNAggregator` - 1D convolution over chunk dimension
- `LSTMAggregator` - Sequential processing for temporal patterns

**Composition Pattern Zoo** (show architectural patterns):
- "Concat pattern" - BERT + regex → concat → classifier (simplest)
- "Skip connection" - Embeddings → transform → residual → classifier
- "Parallel branches" - Same cache → mean + CNN → concat (intermediate)
- "Multi-model gating" - 3 models → learned gate → classifier (advanced)

**Zoo Philosophy:**
- **One concept per example** - no complex patterns with 5 moving parts
- **Progressive complexity** - simple → intermediate → advanced
- **Written in MVP by instructor** - ensures quality and consistency
- **Each example teaches clearly** - code + when to use + variations

**Traditional Documentation** (for onboarding):
- **Quickstart**: Install → load cache → classify in 5 minutes
- **Installation Guide**: `uv pip install vectormesh[full]`
- **Demo Scripts**: 4 complete workflows (from user journeys)
- **Core Concepts**: Composition, chunk-level caching, type system

### API Surface Design

**Core Modules:**

```python
# Caching
from vectormesh import VectorCache
cache = VectorCache.load("model.vmcache")
cache.inspect()  # Rich table in Jupyter
chunks = cache.get_chunks()  # 2DTensor
features = cache.aggregate("mean")  # 1DTensor

# Vectorizers
from vectormesh import TextVectorizer, RegexVectorizer, TFIDFVectorizer
bert = TextVectorizer("bert-base", auto_chunk=True)
regex = RegexVectorizer(patterns=["art\\.\\s*\\d+"])

# Typed Tensors (with .pyi stubs)
from vectormesh.types import OneDTensor, TwoDTensor, ThreeDTensor

# Connectors
from vectormesh import concat, stack
from vectormesh.connectors import Sequential, SkipConnection, ParallelBranch

# Base Classes (for extension)
from vectormesh.aggregators import Aggregator
from vectormesh.vectorizers import Vectorizer

# Composition with >> operator (left-to-right flow)
model = bert(text) >> classifier  # Type-safe composition
# >> reads as "pipe into" (Unix aesthetic)
# No namespace collision with PyTorch * or @
```

**API Design Principles:**
- Flat imports for common operations
- Organized submodules for advanced use
- Consistent naming conventions
- Pydantic models for all inputs/outputs
- `>>` operator for composition (visual flow, no PyTorch conflicts)

### Installation Methods

**Recommended (uv):**
```bash
uv pip install vectormesh[full]  # With hyperopt
uv pip install vectormesh[core]  # Minimal
```

**Fallback (pip):**
```bash
pip install vectormesh[full]
```

**Verification:**
```python
import vectormesh
print(vectormesh.__version__)
vectormesh.check_installation()  # Validates deps, GPU detection
```

### Code Examples & Zoo

**Demo Scripts** (4 complete workflows):
1. Load cache → add regex → classify → F1 score
2. Load cache → custom CNN aggregator → train
3. Build cache on GPU → export → share
4. Hyperopt hypertuning across models (not Ray Tune - better Windows support)

**Zoo Examples** (in MCP, written by instructor in MVP):
- Progressive complexity (simple → advanced)
- One concept per example
- Code + use cases + variations
- Students learn patterns, then explore independently

**No Migration Guide:**
- Students haven't seen akte-classifier
- VectorMesh is new project from their perspective
- akte-classifier serves as internal reference only

### Implementation Considerations

**Distribution:**
- Publish to PyPI
- GitHub repo for issues/contributions
- Semantic versioning (0.1.0 → 1.0.0)

**Dependencies:**
- PyTorch 2.x (pinned, refactor for 3.0 when stable)
- HuggingFace Transformers 4.30+
- Pydantic 2.0+
- tqdm for progress
- **Hyperopt (not Ray Tune)** - better Windows support, TPE algorithm, simpler API

**Testing:**
- Unit tests for core components
- Integration tests with 2-3 HF models
- **Type checking with pyright in strict mode** (faster than mypy, validates `>>` operator type flow)
- Linting with ruff

**Type Safety Validation:**
- Ensure pyright (strict) can follow type inference through `>>` chained operations
- Type stubs must expose educational error messages at IDE level
- Students see type errors at composition time, not runtime

**pyright Configuration:**
```json
// pyrightconfig.json
{
  "typeCheckingMode": "strict",
  "reportMissingTypeStubs": true,
  "reportUnknownMemberType": true,
  "pythonVersion": "3.12"
}
```


## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Platform MVP

VectorMesh builds foundational infrastructure that enables student experimentation and future expansion. The MVP focuses on three core pillars that must work perfectly:

1. **Chunk-level caching infrastructure** - Novel 2DTensor storage enabling aggregation experimentation
2. **Typed tensor system** - Educational error messages catching composition mistakes at definition time
3. **Composition primitives** - Clean API for combining vectorizers with type safety

**Why Platform MVP:**
- Infrastructure must be solid before students build custom aggregators
- Semester 1 validates core assumptions (caching saves time, types prevent errors, composition works)
- Foundation enables Growth Phase expansion (more models, patterns, community contributions)
- Educational context demands reliability over feature breadth

**Resource Requirements:**
- **Development:** 1 senior Python engineer (you) with ML/education domain expertise
- **Infrastructure:** University VM for MCP server hosting
- **Timeline:** Build during course prep, validate with first cohort (30 students)
- **Success Validation:** Semester 1 student feedback + usage analytics from MCP queries

### MVP Feature Set (Phase 1)

**Core User Journeys Supported:**

All 4 documented journeys achievable with MVP:

1. **Sophie (Complete Workflow):** Load cache → add regex → classify → F1 score ✅
2. **Marcus (Custom Extension):** Build CNN aggregator using base classes ✅
3. **Anna (Type Safety):** 2DTensor→Linear error caught at composition time ✅
4. **Raoul (Cache Preparation):** Build 3 caches on GPU → export → share ✅

**Must-Have Capabilities:**

**Caching Infrastructure:**
- VectorCache.load() and .create() with progress bars (tqdm)
- Export/import .vmcache files with content-hash versioning
- .inspect() showing metadata (model, chunks, size, hash)
- .get_chunks() returning 2DTensor for learned aggregation
- .aggregate("mean"|"max") for parameter-free aggregation
- Chunk-level storage: 2DTensor[N, n_chunks, dim]
- DataLoader with automatic padding for variable chunk sizes

**Type System:**
- 1DTensor, 2DTensor, 3DTensor with Pydantic contracts
- Type stubs (.pyi files) for IDE autocomplete
- Type validation at composition time (>> operator)
- Educational error messages with expected vs actual types + hints
- pyright strict mode validation

**Vectorizers:**
- TextVectorizer with auto_chunk parameter
- RegexVectorizer with pattern configuration
- TFIDFVectorizer for baseline features
- 4 curated HF models tested thoroughly:
  - paraphrase-multilingual-mpnet-base-v2 (Dutch, auto-chunk)
  - Qwen/Qwen3-Embedding-0.6B (32k context, CPU-friendly)
  - LaBSE (109 languages)
  - paraphrase-multilingual-MiniLM-L12-v2 (fastest)

**Composition:**
- concat(), stack() for 1DTensor combination
- Sequential connector for pipelines
- >> operator for type-safe chaining
- Base classes: Aggregator, Vectorizer, Connector (for student extension)

**Developer Experience:**
- Python 3.12+ required
- uv package management (vectormesh[core], vectormesh[full])
- Hyperopt integration (Windows-friendly, TPE algorithm)
- Device management (GPU/MPS/CPU auto-detection)

**Documentation:**
- **MCP Server on VM:** API reference, code examples, troubleshooting, model catalog
- **Zoo (minimal):** 4 aggregators + 4 composition patterns (one concept each, written by instructor)
- **Traditional Docs:** Quickstart, installation guide, 4 demo scripts, core concepts

**Quality:**
- Unit tests for core components
- Integration tests with 2-3 HF models
- Type checking with pyright (strict mode)
- Linting with ruff
- pathlib for all file operations (never os)

### Post-MVP Features

**Phase 2 - Growth (Semester 2-3):**

**Expanded Model Library:**
- All 10 curated HF models tested and documented
- BAAI/bge-multilingual-gemma2 (8k context)
- intfloat/e5-mistral-7b-instruct (32k context, English)
- Qwen/Qwen3-Embedding-8B (32k context, 8B params)
- Additional sentence-transformers models

**Architectural Components:**
- SkipConnection (ResNet-style residual learning)
- ParallelBranch (Inception-style multi-path)
- Gate (mixture-of-experts routing with learned weights)
- Students contribute custom connectors (open-closed principle validated)

**Learned Aggregation:**
- Pre-built aggregators in zoo: Attention, CNN, LSTM, WeightedMean
- Examples showing parameter-free vs learned aggregation comparison
- Training utilities for learned aggregators
- Students build and share custom aggregators

**Visualization:**
- ASCII/SVG/HTML composition diagrams
- Type flow visualization showing tensor shapes through pipeline
- Basic category theory notation for composition

**Enhanced Hypertuning:**
- Ray Tune examples (if Windows issues resolved) or stick with Hyperopt
- Example search spaces: model selection + aggregation + architecture
- Best practices documentation for hypertuning with cached vectors

**Phase 3 - Expansion (Long-term Vision):**

**Category Theory Visualization:**
- Commutative diagram validation for composition correctness
- Visual composition with drag-and-drop interface
- Mathematical notation for composition patterns
- Students learn category theory through experimentation

**Community Features:**
- Shared cache repository (students download professor's caches globally)
- Model performance benchmarks on common datasets
- Student-contributed aggregator gallery
- Leaderboards for legal text classification performance

**Advanced Tooling:**
- Profiling tools showing bottlenecks in composition pipelines
- Automatic composition optimization suggestions
- Integration with experiment tracking (MLflow, Weights & Biases)
- Auto-generated composition diagrams from code

### Risk Mitigation Strategy

**Technical Risks:**

**Risk 1: Chunk-level caching complexity doesn't justify benefits**
- **Likelihood:** Low (akte-classifier shows caching value)
- **Impact:** High (core value prop)
- **Mitigation:** MVP validates with 30 students in Semester 1
- **Fallback:** Degrade to aggregated embedding caching (standard approach)
- **Early Signal:** If students never experiment with aggregation strategies (mean vs max vs learned)

**Risk 2: Type system feels heavyweight for Python users**
- **Likelihood:** Medium (Python culture prefers dynamic typing)
- **Impact:** Medium (can make types optional)
- **Mitigation:** Educational error messages must be helpful, not annoying
- **Fallback:** Gradual typing approach (types optional)
- **Early Signal:** Student feedback on "helpful" vs "annoying" type errors

**Risk 3: HuggingFace API changes break abstractions**
- **Likelihood:** Medium (HF evolves rapidly)
- **Impact:** Medium (fixable with updates)
- **Mitigation:** Pin to stable Transformers versions, test 10 curated models before each release
- **Fallback:** Version locking until migration complete
- **Early Signal:** Test suite catches breaking changes

**Market Risks:**

**Risk 4: Composition doesn't beat fine-tuning (students get worse results)**
- **Likelihood:** Low (akte-classifier validates composition advantage)
- **Impact:** High (undermines core thesis)
- **Mitigation:** Provide baseline results showing composition advantages with limited data
- **Fallback:** VectorMesh still valuable for experimentation speed and type safety
- **Early Signal:** Student F1 scores comparing composition vs single-model

**Risk 5: Students don't discover value beyond "just caching"**
- **Likelihood:** Low (MCP guides discovery)
- **Impact:** High (missed learning outcomes)
- **Mitigation:** MCP analytics show which concepts students explore, demo scripts guide experimentation
- **Fallback:** Enhanced documentation and mandatory assignments
- **Early Signal:** MCP query patterns reveal student engagement

**Resource Risks:**

**Risk 6: Students can't share caches (too large, infrastructure issues)**
- **Likelihood:** Low (cache sizes 17-53MB per model)
- **Impact:** Medium (reduces hardware democratization value)
- **Mitigation:** Keep cache sizes under 200MB total, provide university network share or cloud storage
- **Fallback:** Students build own caches locally (defeats GPU democratization)
- **Early Signal:** Monitor actual cache sizes in MVP, optimize if needed

**Risk 7: MCP VM server becomes bottleneck (100+ students querying)**
- **Likelihood:** Low (MCP designed for concurrent queries)
- **Impact:** Medium (degraded student experience)
- **Mitigation:** Docker container on university VM, monitor performance, scale horizontally if needed
- **Fallback:** Students run local MCP servers (loses analytics benefit)
- **Early Signal:** Query latency metrics from MCP server

**Risk 8: Category theory abstraction confuses rather than clarifies (Vision phase)**
- **Likelihood:** Medium (advanced mathematical concept)
- **Impact:** Low (Vision phase only, not MVP)
- **Mitigation:** Keep category theory as optional visualization, validate student interest in Growth Phase first
- **Fallback:** Skip category theory features entirely
- **Early Signal:** Student feedback on composition diagrams in Growth Phase

### Success Metrics for MVP Validation (Semester 1)

**Technical Validation:**
- 90%+ students run 10+ aggregation experiments without re-embedding (chunk caching delivers value)
- Zero 4D→2D tensor mismatch errors during training (type system works)
- 3-model composition achieves higher F1 than single fine-tuned model (composition thesis validated)
- 30 students submit hypertuning results on CPU (hardware democratization works)

**Pedagogical Validation:**
- Zero "how do I cache embeddings?" questions (vs ~98% not discovering it previously)
- 80%+ students debug composition using type error messages alone (educational errors work)
- Students express understanding of "when to aggregate early vs late" (learning outcome achieved)

**Infrastructure Validation:**
- MCP server handles 100+ concurrent queries without degradation
- Cache sizes stay under 200MB for 3 models (shareable via USB/cloud)
- pyright strict mode validates >> operator type flow (type safety pitch holds)

**Decision Point After Semester 1:**
- If technical + pedagogical validation succeeds → Proceed to Growth Phase
- If chunk caching shows no value → Pivot to aggregated caching
- If type system feels heavyweight → Make types optional
- If composition doesn't beat fine-tuning → Refocus on experimentation speed value


## Functional Requirements

### Cache Management

- FR1: Instructors can create vector caches from text corpora using HuggingFace models
- FR2: Instructors can export vector caches to .vmcache files for distribution
- FR3: Students can load .vmcache files to access pre-computed embeddings
- FR4: Users can inspect cache metadata (model name, document count, chunk dimensions, content hash, creation date, file size)
- FR5: Users can retrieve raw chunk embeddings as 2DTensor for custom aggregation
- FR6: Users can aggregate chunks using parameter-free methods (mean, max pooling)
- FR7: System can validate cache integrity using content-hash versioning
- FR8: System can automatically pad variable-length chunks in DataLoader batches with attention masks

### Vectorization & Feature Extraction

- FR9: Users can vectorize text using 4 curated HuggingFace models (mpnet, Qwen3-0.6B, LaBSE, MiniLM)
- FR10: Users can configure automatic text chunking with specified chunk size for long documents
- FR11: Users can extract regex-based features from text using custom pattern configurations
- FR12: Users can extract TF-IDF features from text corpora
- FR13: System can auto-detect and utilize available compute devices (GPU, MPS, CPU)
- FR14: System can display progress bars during vectorization operations

### Composition & Type Safety

- FR15: Users can concatenate 1DTensors to create combined feature vectors
- FR16: Users can stack tensors to create higher-dimensional representations
- FR17: Users can compose vectorizers and layers using >> operator with type validation
- FR18: Users can build sequential pipelines combining multiple vectorizers and connectors
- FR19: System can validate tensor dimension compatibility at composition time (not runtime)
- FR20: System can provide educational error messages showing expected vs actual tensor types with fix suggestions
- FR21: System can enforce type contracts using Pydantic for all vectorizer inputs/outputs

### Model Extension & Customization

- FR22: Developers can create custom aggregators by inheriting from Aggregator base class
- FR23: Developers can create custom vectorizers by inheriting from Vectorizer base class
- FR24: Developers can create custom connectors by inheriting from Connector base class
- FR25: Developers can specify input_type and output_type annotations for custom components
- FR26: System can validate custom component type contracts at composition time
- FR27: Users can access 4 pre-built aggregators from zoo (WeightedMean, Attention, CNN, LSTM)
- FR28: Users can access 4 composition pattern examples from zoo (Concat, SkipConnection, ParallelBranch, Gate)

### Hyperparameter Tuning & Training

- FR29: Users can define hyperparameter search spaces for model selection, aggregation strategies, and architecture
- FR30: Users can execute hyperparameter tuning using Hyperopt with TPE algorithm
- FR31: Users can hypertune across multiple cached models without re-embedding
- FR32: Users can evaluate classification performance with standard metrics (F1, precision, recall)
- FR33: System can integrate with PyTorch nn.Module for seamless model training

### Documentation & Learning Support

- FR34: Students can query VectorMesh documentation via MCP server using natural language
- FR35: MCP server can provide API reference for all classes, methods, and parameters
- FR36: MCP server can provide code examples indexed by use case (caching, composition, aggregation, hypertuning)
- FR37: MCP server can provide conceptual guides explaining chunk-level caching, type system, and composition patterns
- FR38: MCP server can provide troubleshooting guides for common errors with solutions
- FR39: MCP server can provide model catalog listing 10 curated HuggingFace models with statistics
- FR40: Instructors can host MCP server on centralized VM for student access
- FR41: Users can access 4 demo scripts showing complete workflows (load cache, extend models, hypertune, classify)
- FR42: Users can access quickstart documentation for 5-minute onboarding
- FR43: Users can access installation guide covering uv and pip package management

### Development & Distribution

- FR44: Developers can install VectorMesh via uv pip install with core and full dependency options
- FR45: Developers can verify installation and validate dependencies including GPU detection
- FR46: Developers can access type stubs (.pyi files) for IDE autocomplete support
- FR47: System can validate code using pyright in strict type checking mode
- FR48: System can enforce code quality using ruff linter
- FR49: System can run unit tests for core components
- FR50: System can run integration tests with 2-3 HuggingFace models
- FR51: Users can access semantic versioning for stable API contracts


## Non-Functional Requirements

### Performance

**Cache Operations:**
- NFR1: Cache loading from .vmcache files completes within 5 seconds for files up to 200MB
- NFR2: Cache creation displays real-time progress updates via tqdm with <1 second refresh rate
- NFR3: Cache.inspect() metadata retrieval completes within 1 second

**Type Checking:**
- NFR4: Type validation at composition time completes within 100ms for typical pipelines (3-5 components)
- NFR5: Educational error message generation completes within 50ms to provide immediate feedback

**MCP Documentation Server:**
- NFR6: MCP server responds to student queries within 2 seconds for 95th percentile
- NFR7: MCP server maintains <2 second response time with 100 concurrent student queries
- NFR8: MCP server uptime ≥99% during semester (allows brief maintenance windows)

**Vectorization:**
- NFR9: Progress bars update at minimum 1 Hz during vectorization to show responsiveness
- NFR10: Device auto-detection (GPU/MPS/CPU) completes within 500ms on first import

### Integration

**HuggingFace Transformers:**
- NFR11: Compatible with HuggingFace Transformers 4.30+ (tested with each minor version release)
- NFR12: All 4 MVP models (mpnet, Qwen3-0.6B, LaBSE, MiniLM) validated before each VectorMesh release
- NFR13: Graceful degradation when HuggingFace model unavailable (clear error with fallback suggestions)

**PyTorch:**
- NFR14: Compatible with PyTorch 2.x (2.0 through 2.4+)
- NFR15: Seamless integration with PyTorch nn.Module for training pipelines
- NFR16: Automatic device management compatible with PyTorch CUDA, MPS, and CPU backends

**Hyperopt:**
- NFR17: Compatible with Hyperopt 0.2.7+ for hyperparameter tuning
- NFR18: Search spaces integrate with cached vectors without re-embedding overhead
- NFR19: TPE algorithm execution compatible with Windows, macOS, Linux

**MCP Protocol:**
- NFR20: MCP server implementation compliant with MCP specification v1.0+
- NFR21: MCP responses formatted as valid JSON with proper error handling
- NFR22: MCP server supports concurrent connections from 100+ Claude Desktop clients

### Maintainability

**Type Safety:**
- NFR23: 100% type hint coverage for all public APIs
- NFR24: pyright strict mode validates with zero type errors
- NFR25: Type stubs (.pyi files) available for all public modules

**Code Quality:**
- NFR26: Ruff linter passes with zero violations on all committed code
- NFR27: All public functions include docstrings following Google style guide
- NFR28: Code complexity (cyclomatic) ≤10 for individual functions

**Testing:**
- NFR29: Unit test coverage ≥80% for core modules (caching, types, composition)
- NFR30: Integration tests validate 2-3 HuggingFace models with each release
- NFR31: All tests pass on Python 3.12+ across Windows, macOS, Linux

**Dependency Management:**
- NFR32: Dependencies specify minimum versions without upper bounds in pyproject.toml
- NFR33: Dependency security vulnerabilities addressed within 30 days of disclosure
- NFR34: pathlib used exclusively for file operations (zero os.path usage)

**Versioning & Compatibility:**
- NFR35: Semantic versioning (SemVer) enforced for all releases
- NFR36: Breaking API changes only in major version bumps with migration guide
- NFR37: Deprecation warnings provided at least one minor version before removal

