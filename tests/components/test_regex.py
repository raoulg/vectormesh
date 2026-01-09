import pytest
import re
import torch
from vectormesh.components.regex import RegexVectorizer
from vectormesh.exceptions import VectorMeshError
from vectormesh.types import TwoDTensor

def test_initialization_valid():
    """Test standard initialization with valid regex patterns."""
    patterns = {
        "email": r"[\w\.-]+@[\w\.-]+",
        "digits": r"\d+"
    }
    vec = RegexVectorizer(patterns=patterns)
    assert vec.patterns == patterns
    
def test_initialization_frozen():
    """Test that the component is immutable (frozen)."""
    vec = RegexVectorizer(patterns={"digits": r"\d+"})
    with pytest.raises(Exception):
        vec.patterns = {"new": "pattern"}

def test_initialization_invalid_regex():
    """Test that invalid regex patterns raise VectorMeshError with hint."""
    patterns = {
        "broken": r"[" # Unclosed character class
    }
    with pytest.raises(VectorMeshError) as exc_info:
        RegexVectorizer(patterns=patterns)
    
    assert "Invalid regex pattern" in str(exc_info.value)
    assert exc_info.value.hint is not None
    assert exc_info.value.fix is not None

def test_vectorization_logic_dutch_law():
    """Test vectorization using the specific Dutch law examples from the story."""
    patterns = {
        "article_ref": r"(?i)m?artikelen?\s+(?:\d+(?::\d+)?(?:(?:\s+lid\s+\d+)|(?:\s+en\s+\d+(?::\d+)?))?)\b",
        "wetboek": r"(?i)(?:Burgerlijk\s+Wetboek|Wet\s+Bodembescherming)"
    }
    # Note: These are simple regex proxies for the complex examples, 
    # focusing on capturing the intent of the story's requirements rather than perfect NLP.
    # We refine regexes to match the specific story examples.
    
    # Refined patterns for the specific examples:
    # 1. "artikel 265 Boek 3 van het Burgerlijk Wetboek" -> Matches article + wetboek
    # 2. "artikel 7:2 Burgerlijk Wetboek" -> Matches article + wetboek
    # 3. "artikel 7:26 lid 3 van het Burgerlijk Wetboek" -> Matches article + wetboek
    # 4. "artikelen 6:251 en 6:252 Burgerlijk Wetboek" -> Matches article(s) + wetboek
    # 5. "artikel 55 Wet Bodembescherming" -> Matches article + wetboek
    # 6. "artikel 6.5 en 31.5 van de koopovereenkomst" -> Should match article pattern (maybe) BUT NOT wetboek pattern.
    
    # Let's use more specific patterns as implied by the story to strictly distinguish "Wetboek" context.
    # The story implies we want to find "artikelen" mentions especially when connected to Law.
    # But strictly speaking, a RegexVectorizer simply applies the patterns given.
    # So we define patterns that we EXPECT to trigger on these inputs.
    
    patterns = {
        "law_article": r"(?i)artikelen?\s+\d+(?::\d+)?",
        "wetboek_context": r"(?i)(?:Burgerlijk\s+Wetboek|Wet\s+Bodembescherming)"
    }
    
    vec = RegexVectorizer(patterns=patterns)
    
    positive_texts = [
        "artikel 265 Boek 3 van het Burgerlijk Wetboek",
        "artikel 7:2 Burgerlijk Wetboek",
        "artikel 7:26 lid 3 van het Burgerlijk Wetboek",
        "artikelen 6:251 en 6:252 Burgerlijk Wetboek",
        "artikel 55 Wet Bodembescherming"
    ]
    
    negative_text = "artikel 6.5 en 31.5 van de koopovereenkomst"
    
    batch = positive_texts + [negative_text]
    # Expectation: 
    # Positives should match BOTH or at least ONE based on the patterns.
    # Specific requirement: "find artikelen... because we need to find also the mention of the wetboek".
    # This implies we might want a pattern that captures the combination, OR just features for each.
    # The Vectorizer just outputs features.
    
    vectors = vec(batch)
    
    assert isinstance(vectors, torch.Tensor)
    assert vectors.shape == (6, 2)
    assert vectors.dtype == torch.float32
    
    # Check that "Wetboek" pattern matches for positives but NOT for negative
    wetboek_col_idx = list(patterns.keys()).index("wetboek_context")
    
    # Positives have wetboek mention
    assert torch.all(vectors[:5, wetboek_col_idx] == 1.0)
    
    # Negative does NOT have wetboek mention
    assert vectors[5, wetboek_col_idx] == 0.0

def test_no_matches():
    patterns = {"foo": r"foo"}
    vec = RegexVectorizer(patterns=patterns)
    res = vec(["bar", "baz"])
    assert torch.all(res == 0.0)

def test_multiple_patterns_same_text():
    patterns = {
        "hello": r"hello",
        "world": r"world"
    }
    vec = RegexVectorizer(patterns=patterns)
    res = vec(["hello world"])
    assert torch.all(res == 1.0)
