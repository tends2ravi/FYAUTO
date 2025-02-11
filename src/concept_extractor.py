"""
Concept extraction module using NLP techniques.
"""
from typing import List, Dict, Tuple
import spacy
from collections import defaultdict
from loguru import logger
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ConceptExtractor:
    """Extract key concepts from text for visual generation."""
    
    def __init__(self):
        """Initialize the concept extractor."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.stop_words = set(stopwords.words('english'))
        self.visual_pos = {'NOUN', 'PROPN', 'ADJ'}  # Parts of speech that are visually representable
    
    def extract_concepts(
        self,
        text: str,
        max_concepts: int = 3,
        min_relevance: float = 0.3
    ) -> List[Dict[str, any]]:
        """
        Extract key concepts from text that would make good visuals.
        
        Args:
            text: Input text
            max_concepts: Maximum number of concepts to extract
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of concepts with metadata
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract candidate concepts
        candidates = self._extract_candidates(doc)
        
        # Score candidates
        scored_candidates = self._score_candidates(candidates, doc)
        
        # Filter and rank concepts
        concepts = self._rank_concepts(scored_candidates, max_concepts, min_relevance)
        
        return concepts
    
    def _extract_candidates(self, doc) -> List[Tuple[str, str]]:
        """Extract candidate concepts and their types."""
        candidates = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in {'PRODUCT', 'ORG', 'PERSON', 'GPE', 'LOC', 'WORK_OF_ART'}:
                candidates.append((ent.text, 'entity'))
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            # Filter out chunks that are mostly stop words
            content_words = [token for token in chunk if not token.is_stop]
            if content_words:
                candidates.append((chunk.text, 'noun_phrase'))
        
        # Extract important individual words
        for token in doc:
            if (token.pos_ in self.visual_pos and 
                not token.is_stop and 
                token.text.lower() not in self.stop_words):
                candidates.append((token.text, 'word'))
        
        return candidates
    
    def _score_candidates(
        self,
        candidates: List[Tuple[str, str]],
        doc
    ) -> List[Dict[str, any]]:
        """Score candidates based on multiple factors."""
        scored = []
        
        # Get document statistics
        word_freq = defaultdict(int)
        for token in doc:
            if not token.is_stop:
                word_freq[token.lower_] += 1
        
        for text, concept_type in candidates:
            # Create TextBlob for sentiment analysis
            blob = TextBlob(text)
            
            # Calculate base score
            base_score = self._calculate_base_score(text, concept_type, word_freq)
            
            # Calculate visual score
            visual_score = self._calculate_visual_score(text)
            
            # Calculate sentiment intensity
            sentiment = abs(blob.sentiment.polarity)
            
            # Calculate final score
            final_score = (base_score * 0.5 + 
                         visual_score * 0.3 + 
                         sentiment * 0.2)
            
            scored.append({
                'text': text,
                'type': concept_type,
                'score': final_score,
                'visual_score': visual_score,
                'sentiment': blob.sentiment.polarity
            })
        
        return scored
    
    def _calculate_base_score(
        self,
        text: str,
        concept_type: str,
        word_freq: Dict[str, int]
    ) -> float:
        """Calculate base importance score."""
        # Weight by concept type
        type_weights = {
            'entity': 1.0,
            'noun_phrase': 0.8,
            'word': 0.6
        }
        
        # Calculate frequency score
        freq_score = sum(word_freq[word.lower()] 
                        for word in text.split())
        freq_score = min(freq_score / 10, 1.0)  # Normalize
        
        return type_weights[concept_type] * freq_score
    
    def _calculate_visual_score(self, text: str) -> float:
        """Calculate how visually representable a concept is."""
        doc = self.nlp(text)
        
        # Words that suggest good visuals
        visual_indicators = {
            'show', 'display', 'look', 'appear', 'visual',
            'image', 'picture', 'scene', 'view', 'design'
        }
        
        # Abstract words that are hard to visualize
        abstract_indicators = {
            'concept', 'idea', 'theory', 'thought', 'philosophy',
            'logic', 'meaning', 'reason', 'principle'
        }
        
        # Count visual and abstract words
        visual_count = sum(1 for token in doc 
                         if token.lower_ in visual_indicators)
        abstract_count = sum(1 for token in doc 
                           if token.lower_ in abstract_indicators)
        
        # Calculate score
        score = 0.5  # Base score
        score += 0.1 * visual_count
        score -= 0.1 * abstract_count
        
        return max(0.1, min(score, 1.0))  # Clamp between 0.1 and 1.0
    
    def _rank_concepts(
        self,
        scored_candidates: List[Dict[str, any]],
        max_concepts: int,
        min_relevance: float
    ) -> List[Dict[str, any]]:
        """Rank and filter concepts."""
        # Remove duplicates (keep highest scored)
        unique_concepts = {}
        for concept in scored_candidates:
            text = concept['text'].lower()
            if (text not in unique_concepts or 
                concept['score'] > unique_concepts[text]['score']):
                unique_concepts[text] = concept
        
        # Sort by score
        ranked = sorted(
            unique_concepts.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Filter by minimum relevance
        filtered = [c for c in ranked if c['score'] >= min_relevance]
        
        return filtered[:max_concepts] 