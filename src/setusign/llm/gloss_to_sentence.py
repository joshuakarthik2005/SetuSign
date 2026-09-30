"""
Gloss-to-Sentence Converter for SetuSign
==========================================
Converts sign language glosses (individual recognized words) into
natural English/Hindi sentences using template-based generation.

In production, this would use an on-device LLM (e.g., Phi-3-mini via ONNX).
For the demo, we use a lightweight rule-based approach that works offline.

Examples:
    ["hello"] -> "Hello!"
    ["bank", "go"] -> "I want to go to the bank."
    ["doctor", "time"] -> "What time is the doctor available?"
"""

import re
from typing import List, Optional


# Service counter phrase templates
TEMPLATES = {
    # Greetings
    "hello": "Hello!",
    "goodmorning": "Good morning!",
    "thankyou": "Thank you!",

    # Places / Destinations
    "bank": "I need to go to the bank.",
    "court": "I need to visit the court.",
    "storeorshop": "I want to go to the store.",

    # People
    "father": "My father is here.",
    "brother": "My brother needs help.",
    "boy": "The boy needs assistance.",
    "girl": "The girl needs assistance.",

    # Actions / Needs
    "pen": "I need a pen.",
    "cellphone": "Can I use a phone?",
    "shoes": "I need new shoes.",
    "tshirt": "I am looking for a T-shirt.",
    "hat": "I need a hat.",

    # Descriptions
    "hot": "It is very hot.",
    "good": "That is good.",
    "happy": "I am happy.",
    "quiet": "Please be quiet.",
    "loud": "That is too loud.",
    "biglarge": "It is too big.",
    "smalllittle": "It is too small.",
    "long": "It is too long.",
    "short": "It is too short.",
    "new": "I need something new.",
    "dry": "It is dry.",
    "red": "The red one, please.",
    "white": "The white one, please.",
    "black": "The black one, please.",

    # Time
    "monday": "It is on Monday.",
    "time": "What time is it?",
    "year": "Which year?",
    "summer": "It is summer.",

    # Animals
    "bird": "I see a bird.",
    "cow": "There is a cow.",
    "dog": "There is a dog.",
    "car": "I need a car.",

    # Complex
    "teacher": "I need to speak with the teacher.",
    "priest": "I need to see the priest.",
    "election": "When is the election?",
    "death": "There has been a death in the family.",
    "fall": "Someone has fallen.",
    "paint": "I need paint.",
    "fan": "Please turn on the fan.",
    "window": "Please open the window.",
    "house": "I need help with my house.",
    "trainticket": "I need a train ticket.",

    # Pronouns
    "i": "I",
    "it": "It",
    "youplural": "All of you",
}

# Multi-gloss patterns (ordered by priority)
MULTI_GLOSS_PATTERNS = [
    # Greetings + place
    (["hello", "bank"], "Hello! I need help at the bank."),
    (["hello", "doctor"], "Hello! I need to see a doctor."),
    (["hello", "teacher"], "Hello! I need to speak with the teacher."),

    # Need + item
    (["i", "pen"], "I need a pen, please."),
    (["i", "cellphone"], "Can I use a phone, please?"),
    (["i", "trainticket"], "I need a train ticket, please."),
    (["i", "car"], "I need a car, please."),

    # Questions
    (["time", "doctor"], "What time is the doctor available?"),
    (["time", "bank"], "What time does the bank open?"),
    (["time", "trainticket"], "What time is the train?"),

    # Descriptions
    (["hot", "fan"], "It is hot. Please turn on the fan."),
    (["hot", "window"], "It is hot. Please open the window."),

    # Actions
    (["thankyou", "goodbye"], "Thank you! Goodbye!"),
    (["hello", "goodmorning"], "Good morning! Hello!"),
]


class GlossToSentence:
    """
    Converts sign language glosses into natural sentences.
    
    Uses template matching with fallback to simple concatenation.
    Designed to be replaced by an on-device LLM in production.
    """

    def __init__(self):
        self.templates = TEMPLATES
        self.multi_patterns = MULTI_GLOSS_PATTERNS
        self.history = []  # Recent recognized glosses
        self.sentence_buffer = []

    def convert(self, glosses: List[str]) -> str:
        """
        Convert a list of glosses to a natural sentence.
        
        Args:
            glosses: List of recognized sign glosses (lowercase)
        
        Returns:
            Natural language sentence
        """
        if not glosses:
            return ""

        # Normalize
        glosses = [g.lower().strip() for g in glosses if g.strip()]

        # Try multi-gloss patterns first
        for pattern, sentence in self.multi_patterns:
            if all(p in glosses for p in pattern):
                return sentence

        # Single gloss lookup
        if len(glosses) == 1:
            gloss = glosses[0]
            if gloss in self.templates:
                return self.templates[gloss]
            return gloss.capitalize() + "."

        # Multi-gloss: combine individual templates
        parts = []
        for gloss in glosses:
            if gloss in self.templates:
                parts.append(self.templates[gloss])
            else:
                parts.append(gloss.capitalize())

        return " ".join(parts)

    def add_to_context(self, gloss: str):
        """Add a recognized gloss to the context window."""
        self.history.append(gloss)
        # Keep last 10
        if len(self.history) > 10:
            self.history.pop(0)

    def get_contextual_sentence(self, current_gloss: str) -> str:
        """
        Generate a sentence considering recent context.
        
        If the user signs "bank" followed by "time",
        this generates "What time does the bank open?"
        """
        self.add_to_context(current_gloss)

        # Try recent pairs
        if len(self.history) >= 2:
            recent = self.history[-2:]
            combined = self.convert(recent)
            if combined != " ".join(r.capitalize() for r in recent):
                return combined

        # Fall back to single gloss
        return self.convert([current_gloss])

    def get_all_templates(self) -> dict:
        """Return all available templates."""
        return dict(self.templates)
