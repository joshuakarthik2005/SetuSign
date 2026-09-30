"""Test gloss-to-sentence converter."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.setusign.llm.gloss_to_sentence import GlossToSentence


def test_single_gloss():
    """Test single gloss conversion."""
    print("1. Testing single gloss conversion...")
    g2s = GlossToSentence()

    tests = [
        (["hello"], "Hello!"),
        (["bank"], "I need to go to the bank."),
        (["hot"], "It is very hot."),
        (["thankyou"], "Thank you!"),
    ]

    for glosses, expected in tests:
        result = g2s.convert(glosses)
        assert result == expected, f"Expected '{expected}', got '{result}'"
        print(f"   {glosses} -> {result}")

    print("   PASS")


def test_multi_gloss():
    """Test multi-gloss pattern matching."""
    print("\n2. Testing multi-gloss patterns...")
    g2s = GlossToSentence()

    tests = [
        (["hello", "bank"], "Hello! I need help at the bank."),
        (["time", "doctor"], "What time is the doctor available?"),
        (["hot", "fan"], "It is hot. Please turn on the fan."),
    ]

    for glosses, expected in tests:
        result = g2s.convert(glosses)
        assert result == expected, f"Expected '{expected}', got '{result}'"
        print(f"   {glosses} -> {result}")

    print("   PASS")


def test_context():
    """Test contextual sentence generation."""
    print("\n3. Testing contextual generation...")
    g2s = GlossToSentence()

    # Sign "time" then "bank" -> should give contextual sentence
    s1 = g2s.get_contextual_sentence("time")
    print(f"   'time' -> {s1}")
    s2 = g2s.get_contextual_sentence("bank")
    print(f"   'bank' (after 'time') -> {s2}")

    print("   PASS")


def test_unknown_gloss():
    """Test unknown gloss handling."""
    print("\n4. Testing unknown gloss...")
    g2s = GlossToSentence()
    result = g2s.convert(["xyz_unknown"])
    assert "Xyz_unknown" in result
    print(f"   ['xyz_unknown'] -> {result}")
    print("   PASS")


def main():
    print("=" * 50)
    print("Gloss-to-Sentence Tests")
    print("=" * 50)

    test_single_gloss()
    test_multi_gloss()
    test_context()
    test_unknown_gloss()

    print(f"\n{'='*50}")
    print("ALL GLOSS TESTS PASSED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
