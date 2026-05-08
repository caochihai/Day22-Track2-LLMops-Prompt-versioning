import re
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ── 1. Imports ───────────────────────────────────────────────────────────────
from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
    ValidationResult,
)
from guardrails.validator_base import OnFailAction


# ── 2. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).

    Patterns detected:
      - EMAIL: xxx@xxx.xxx
      - PHONE: (123) 456-7890 or 123-456-7890
      - SSN:   123-45-6789
      - CREDIT CARD: 1234 5678 9012 3456 (or dashes)
    """

    # Define regex patterns as class constants
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        """
        Check value for PII; if found, return FailResult with fix_value (redacted).
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append(pii_type)

        if found_pii:
            print(f"  ⚠️  Detected PII: {list(set(found_pii))}")
            return FailResult(
                error_message=f"PII detected: {list(set(found_pii))}",
                fix_value=redacted_text
            )
        return PassResult()


# ── 3. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Attempt to repair a JSON string.
        """
        text = text.strip()

        # Remove markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Single quotes → double quotes
        text = text.replace("'", '"')

        # Remove trailing commas
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        """
        Try to parse value as JSON. If it fails, return FailResult with repaired fix_value.
        """
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # Try repair
        try:
            repaired_text = self._repair(value)
            # Verify if it's valid now
            json.loads(repaired_text)
            print(f"  🔧 JSON repaired successfully")
            return FailResult(
                error_message="Invalid JSON format, but repaired.",
                fix_value=repaired_text
            )
        except json.JSONDecodeError as e:
            return FailResult(error_message=f"Invalid JSON after repair attempt: {e}")


# ── 4. PII Guard demo ────────────────────────────────────────────────────────
def demo_pii_guard():
    """
    Create a Guard with PIIDetector and test 6 sample texts.
    """
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    # Create the guard
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        # Validate the text
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 5. JSON Guard demo ────────────────────────────────────────────────────────
def demo_json_guard():
    """
    Create a Guard with JSONFormatter and test 5 sample strings.
    """
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    # Create the guard
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        # Validate the text
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        # Show output (formatted JSON or error message)
        output_preview = str(result.validated_output)[:60].replace('\n', ' ')
        print(f"  Output: {output_preview}")


# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Step 4 complete!")


if __name__ == "__main__":
    main()
