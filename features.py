import re
import textstat


# List of common suspicious words in smishing messages
SUSPICIOUS_WORDS = [
    "urgent", "immediately", "act now", "attention", "important",
    "limited time", "alert", "hurry", "final notice", "winner",
    "claim now", "call now", "verify", "action required", "reply now",
    "warning", "IRS", "I.R.S", "arrest", "account", "free", "bank",
    "click", "claim", "limited", "lottery"
]


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, text):
        # Check if contains URL
        url = self.contains_url(text)
        contains_phone = self.contains_phone_number(text)
        contains_suspicious = self.contains_suspicious_words(text)
        readability_score = textstat.flesch_reading_ease(text)

        features = [url, contains_phone, contains_suspicious, readability_score]

        return features

    @staticmethod
    def contains_url(text):
        """Check if the SMS contains a URL"""
        return bool(re.search(r"https?://\S+|www\.\S+", text))

    @staticmethod
    def contains_phone_number(text):
        """Check if the SMS contains a phone number"""
        # Regular expression pattern to match phone numbers
        phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{1,4}\)[-.\s]?)?(?:\d{1,4}[-.\s]?){2,4}\d\b')

        # Find all phone numbers in the text
        phone_numbers = phone_pattern.findall(text)

        return bool(phone_numbers)

    @staticmethod
    def contains_suspicious_words(text):
        """Check if an SMS contains suspicious words."""
        text = text.lower()
        for word in SUSPICIOUS_WORDS:
            if re.search(r'\b' + re.escape(word) + r'\b', text):  # Match whole words
                return 1
        return 0