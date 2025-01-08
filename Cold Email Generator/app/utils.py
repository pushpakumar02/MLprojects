import re

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove special characters (keep only alphanumeric characters and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)

    # Trim leading and trailing whitespace
    text = text.strip()

    # Remove extra whitespace between words
    text = ' '.join(text.split())

    return text
