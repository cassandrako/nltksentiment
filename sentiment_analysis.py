#inspired by https://github.com/Jordan396/Twitter-Sentiment-Analysis

import nltk
from textblob import TextBlob

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentence_results = []

    print("\nSentence-Level Sentiment Analysis:")
    for sentence in blob.sentences:
        polarity = sentence.sentiment.polarity
        subjectivity = sentence.sentiment.subjectivity
        print(f"Sentence: {sentence}")
        print(f"Polarity: {polarity}, Subjectivity: {subjectivity}")
        sentence_results.append((sentence, polarity, subjectivity))
    
    overall_polarity = blob.sentiment.polarity
    overall_subjectivity = blob.sentiment.subjectivity
    print("\nOverall Sentiment Analysis:")
    print(f"Polarity: {overall_polarity}, Subjectivity: {overall_subjectivity}")

    return overall_polarity, overall_subjectivity, sentence_results


# Function to tokenize and perform POS tagging using NLTK
def nltk_analysis(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    print("\nNLTK Part-of-Speech Tagging:")
    for word, tag in tagged:
        print(f"{word}: {tag}")

    return tagged


if __name__ == "__main__":
    # Poem from the perspective of a TFT player
    tft_poem = """
    The queue pops, my heart beats fast,
    A new set dawns, the meta won't last.
    Hexcore glitches and Dragons soar,
    A battlefield awaits, strategy at its core.

    In BoxBox's Bootcamp, we refine our play,
    Late-night games stretch till the break of day.
    With tacticians cheering, the rounds unfold,
    Each win feels sweet, each loss makes us bold.

    Yet balance wavers, frustration grows,
    When RNG strikes or a synergy blows.
    Still, the joy of learning, the thrill of the fight,
    Makes TFT a game that feels just right.
    """

    # Perform NLTK tokenization and POS tagging
    print("NLTK Analysis of TFT Poem:\n")
    nltk_analysis(tft_poem)

    # Perform sentiment analysis with TextBlob
    print("\nTextBlob Sentiment Analysis of TFT Poem:\n")
    analyze_sentiment(tft_poem)
