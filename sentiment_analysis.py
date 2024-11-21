import nltk
from textblob import TextBlob

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

keywords = ["Dragons", "Innovators", "TFT", "Mechanics", "Synergies", "Sets"]

def classify_sentiment(polarity):
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

file_name = input("Enter the filename of the text to analyze (or press Enter for default): ")
if file_name.strip():
    with open(file_name, "r") as file:
        text = file.read()
else:
    text = """
    Teamfight Tactics (TFT) is a strategic auto-battler by Riot Games set in the League of Legends universe-- and also my favorite game at the moment. 
    It challenges players to build, position, and strategize with unique champions and traits in turn-based rounds. 
    Each set introduces new gameplay elements, traits, and unique mechanics. Here’s a detailed breakdown of all the sets I have played:

    Set 6: Gizmos & Gadgets
    Gizmos & Gadgets was themed around innovation and invention, introducing champions with tech-powered abilities.
    Notable Traits: Innovator, Scrap, and Chemtech.
    Innovator units summoned a mechanical companion, which grew stronger with more Innovators on the team.
    Chemtech champions regenerated health and gained attack speed based on missing health, making them effective for longer fights.

    Set 7: Dragonlands
    Dragonlands took a high-fantasy approach, adding mythical dragons and beasts.
    Notable Traits: Dragon, Jade, and Whispers.
    Dragon units were powerful but cost more and could only be played one at a time, offering unique advantages.
    Jade champions summoned statues that provided healing and attack speed, making them sustainable and effective in longer fights.

    Set 8: Monsters Attack!
    Monsters Attack! introduced a superhero vs. supervillain theme, adding urban chaos to the battlefield.
    Notable Traits: Anima Squad, Underground, and Aegis.
    Anima Squad champions gained fame for each kill, resulting in permanent health bonuses, while Underground champions could complete heists for escalating rewards. 

    Each set has brought fresh challenges and engaging synergies, from Innovator's mechanical companions to Dragonlands' mythical units and Monsters Attack!'s city heist rewards. TFT remains dynamic and complex, offering endless strategic depth for players.
    """

tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)

nouns, verbs, adjectives, others = [], [], [], []
for word, tag in tagged:
    if tag.startswith('NN'):
        nouns.append(word)
    elif tag.startswith('VB'):
        verbs.append(word)
    elif tag.startswith('JJ'):
        adjectives.append(word)
    else:
        others.append(word)

print("\nCategorized Tokens:")
print(f"Nouns: {nouns}")
print(f"Verbs: {verbs}")
print(f"Adjectives: {adjectives}")
print(f"Others: {others}")

print("\nKeyword Analysis:")
for keyword in keywords:
    if keyword in text:
        print(f"Keyword '{keyword}' detected in text!")

blob = TextBlob(text)
sentiment = blob.sentiment
sentiment_class = classify_sentiment(sentiment.polarity)

print("\nSentiment Analysis Results:")
print(f"Polarity: {sentiment.polarity} ({sentiment_class})")
print(f"Subjectivity: {sentiment.subjectivity}")

print("\nReflection:")
print("This analysis highlights keywords related to TFT and categorizes tokens by their grammatical roles.")
print("Sentiment analysis also classifies the overall tone of the text as Positive, Neutral, or Negative.")
print("Using TextBlob simplifies sentiment analysis, while NLTK provides detailed token and POS information.")
