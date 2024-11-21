This project performs sentiment analysis and part-of-speech (POS) tagging on text related to Teamfight Tactics (TFT). The script features dynamic text input, token categorization, keyword detection, and sentiment classification.

Example Output:

Categorized Tokens:
Nouns: ['Teamfight', 'Tactics', 'TFT', 'game', 'Riot', 'Games', 'League', 'Legends', ...]
Verbs: ['is', 'challenges', 'build', 'position', 'introduces', ...]
Adjectives: ['strategic', 'unique', 'recent', 'new', ...]
Others: ['(', ')', '.', ',', ...]

Keyword Analysis:
Keyword 'Dragons' detected in text!
Keyword 'TFT' detected in text!
Keyword 'Innovators' detected in text!

Sentiment Analysis Results:
Polarity: 0.3125 (Positive)
Subjectivity: 0.5625

Reflection:
This analysis highlights keywords related to TFT and categorizes tokens by their grammatical roles.
Sentiment analysis also classifies the overall tone of the text as Positive, Neutral, or Negative.
Using TextBlob simplifies sentiment analysis, while NLTK provides detailed token and POS information.
