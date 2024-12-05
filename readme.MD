# TFT Sentiment Analysis Poem

This project analyzes a poem written from the perspective of a *Teamfight Tactics (TFT)* player. The poem reflects the excitement, challenges, and camaraderie of playing a new set and participating in community events like *BoxBox's Bootcamp*. Due to the improvements to TFT as a game each set, the playerbase has immense amounts of passion and like to participate in competitions to support their favourite creators. The script evaluates the sentiment at both the sentence level and for the entire poem.

## Features

1. A poem capturing the emotions of a TFT player during a new set launch-- especially as someone whose favourite game is TFT and has been playing since set 4.
2. Analyzes polarity and subjectivity of each stanza.
3. Evaluates the sentiment of the poem as a whole.

## Example Output from Script

NLTK Part-of-Speech Tagging:
The: DT
queue: NN
pops: VBZ
,: ,
my: PRP$
heart: NN
beats: VBZ
fast: RB
,: ,
...

Sentence-Level Sentiment Analysis:
Sentence: The queue pops, my heart beats fast,
Polarity: 0.1, Subjectivity: 0.4
Sentence: A new set dawns, the meta won't last.
Polarity: 0.0, Subjectivity: 0.2
...

Overall Sentiment Analysis:
Polarity: 0.2, Subjectivity: 0.5


