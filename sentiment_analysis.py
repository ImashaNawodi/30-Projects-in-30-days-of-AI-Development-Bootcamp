# Import necessary libraries
import nltk
from nltk.corpus import movie_reviews   # Built-in dataset of movie reviews (positive/negative)
from nltk.classify import NaiveBayesClassifier  # Naive Bayes model from NLTK
from nltk.classify.util import accuracy as nltk_accuracy  # Utility to calculate accuracy
from nltk.corpus import stopwords   # Stopwords list (common words like 'the', 'is', etc.)
import random   # To shuffle the dataset

# Download the NLTK data files (only need to run once)
nltk.download('movie_reviews')  # Movie reviews dataset
nltk.download('punkt')          # Tokenizer for splitting text into words
nltk.download('stopwords')      # Common stopwords list

# ---------------- Feature Extraction ---------------- #
# Function to convert a list of words into a dictionary format for classifier
# Example: ["good", "movie"] -> {"good": True, "movie": True}
def extract_features(words):
    return {word: True for word in words}

# ---------------- Load Dataset ---------------- #
# Each review in the dataset is stored as (list of words, category)
# category = 'pos' (positive) or 'neg' (negative)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle dataset to mix positive and negative reviews randomly
random.shuffle(documents)

# ---------------- Prepare Data ---------------- #
# Convert each review into feature dictionary using extract_features()
# Example: ({"good": True, "movie": True}, "pos")
featuresets = [(extract_features(d), c) for (d, c) in documents]

# Split dataset into training and testing
# First 1600 for training, remaining ~400 for testing
train_set, test_set = featuresets[:1600], featuresets[1600:]

# ---------------- Train Classifier ---------------- #
# Train Naive Bayes classifier on training set
classifier = NaiveBayesClassifier.train(train_set)

# ---------------- Evaluate Classifier ---------------- #
# Measure accuracy on test set
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show top 10 words that best indicate sentiment (positive or negative)
classifier.show_most_informative_features(10)

# ---------------- Sentiment Analysis Function ---------------- #
def analyze_sentiment(text):
    # Tokenize input text into words
    words = nltk.word_tokenize(text)
    
    # Remove stopwords (common words that don’t add meaning, e.g., "the", "is")
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    
    # Extract features (dictionary of words)
    features = extract_features(words)
    
    # Classify as 'pos' (positive) or 'neg' (negative)
    return classifier.classify(features)

# ---------------- Test with New Sentences ---------------- #
test_sentences = [
    "This movie is absolutely fantastic! The acting, the story, everything was amazing!",
    "I hated this movie. It was a waste of time and money.",
    "The plot was a bit dull, but the performances were great.",
    "I have mixed feelings about this film. It was okay, not great but not terrible either."
]

# Print predictions for custom sentences
for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {analyze_sentiment(sentence)}")
    print()
