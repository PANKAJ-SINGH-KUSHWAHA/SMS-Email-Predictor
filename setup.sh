mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
!python -c "import nltk; nltk.download('punkt')"
!python -c "import nltk; nltk.download('stopwords')"
!python -c "import nltk; nltk.download('wordnet')"
!python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
!python -c "import nltk; nltk.download('maxent_ne_chunker')"
!python -c "import nltk; nltk.download('words')"
