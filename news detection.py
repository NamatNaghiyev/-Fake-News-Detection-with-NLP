import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# NLTK resurslarını yükləyirik
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Lemmatizer obyektini yaradırıq
lemmatizer = WordNetLemmatizer()

# Stopwords siyahısını yükləyirik
stop_words = set(stopwords.words('english'))

# Mətnin təmizlənməsi funksiyası
def clean_text(text):
    # Kiçik hərflərə çeviririk
    text = text.lower()

    # Xüsusi simvolları, rəqəmləri və s. silirik
    text = re.sub(r'\W', ' ', text)

    # Tək sözlərə ayırırıq (tokenization)
    words = text.split()

    # Stopwords-ləri aradan qaldırırıq və lemmatizasiya edirik
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Təmizlənmiş mətni birləşdiririk
    return ' '.join(words)

# Başlıq
st.title("📰 Fake News Classifier")

st.markdown("""
    Bu tətbiq **"Fake News"** və **"Real News"** mətni təyin edir.  
    Aşağıya bir xəbər mətnini yaz və görək sənə nə cavab verir!
""")

# Fayl yollarını istifadiçidən almaq
true_path = st.text_input("True News CSV faylının yolu:", r'True.csv.zip')
fake_path = st.text_input("Fake News CSV faylının yolu:", r'Fake.csv.zip')

# Fayllar varsa işləyir
if os.path.exists(true_path) and os.path.exists(fake_path):
    # Verilənləri yükləyirik
    true_news = pd.read_csv(true_path)
    fake_news = pd.read_csv(fake_path)

    true_news['label'] = 'real'
    fake_news['label'] = 'fake'

    df = pd.concat([true_news, fake_news], ignore_index=True)

    # Mətnləri təmizləyirik
    df['clean_text'] = df['text'].apply(clean_text)

    # Tokenizer yaradılır
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['clean_text'])

    X = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(X, maxlen=500)

    y = df['label'].apply(lambda x: 1 if x == 'fake' else 0).values

    # Məlumatı təlim və test dəstələrinə bölürük
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model yaradılır
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    with st.spinner('Model təlim olunur... biraz səbirli ol... 🔥'):
        model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    
    st.success('Model hazırdı! 🎯')

    # İstifadəçidən mətni almaq
    user_input = st.text_area("Xəbər mətnini daxil edin:")

    if st.button("Təsnif et"):
        if user_input:
            # Mətni təmizləyirik və modelin təxminini alırıq
            cleaned_text = clean_text(user_input)
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=500)
            prediction = model.predict(padded_sequence)

            # Nəticəni göstəririk
            if prediction > 0.5:
                st.error("Xəbər: **FAKE** ❌")
            else:
                st.success("Xəbər: **REAL** ✅")
        else:
            st.warning("Xahiş edirik, mətn daxil edin.")
else:
    st.warning("CSV fayllarını doğru göstər. Yol səhvdir, düz elə!")

