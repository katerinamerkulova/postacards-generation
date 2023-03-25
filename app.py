import random
import re

import streamlit as st

from keywords import Analyzer
from stable_diffusion_generator import Generator as sd_generator

analyzer = Analyzer()
sd = sd_generator()

st.title('Реконструкция открыток')
st.subheader('Период открыток')
start_year, end_year = st.slider('', 1891, 2014, (1941, 1965))

analyzer = Analyzer()

key_words = analyzer.pipeline(start_year, end_year)
st.subheader('Ключевые слова')
st.text(key_words)

key_words = st.text_input('Ключевые слова', key_words)
key_words = re.findall('\w+', key_words)

text = (
    f"Почтовая открытка "
    f"{start_year}-{end_year} годов: "
    f"{' '.join(key_words)}"
)
st.text(text)

if st.button('Генерировать открытки!'):
    images = sd.get_images(text)
    st.image(images, width=350)
