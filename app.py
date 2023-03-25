import random
import re

import streamlit as st

from keywords import get_keywords
from stable_diffusion_generator import Generator as sd_generator

st.title('Реконструкция открыток')
st.subheader('Период открыток')
years = st.slider('', 1891, 2014, (1941, 1965))

key_words = get_keywords(years)
st.subheader('Ключевые слова')
st.text(f'{" ".join(key_words)}')

key_words = st.text_input('Ключевые слова', f'{", ".join(key_words)}')
key_words = re.findall('\w+', key_words)

sd = sd_generator()

text = (
    f"Почтовая открытка "
    # f"{random.randint(*years)} год: "
    f"{'-'.join((str(x) for x in years))} годов: "
    f"{', '.join(key_words)}"
)
st.text(text)

images = sd.get_images(text)
st.image(images, width=350)
