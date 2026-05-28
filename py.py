import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rekomendasi Buku", page_icon="📚")
st.title("📚 Sistem Rekomendasi Buku")

# Cell 1: Load data
try:
    df = pd.read_csv("books.csv")
except FileNotFoundError:
    st.error("File 'books.csv' tidak ditemukan. Buat file CSV dengan kolom 'judul' dan 'deskripsi'")
    st.stop()

# Cell 2: Validasi kolom
if 'judul' not in df.columns or 'deskripsi' not in df.columns:
    st.error("File CSV harus punya kolom 'judul' dan 'deskripsi'")
    st.stop()

# Cell 3: Bersihkan data
df['judul'] = df['judul'].astype(str).str.strip()
df['deskripsi'] = df['deskripsi'].astype(str).fillna('')

# Hapus baris yang deskripsinya kosong
df = df[df['deskripsi'].str.strip()!= '']
if df.empty:
    st.error("Data kosong setelah dibersihkan")
    st.stop()

# Cell 4: TF-IDF
vectorizer = TfidfVectorizer(stop_words=['dan', 'atau', 'yang', 'di', 'ke', 'dari'], lowercase=True)
tfidf_matrix = vectorizer.fit_transform(df['deskripsi'])

# Cell 5: UI Input
book_list = df['judul'].tolist()
book_input = st.selectbox("Pilih Judul Buku:", [""] + book_list)

if st.button("Cari Buku Mirip"):
    if book_input == "":
        st.warning("Pilih judul buku dulu!")
        st.stop()

    # Cell 6: Cari index yang benar
    mask = df['judul'].str.lower() == book_input.lower().strip()
    if not mask.any():
        st.error("Judul buku tidak ditemukan")
        st.stop()

    idx = mask.idxmax() # ini yang benar, bukan df.index[0]

    # Cell 7: Hitung similarity dan tampilkan hasil
    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    scores = list(enumerate(similarity))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Ambil top 5 selain buku itu sendiri
    top_books = [(i, s) for i, s in scores if i!= idx][:5]

    if not top_books:
        st.info("Tidak ada buku mirip yang ditemukan")
        st.stop()

    st.subheader("Hasil Rekomendasi")
    titles = []
    values = []

    for i, score in top_books:
        title = df.iloc[i]['judul']
        st.write(f"**{title}** → `{score:.4f}`")
        titles.append(title)
        values.append(score)

    # Grafik
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(titles, values, color='#4CAF50')
    plt.xticks(rotation=25, ha='right')
    plt.ylabel("Skor Kemiripan")
    plt.ylim(0, 1)
    plt.tight_layout()
    st.pyplot(fig)