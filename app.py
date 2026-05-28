import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Library System")

st.title("📚 Smart Library Recommendation System")
st.write("Proyek Matematika Terapan Menggunakan Python")

# ==================================
# DATA BUKU
# ==================================

books = [
    "Akuntansi Keuangan Lanjutan",
    "Akuntansi Manajemen",
    "Matematika Diferensial",
    "Kalkulus Integral",
    "Statistika dan Probabilitas",
    "Statistika Deskriptif",
    "Pengembangan Soft Skill",
    "Pendidikan Non Formal",
    "Folklor Batak",
    "Menulis Kritik dan Esai",
    "Bisnis Modern",
    "Kewirausahaan",
    "Pemodelan Matematika",
    "Pemodelan Matematika Terapan"
]

descriptions = [
    "akuntansi keuangan laporan perusahaan",
    "akuntansi manajemen biaya perusahaan",
    "matematika diferensial turunan fungsi",
    "kalkulus integral aplikasi matematika",
    "statistika probabilitas data penelitian",
    "statistika deskriptif penelitian data",
    "pengembangan soft skill pendidikan",
    "manajemen pendidikan non formal",
    "budaya folklor batak tradisional",
    "menulis kritik esai bahasa",
    "bisnis modern manajemen usaha",
    "wirausaha bisnis pengembangan usaha",
    "pemodelan matematika sistem",
    "pemodelan matematika aplikasi sistem"
]

# ==================================
# MENU
# ==================================

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Data Buku",
        "Himpunan",
        "Kombinatorika",
        "TF-IDF & Matriks",
        "Cosine Similarity",
        "SPL",
        "Determinan",
        "Boolean & Logika"
    ]
)

# ==================================
# DATA BUKU
# ==================================

if menu == "Data Buku":

    df = pd.DataFrame({
        "Judul Buku": books
    })

    st.dataframe(df)

# ==================================
# HIMPUNAN
# ==================================

elif menu == "Himpunan":

    st.header("Operasi Himpunan")

    A = set(descriptions[2].split())
    B = set(descriptions[3].split())

    st.write("A =", A)
    st.write("B =", B)

    st.write("Irisan (A ∩ B)")
    st.write(A.intersection(B))

    st.write("Gabungan (A ∪ B)")
    st.write(A.union(B))

# ==================================
# KOMBINATORIKA
# ==================================

elif menu == "Kombinatorika":

    st.header("Kombinatorika")

    n = len(books)

    pasangan = (n * (n - 1)) // 2

    st.latex(r"C(n,2)=\frac{n!}{2!(n-2)!}")

    st.success(
        f"Jumlah pasangan buku yang dapat dibandingkan = {pasangan}"
    )

# ==================================
# TF-IDF
# ==================================

elif menu == "TF-IDF & Matriks":

    st.header("TF-IDF Matrix")

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(descriptions)

    matrix = pd.DataFrame(
        tfidf.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    st.dataframe(matrix)

# ==================================
# COSINE SIMILARITY
# ==================================

elif menu == "Cosine Similarity":

    st.header("Cosine Similarity")

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(descriptions)

    similarity = cosine_similarity(tfidf)

    df = pd.DataFrame(
        similarity,
        index=books,
        columns=books
    )

    st.dataframe(df)

    st.subheader("Grafik Similarity Buku Pertama")

    plt.figure(figsize=(10,4))

    plt.bar(
        books,
        similarity[0]
    )

    plt.xticks(rotation=90)

    st.pyplot(plt)

# ==================================
# SPL
# ==================================

elif menu == "SPL":

    st.header("Sistem Persamaan Linear")

    A = np.array([
        [1,1,1],
        [2,1,-1],
        [1,-1,1]
    ])

    B = np.array([
        10,
        5,
        7
    ])

    hasil = np.linalg.solve(A,B)

    st.latex(r"x+y+z=10")
    st.latex(r"2x+y-z=5")
    st.latex(r"x-y+z=7")

    st.success(f"x = {hasil[0]:.2f}")
    st.success(f"y = {hasil[1]:.2f}")
    st.success(f"z = {hasil[2]:.2f}")

# ==================================
# DETERMINAN
# ==================================

elif menu == "Determinan":

    st.header("Determinan Matriks")

    A = np.array([
        [1,1,1],
        [2,1,-1],
        [1,-1,1]
    ])

    det = np.linalg.det(A)

    st.write(A)

    st.success(f"Determinan = {det:.2f}")

# ==================================
# BOOLEAN
# ==================================

elif menu == "Boolean & Logika":

    st.header("Boolean dan Logika")

    similarity = 0.8

    kategori_sama = True

    penulis_sama = False

    hasil = (
        (similarity > 0.7 and kategori_sama)
        or
        (similarity > 0.7 and penulis_sama)
    )

    st.write("Similarity =", similarity)
    st.write("Kategori Sama =", kategori_sama)
    st.write("Penulis Sama =", penulis_sama)

    st.success(f"Hasil Boolean = {hasil}")

    if hasil:
        st.success("Buku Direkomendasikan")
    else:
        st.error("Buku Tidak Direkomendasikan")