#IMPORT LIBRARY YANG DIBUTUHKAN
import streamlit as st
import numpy as np
from PIL import Image
# LIBRARY ATAU ENGINE UNTUK MENJALANKAN MODEL
from ai_edge_litert.interpreter import Interpreter
import gdown
import os

#DIGUNAKAN UNTUK JUDUL H1
st.title("FLOWER IMAGE CLASSIFICATION")

#DIGUNAKAN UNTUK MEMBUAT GARIS PEMISAH
st.divider()

#DIGUNAKAN UNTUK SUB JUDUL H2
st.header("Flower Image classification with 102 classes")
#MEMBUAT TOMBOL UPLUAD FILE
uploaded_image = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])

#ARRAY LIST NAMA BUNGA
#!!! URUTANNYA HARUS SESUAI DENGAN YANG DI TRAINING(COLABS)
class_names = [
    'alpine sea holly', 'anthurium', 'artichoke', 'azalea', 'ball moss',
    'balloon flower', 'barbeton daisy', 'bearded iris', 'bee balm',
    'bird of paradise', 'bishop of llandaff', 'black-eyed susan',
    'blackberry lily', 'blanket flower', 'bolero deep blue', 'bougainvillea',
    'bromelia', 'buttercup', 'californian poppy', 'camellia', 'canna lily',
    'canterbury bells', 'cape flower', 'carnation', 'cautleya spicata',
    'clematis', "colt's foot", 'columbine', 'common dandelion', 'corn poppy',
    'cyclamen', 'daffodil', 'desert-rose', 'english marigold', 'fire lily',
    'foxglove', 'frangipani', 'fritillary', 'garden phlox', 'gaura', 'gazania',
    'geranium', 'giant white arum lily', 'globe thistle', 'globe-flower',
    'grape hyacinth', 'great masterwort', 'hard-leaved pocket orchid',
    'hibiscus', 'hippeastrum', 'japanese anemone', 'king protea', 'lenten rose',
    'lotus', 'love in the mist', 'magnolia', 'mallow', 'marigold', 'mexican aster',
    'mexican petunia', 'monkshood', 'moon orchid', 'morning glory',
    'orange dahlia', 'osteospermum', 'oxeye daisy', 'passion flower',
    'pelargonium', 'peruvian lily', 'petunia', 'pincushion flower',
    'pink primrose', 'pink-yellow dahlia?', 'poinsettia', 'primula',
    'prince of wales feathers', 'purple coneflower', 'red ginger', 'rose',
    'ruby-lipped cattleya', 'siam tulip', 'silverbush', 'snapdragon',
    'spear thistle', 'spring crocus', 'stemless gentian', 'sunflower',
    'sweet pea', 'sweet william', 'sword lily', 'thorn apple', 'tiger lily',
    'toad lily', 'tree mallow', 'tree poppy', 'trumpet creeper', 'wallflower',
    'water lily', 'watercress', 'wild pansy', 'windflower', 'yellow iris'
]

# MEMBUAT SIDEBAR LIST FLOWERS YANG DIDUGUNG
st.sidebar.title("LIST FLOWERS")

#MEMBUAT LIST NAMA KELAS
st.sidebar.write(
    f"Model supports **{len(class_names)} types of flowers**:"
)

for flower in class_names:
    st.sidebar.write(f"• {flower.title()}")
# MEMBUAT FUNGSI LOAD MODEL
# @st.cache_resource DIGUNAKAN UNTUK MENYIMPAN HASIL FUNGSI (INTERPRETER) DI MEMORY CACHE

MODEL_PATH = "model_efficientnet_bunga.tflite"

FILE_ID = "1ueifMf0aGzXhcKgngn7wYK8rH8ZtxuTX"

MODEL_URL = (
    f"https://drive.google.com/uc?id={FILE_ID}"
)

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        with st.spinner(
            "Downloading model, please wait..."
        ):
            gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False
            )

    interpreter = Interpreter(
        model_path=MODEL_PATH
    )

    return interpreter

# MEMANGGIL FUNGSI LOAD MODEL
interpreter = load_model()
#MENYIAPKAN MEMORY INPUT/OUTPT TENSOR MODEL
interpreter.allocate_tensors()

# MENGAMBIL INFO SPESIFIKASI INPUT (SHAPE,DTYPE, INDEX)
input_details = interpreter.get_input_details()
# MENGAMBIL INFO SPESIFIKASI OUTPUT MODEL
output_details = interpreter.get_output_details()

#MENYIMPAN BENTUK INPUT YANG DIHARAPKAN MODEL
input_shape = input_details[0]['shape']

# PERCABANGAN KONDISI,
# MENYATAKAN HARUS UPLUAD IMAGE(IS NOT NONE)
if uploaded_image is not None:
    #PREPROCESSING GAMBAR
    #MEMBUKA FILE GAMBAR DAN UPLUAD SERTA MEMASTIKAN FORMATNYA RGB UNTUK MENGHINDARI EROR JIKA GAMBAR FORMAT LAIN
    image = Image.open(
        uploaded_image).convert("RGB")
    
    #MENGUBAH UKURAN GAMBAR MENJADI 224x224 PIKSEL
    #MENGGUNAKAN ALGORITMA LANCZOZ UNTUK KUALITAS REZIZE TERBAIK
    resized_img = image.resize(
        (224,224),
        Image.Resampling.LANCZOS
        )
    
    # MENGKONVERSI GAMBAR DARI FORMAT PIL KE DALAM BENTUK NUMPY ARRAY
    # BERTIPE FLOAT SHAPE 3 DIMENSI (224 x 224 x 3)
    img_array = np.array(
        resized_img,
        dtype=np.float32
    )
    
    # MENAMBAHKAN DIMENSI BATCH DIDEPAN
    # SHAPE MENJADI (1, 224, 224, 3)
    input_data = np.expand_dims(
        img_array,
        axis=0
    )
    

    # PROSES PREDIKSI 
    # MEMASUKKAN GAMBAR KE MODEL
    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )
    
    # MENJALANKAN INFFERENSI - MODEL MEMPROSES GAMBAR DAN MENGHASILKAN PRDIKSI
    interpreter.invoke()
    
    # MENGAMBIL HASIL OUTPUT BERUPA ARRAY PROBABILITAS SETIAP 102 KELAS
    output_data = interpreter.get_tensor(
        output_details[0]['index'])
    

    # MENAMPILKAN HASIL DENGAN MENCARI NILAI INDEX TERTINGGI
    pred_class = class_names[output_data.argmax()]
    

    # MEMBUAT GARIS PEMBATAS
    st.divider()
    # MEMBUAT H2 
    st.header("RESULT PREDICTION")
    # MEMBUAT JUDUL YANG MERUPAKAN HASIL PREDIKSI
    st.subheader(pred_class)
    # MENAMPILKAN GAMBAR ASLI
    st.image(
        image,
        use_container_width=True
        )
