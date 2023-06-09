# Laporan Proyek Machine Learning - Jelvin Krisna Putra 

## Project Overview

Seperti yang kita ketahui dunia hiburan sekarang sudah menjadi pasar yang cukup menjanjikan ditambah kehadiran _streaming online_ membuat para pembuat video hiburan seperti sinetron hingga _anime_ memanfaatkan hal yang sama untuk meningkatkan pendapatan mereka, sudah menjadi rahasia umum dimana film bajakan itu merajalela di forum-forum internet, padahal kehadiran dari film bajakan ini merugikan produser hingga aktor dari film itu sendiri, berlaku juga pada *anime* .

Dikutip dari Suharno (dalam [[1]](https://www.google.com/url?sa=t&source=web&rct=j&url=https://akperyarsismd.e-journal.id/BNJ/article/download/48/38&ved=2ahUKEwin3LuR6bL_AhV73jgGHWluBlgQFnoECA0QAQ&usg=AOvVaw3dMYTSl3sAorJ9pLC5Dzkq)) Indonesia menjadi
salah satu negara dengan pengguna aktif smartphone terbesar keempat di dunia setelah negara Cina, India, dan
Amerika. Dimana ini memiliki 2 dampak yang satunya negatif dimana bisa meningkatkan kemungkinan *miopia* atau rabun jauh pada generasi muda, dan ada juga dampak positif dimana munculnya peluang bisnis yang dimanfaatkan oleh para pengusaha muda atau bahkan developer untuk membuat model bisnis yang berorientasi dalam penyaluran video digital atau film yang legal dan berlisensi seperti yang ada pada _bilibili_

"Globalisasi budaya populer Jepang kini dibuktikan dengan semakin banyaknya acara-acara yang bertemakan Jepang seperti karaoke, festival *manga*, kontes cosplay, dan yang tidak terlewatkan adalah *anime*." [[2]](https://journal.student.uny.ac.id/ojs/index.php/societas/article/viewFile/9099/8770&ved=2ahUKEwj4tabp4rL_AhUkT2wGHZVbD9cQFnoECAwQAQ&usg=AOvVaw0J8N1WUasONJecD-l8nj35). Internet 4.0 dan globalisasi memungkinkan budaya dari luar untuk bisa masuk ke dalam dengan mudahnya. 

Pentingnya sistem rekomendasi dalam suatu aplikasi streaming video adalah sangat penting karena *user* terkadang tidak terlalu lama menghabiskan waktu pada aplikasi yang membuat *user* tidak nyaman atau bahkan tidak relevan dengan dirinya yang berdampak pada menurunnya pendapatan suatu aplikasi karena sepinya pengunjung dari suatu aplikasi mau itu berbasis website maupun *android app*

Oleh karena itu dalam project kali ini saya sebagai _data science_ diminta oleh perusahaan fiktif _balibali_ untuk membuat _model machine learning_ yang mampu merekomendasikan penggunanya sejumlah anime yang sesuai dengan preferensi pengguna masing-masing, dengan tujuan untuk meningkatkan pendapatan perusahaan dan membuat perusahaan bisa bertahan untuk waktu yang cukup lama karena sengitnya persaingan dari perusahaan lain.

## Business Understanding

### Problem Statements
Adapun permasalahan yang dialami oleh perusahaan ini adalah sebagai berikut:
1. Bagaimana cara menentukan rekomendasi *anime* yang sesuai dengan preferensi konsumen?
2. Mengapa model rekomendasi anime ini perlu untuk dibuat?
3. Apa yang diharapkan dari model rekomendasi anime pada aplikasi milik perusahaan *balibali* ini jangka pendek maupun jangka panjangnya?

### Goals
Dari permasalahan di atas dapat diketahui tujuan dari proyek ini adalah:
1. Menentukan rekomendasi yang personal dan relevan dengan setiap pengguna aplikasi.
2. Meningkatkan pengalaman pengguna dalam penggunaan aplikasi milik perusahaan *balibali* di versi terbaru mendatang.
3. Meningkatkan pendapatan perusahaan melalui iklan yang disisipkan setiap kali user menonton anime serta mempertahankan perusahaan agar tetap bisa bersaing dan relevan dengan teknologi.

Adapun solusi yang diterapkan untuk melatih model adalah dengan menggunakan model pendekatan *Collaborative Filtering* yang merupakan algoritma untuk mengidentifikasi kesamaan antara pengguna berdasarkan preferensi rating mereka, dan memberikan rekomendasi anime berdasarkan rating pengguna lain yang memiliki preferensi serupa, serta mengukur tingkat akurasi model menggunakan metrik *Root Mean Square Error (RMSE)*.

## Data Understanding

Dataset "*Anime Recommendations Database*" terdiri dari beberapa file CSV, termasuk file "*rating.csv*" yang berisi informasi tentang peringkat anime yang diberikan oleh pengguna serta "*anime.csv*" yang berisi tentang informasi dari anime mulai dari nama hingga episode dan member. Dataset ini dapat diunduh dari tautan berikut: Kaggle - Anime Recommendations Database

Variabel-variabel pada dataset "*rating.csv*" antara lain:
1. user_id: ID pengguna
2. anime_id: ID anime
3. rating: Peringkat yang diberikan oleh pengguna untuk anime tersebut

Berikut adalah beberapa variabel yang terdapat dalam dataset "anime.csv":
1. anime_id: ID unik untuk setiap anime.
2. name: Judul anime.
3. genre: Genre-genre yang terkait dengan anime.
4. type: Tipe anime (TV, Movie, OVA, dll.).
5. episodes: Jumlah episode dalam anime.
6. rating: Peringkat rata-rata yang diberikan oleh pengguna.
7. members: Jumlah anggota yang menambahkan anime ke daftar mereka.

Untuk memahami data dengan lebih baik, beberapa tahapan yang dapat dilakukan antara lain adalah:
1. Melakukan analisis statistik deskriptif untuk memahami statistik dasar, seperti rata-rata, median, dan deviasi standar dari variabel-variabel yang relevan.
2. Mengidentifikasi dan menangani nilai-nilai yang hilang atau tidak valid dalam dataset.
3. Melakukan pemetaan atau pengkodean variabel kategorikal menjadi representasi numerik yang sesuai, jika diperlukan.

## Data Preparation

Beberapa teknik yang diterapkan untuk membersihkan dan mempersiapkan data *"Anime Recommendations Database*" adalah sebagai berikut:
1. Load Data:
   - Import dan cek versi dari library *tensorflow* serta menghubungkan google drive ke dalam website collab google untuk mendapatkan file *kaggle.json*
   - Mengunduh dan mengekstrak file zip yang didapatkan dari sumber dataset.
   - Membaca file *rating.csv* dengan _library pandas_ yang ditampilkan ke dalam *variabel df* dengan tipe data *dataframe*, lalu menampilkan data awal, akhir dan jumlah data yang tersedia 7813737 *rows* × 3 *columns*.
   - Membaca file *anime.csv* dengan library yang sama pada poin pertama dan menampungnya ke dalam variabel *anime_df* dan ditampilkan data awal, akhir dan jumlah data yang tersedia dalam 12294 *rows* × 7 *columns*.
2. Data Cleaning:
   - Melakukan pengecekan berapa nilai null dalam file *anime_csv* dengan memanggil fungsi ```isna().sum()``` yang hasilnya dapat dilihat di tabel 1

Tabel 1. Hasil dari pengecekan nilai NaN pada *anime_csv*
| Column    | Missing Values |
|-----------|----------------|
| anime_id  | 0              |
| name      | 0              |
| genre     | 62             |
| type      | 25             |
| episodes  | 0              |
| rating    | 230            |
| members   | 0              |
   - Dari tabel 1 bisa dilihat bahwa kolom *genre* memiliki _NaN value_ pada 62 record, kolom *type* memiliki _NaN value_ pada 25 record, kolom *rating* memiliki _NaN value_ pada 230 record, karena data null pada dataset kali ini tidak banyak maka semua data null record-nya bisa di hilangkan dengan memanggil fungsi ```anime_csv.dropna()``` dan lakukan pengecekan lagi untuk null value untuk memastikan bahwa semua null value benar sudah dihilangkan.
   - Melakukan pengurutan pada file "*anime.csv*" berdasarkan kolom "*anime_id*" dan menyimpannya ke dalam variabel *fix_anime* secara *ascending* atau kecil ke besar, dengan total 12017 baris dan 7 kolom yang dapat dilihat pada tabel 2.
 
 Tabel 2. Data Anime Setelah Diurutkan
 | anime_id | name                                   | genre                                              | type    | episodes | rating | members |
|----------|----------------------------------------|----------------------------------------------------|---------|----------|--------|---------|
| 22       | Cowboy Bebop                           | Action, Adventure, Comedy, Drama, Sci-Fi, Space    | TV      | 26       | 8.82   | 486824  |
| 152      | Cowboy Bebop: Tengoku no Tobira         | Action, Drama, Mystery, Sci-Fi, Space              | Movie   | 1        | 8.40   | 137636  |
| 214      | Trigun                                 | Action, Comedy, Sci-Fi                             | TV      | 26       | 8.32   | 283069  |
| 2095     | Witch Hunter Robin                     | Action, Drama, Magic, Mystery, Police, Supernatural| TV      | 26       | 7.36   | 64905   |
| 3159     | Beet the Vandel Buster                 | Adventure, Fantasy, Shounen, Supernatural          | TV      | 52       | 7.06   | 9848    |
| ...      | ...                                    | ...                                                | ...     | ...      | ...    | ...     |
| 9991     | Platonic Chain: Ansatsu Jikkouchuu      | Sci-Fi, Slice of Life                              | Special | 1        | 1.67   | 51      |
| 10444    | Sushi Azarashi                         | Comedy                                             | TV      | 30       | 3.00   | 12      |
| 9266     | Kochinpa! Dainiki                      | Comedy                                             | TV      | 24       | 3.40   | 75      |
| 2726     | Pokemon Generations                    | Action, Adventure, Fantasy, Game, Kids             | ONA     | 18       | 7.21   | 295     |
| 9586     | Mobile Suit Gakuen: G-Reco Koushien     | Comedy                                             | Special | 9        | 5.67   | 94      |

   - Menampilkan data unik pada kolom anime_id dan dibandingkan dengan jumlah data yang ternyata sama-sama 12017 sehingga data ini tidak ada perubahan serta menampilkan data unik pada kolom type dimana didapatkan bahwa ada 6 tipe anime yang tersedia pada dataset ini.
3. Proses encoding
   - Tujuan utama encoding data adalah menggantikan nilai ID pengguna asli dengan angka yang berurutan, sehingga memudahkan pengolahan data dan meminimalkan penggunaan memori. Dengan melakukan encoding, kita dapat mengonversi ID pengguna yang awalnya berupa string atau nilai kategorik menjadi angka-angka yang dapat digunakan oleh algoritma machine learning.
   - Dimulai dengan mengambil nilai unik dari kolom "user_id" dalam DataFrame dan mengkonversinya menjadi list dengan menggunakan fungsi ```unique()``` dan ```tolist()```. Hasilnya disimpan dalam variabel *user_ids*, yang memberikan daftar semua ID pengguna yang ada dalam dataset, lanjut dengan membuat pemetaan antara ID asli dan angka berurutan yang disimpan pada ```user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}``` dan dilakukan sebaliknya untuk variabel ```user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}``` angka berurutan dilakukan pemetaan dengan value dari id asli.
   - Dan lakukan encode untuk kolom anime_id dengan proses yang sama.
   - Buat kolom baru bernama user dan anime dengan isi dari hasil encoding user_id dan anime_id, yang dilanjutkan dengan melihat info kolom dan pemakaian memori dari data rating yang memiliki 5 kolom dan berukuran *298MB*.
   - Cek data *NaN* atau *null* pada data rating, dan didapatkan bahwa tidak ada data *NaN* di semua kolom sehingga data sudah bisa dikatakan bersih, dan dilanjutkan dengan menganalisa bentuk data seperti maksimal, minimal, standar deviasi, dan lainnya dengan memanggil fungsi ```describe()``` dengan hasil didapatkan bahwa kolom rating minimal *value* adalah -1
   - Setelah di cek ternyata ada 1.476.496 data yang memiliki nilai rating -1, karena rating seharusnya hanya boleh di rentang 0 hingga 10 dan data yang kita miliki sebanyak 7 juta lebih data, maka data ini kita hilangkan dan tersisa 6337241 data.
3. Manajemen Data
   - Mengambil 5000 data teratas dengan tetap memastikan rating ada di range 0 hingga 10 karena keterbatasan memori, *resource* yang kita miliki dan data ini sudah cukup untuk proses pembentukan model kita kali ini.
   - mencari nilai rating tertinggi, dan terendah yang diletakkan dalam variabel *max_rating* dan *min_rating* secara berurutan serta jumlah pengguna dan anime.
   - Mengubah tipe data pada kolom rating semula *int 64 bit* menjadi *float 32 bit* yang bertujuan untuk mengoptimalkan penggunaan memori. Tipe data *float32* membutuhkan lebih sedikit ruang penyimpanan dibandingkan dengan tipe data *int64*.
   - Mengacak urutan baris dalam DataFrame dengan kode ```df = df.sample(frac=1, random_state=42)```, Tujuan dari mengacak urutan baris adalah untuk memastikan bahwa data yang digunakan dalam model tidak memiliki pola urutan yang dapat mempengaruhi kinerja model, dimana Argumen ```frac=1``` mengindikasikan bahwa kita ingin mengambil seluruh baris dalam data, sedangkan random_state=42 digunakan untuk menjaga keacakan hasil data dimana dengan menggunakan nilai random_state yang sama, kita dapat memastikan bahwa setiap kali kode dijalankan, hasil acakan akan sama.
   - Kemudian bagi dataset menjadi skala 90:10 dimana totalnya sejumlah 4500 data latih dan 500 data test.

## Modeling

Pada tahap ini, kita akan menggunakan teknik *collaborative filtering*, dimana *Collaborative Filtering (CF)* adalah salah satu teknik dalam sistem rekomendasi yang mengandalkan kolaborasi antara pengguna dan item untuk memberikan rekomendasi yang personal dan relevan. Metode ini didasarkan pada asumsi bahwa jika dua pengguna memiliki preferensi yang serupa dalam masa lalu, maka kemungkinan besar mereka akan memiliki preferensi yang serupa di masa depan, metode ini digunakan karena selaras dengan tujuan dari pembuatan model dimana model diharapkan bisa memberikan rekomendasi yang personal dan relevan dengan pengguna sehingga teknik *collaborative filtering* menjadi teknik yang digunakan dalam pembuatan model kali ini

Adapun kelebihan teknik ini adalah mampu memberikan rekomendasi yang personal dan relevan berdasarkan preferensi sebelumnya, mampu menemukan hubungan kompleks antara pengguna dan item yang sulit diidentifikasi dengan pendekatan lain. Untuk kekurangan dari teknik ini adalah memerlukan data yang cukup besar dan informasi riwayat preferensi yang lengkap, rentan terhadap masalah cold-start, yaitu masalah saat sistem menghadapi pengguna baru yang belum ada riwayat preferensinya, dan tidak mampu memberikan rekomendasi untuk item yang belum dikenal oleh pengguna.

Untuk Mengimplementasikannya kita mulai dengan membuat class *RecommenderNet* dengan *keras Model class*. Kode *class RecommenderNet* ini terinspirasi dari tutorial dalam situs [Keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) dengan beberapa adaptasi sesuai kasus yang sedang kita selesaikan yaitu rekomendasi untuk anime. Kemudian dilanjutkan dengan melakukan compile terhadap model yang sudah kita buat dengan parameter loss adalah ```Binary Cross entropy``` optimizer-nya menggunakan ```Adam(learning_rate=0.001)``` metrik-nya adalah Root Mean Square Error yang diimport dari tensorflow dan keras.

Sebelum dilakukan proses pelatihan model kita akan mengimplementasi callback ke dalam model sehingga jika model sudah berada di tahap error yang kita inginkan proses pelatihan bisa dihentikan agar tidak membuang banyak waktu dan resource pada sistem kita, dan pada kasus ini pelatihan akan dihentikan jika nilai error pada root mean square error dibawah 16%.

Kemudian setelah itu kita bisa melatih model dengan memanggil fungsi ```fit()``` dengan parameter x adalah *x_train*, y adalah *y_train*, batch_size diisi 8, epochs atau perulangan diisi sebanyak 100 kali, validation_data diisi dengan (x_val, y_val), dan yang tak kalah penting callbacks diisi dengan callbacks dari yang telah dibuat sebelumnya.

Untuk model terbaik akan dilihat dari model yang memiliki *RMSE* terkecil.

## Evaluation

Metrik evaluasi yang akan digunakan untuk menganalisis kinerja model dalam memprediksi rekomendasi anime untuk pengguna adalah Root Mean Squared Error (RMSE). 


RMSE = $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y - \hat{y})^2}$

Dari rumus di atas, *Root Mean Square Error* dihitung dari akar kuadrat dari rata-rata selisih kuadrat antara nilai prediksi dan nilai sebenarnya. RMSE memberikan gambaran tentang sejauh mana prediksi model kita menyimpang dari nilai sebenarnya. Semakin kecil nilai RMSE, semakin baik performa model dalam melakukan prediksi yang akurat.

![Metrik Model](https://raw.githubusercontent.com/krisna31/anime-recommendation-collaborative-filtering/main/images/metrik-akhir.png)
Gambar 2. Visualisasi Metrik

Dapat dilihat dari gambar 2 nilai error RMSE turun hingga sesuai dengan target awal kita dimana nilai error dibawah 16% sehingga pada *epoch* ke 6 pelatihan model sudah bisa terhenti dengan *callback* yang terpanggil.

Kemudian kita coba menggunakan model yang dimulai dengan mengambil sampel user secara acak dan definisikan variabel *anime_not_watched* yang berisikan anime belum ditonton oleh pengguna diisi dengan logika operasi bitwise, selanjutnya untuk memperoleh rekomendasi restoran, gunakan fungsi ```model.predict()```, kemudian kita tampilkan hasilnya mulai dari anime dengan rating tinggi dari pengguna, dan rekomendasi anime dengan output bisa dilihat pada bagan dibawah.

```
Anime with high ratings from user
--------------------------------
Kara no Kyoukai 5: Mujun Rasen : Action, Drama, Mystery, Romance, Supernatural, Thriller
Ookami to Koushinryou II : Adventure, Fantasy, Historical, Romance
Fate/Zero 2nd Season : Action, Fantasy, Supernatural, Thriller
Magi: The Kingdom of Magic : Action, Adventure, Fantasy, Magic, Shounen
Shigatsu wa Kimi no Uso : Drama, Music, Romance, School, Shounen
--------------------------------
Top 10 anime recommendation
--------------------------------
Cowboy Bebop : Action, Adventure, Comedy, Drama, Sci-Fi, Space
Sen to Chihiro no Kamikakushi : Adventure, Drama, Supernatural
Howl no Ugoku Shiro : Adventure, Drama, Fantasy, Romance
Katekyo Hitman Reborn! : Action, Comedy, Shounen, Super Power
Toki wo Kakeru Shoujo : Adventure, Drama, Romance, Sci-Fi
Durarara!! : Action, Mystery, Supernatural
Kuroko no Basket : Comedy, School, Shounen, Sports
Danshi Koukousei no Nichijou : Comedy, School, Shounen, Slice of Life
Sukitte Ii na yo. : Romance, School, Shoujo
Haikyuu!! Second Season : Comedy, Drama, School, Shounen, Sports
```

Bisa dilihat dari hasil rekomendasi hasil rekomendasi memiliki genre yang serupa misalnya *adventure* untuk anime *Ookami to Koushinryou II*, direkomendasikan anime dengan genre yang serupa seperti *Cowboy Bebop, Toki wo Kakeru Shoujo*. Dari hasil ini dapat disimpulkan bahwasanya model yang kita buat dikatakan berhasil dan dapat diimplementasikan ke aplikasi *balibali*.

# Daftar Referensi
[1]	Dinda Puput Oktafia , Noor Yunida Triana , Roro Lintang Suryani, “Durasi Penggunaan Gadget Terhadap Personal Sosial Pada Anak Usia Prasekolah: Literatur Review,” Jurnal Borneo Nursing, vol. 4, no. 1, 2021.
[2]	Prista Ardi Nugroho and Grendi Hendrastomo,“Anime Sebagai Budaya Populer (Studi Pada Komunitas Anime di Yogyakarta)," Jurnal Pendidikan Sosiologi, 2016.
