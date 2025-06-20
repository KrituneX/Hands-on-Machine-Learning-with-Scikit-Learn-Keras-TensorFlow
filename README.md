# 📘 Ringkasan Hands-On Machine Learning
*Oleh Aurélien Géron – Edisi ke-2*

Dokumen ini berisi ringkasan tiap bab dari buku *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 
Setiap bab merangkum konsep penting, formula, serta hands-on code yang relevan untuk digunakan di Google Colab.

Dibagi menjadi dua bagian besar:

## ⚙️ Bagian I – Dasar-Dasar Pembelajaran Mesin

**Chapter 1 – The Machine Learning Landscape**  
- Menjelaskan berbagai jenis model pembelajaran mesin: supervised, unsupervised, reinforcement.
- Pengenalan regresi, klasifikasi, clustering, reduksi dimensi.
- Contoh aplikasi di dunia nyata seperti filter spam, prediksi harga, sistem rekomendasi.

**Chapter 2 – End-to-End Machine Learning Project**  
- Studi kasus lengkap proyek ML untuk prediksi harga rumah.
- Proses meliputi: pengumpulan data, eksplorasi, pembersihan, pemisahan train/test, pemodelan, evaluasi, dan deployment.

**Chapter 3 – Classification**  
- Fokus pada klasifikasi menggunakan dataset MNIST.
- Evaluasi model klasifikasi: confusion matrix, precision, recall, F1 score.
- Strategi multilabel dan multiclass classification.

**Chapter 4 – Training Models**  
- Penjelasan training linear model menggunakan cost function dan gradient descent.
- Regularisasi model untuk menghindari overfitting (Ridge, Lasso).

**Chapter 5 – Support Vector Machines**  
- SVM untuk klasifikasi dan regresi.
- Penjelasan margin maksimum, kernel trick, dan hyperparameter tuning.

**Chapter 6 – Decision Trees**  
- Bagaimana decision tree bekerja dalam mempartisi data.
- Pruning dan kontrol kompleksitas untuk menghindari overfitting.

**Chapter 7 – Ensemble Learning and Random Forests**  
- Kombinasi banyak model (bagging, boosting) untuk meningkatkan performa.
- Random Forests, AdaBoost, Gradient Boosting.

**Chapter 8 – Dimensionality Reduction**  
- Teknik untuk mengurangi jumlah fitur: PCA, t-SNE.
- Trade-off antara kompleksitas dan interpretabilitas.

**Chapter 9 – Unsupervised Learning Techniques**  
- Clustering (K-Means, DBSCAN), anomaly detection.
- Reduksi dimensi untuk visualisasi.

## 🤖 Bagian II – Pembelajaran Mendalam (Deep Learning)

**Chapter 10 – Introduction to Artificial Neural Networks**  
- Struktur dasar neural network: neuron, layer, fungsi aktivasi.
- Konsep feedforward, backpropagation, dan gradient descent.

**Chapter 11 – Training Deep Neural Nets**  
- Masalah vanishing/exploding gradients.
- Teknik peningkatan training: batch normalization, dropout, optimizers seperti Adam.
- Early stopping dan regularisasi.

**Chapter 12 – Custom Models and Training with TensorFlow**  
- Pembuatan model kustom dengan tf.keras.Model dan tf.keras.layers.Layer.
- Penggunaan tf.GradientTape untuk training manual.
- Fungsi loss dan aktivasi kustom.

**Chapter 13 – Loading and Preprocessing Data with TensorFlow**  
- Membaca dan mengelola dataset besar menggunakan tf.data.
- Pipeline preprocessing dengan batching, caching, dan prefetching.

**Chapter 14 – Deep Computer Vision Using CNNs**  
- Convolutional Neural Networks untuk pengenalan gambar.
- Layer konvolusi, pooling, padding, dan arsitektur populer (LeNet, VGG, ResNet).

**Chapter 15 – Processing Sequences Using RNNs and CNNs**  
- Recurrent Neural Networks untuk data sekuensial.
- Penjelasan tentang vanishing gradient, LSTM dan GRU.

**Chapter 16 – Natural Language Processing with RNNs and Attention**  
- Tokenisasi teks, word embeddings, sequence-to-sequence model.
- Attention mechanism dan Transformer.

**Chapter 17 – Representation and Generative Learning**  
- Autoencoder untuk reduksi dimensi dan deteksi anomali.
- Generative Adversarial Networks (GANs) untuk membuat data sintetis.

**Chapter 18 – Reinforcement Learning**  
- Agent-Environment loop, reward, policy, value function.
- Q-learning, DQN, dan strategi eksplorasi vs eksploitasi.

**Chapter 19 – Deployment with TensorFlow Serving**  
- Deployment model ke lingkungan produksi dengan TensorFlow Serving.
- Penyimpanan model (SavedModel), REST API, dan format request.

**Chapter 20 – AutoML**  
- Pengenalan AutoML dan Google’s AutoKeras.
- Automatisasi preprocessing, pemilihan model, tuning hyperparameter.

---
