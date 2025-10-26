# 🧠 Face Parsing & Composition System (Python + Keras)

> 🕒 **Note:** This project was originally developed in **November 2023** and uploaded to GitHub in **October 2025** for portfolio and documentation purposes.

This Python-based deep learning project implements a **face parsing pipeline** using **Keras** and **TensorFlow**, trained on the **CelebAMask-HQ (CelebAMask-1M)** dataset. It leverages **U-Net** and **Attention U-Net** architectures to segment facial components, with enhanced training speed through **autoencoder-based weight initialization**.

The training process was accelerated by initializing the **U-Net encoder weights** using a pretrained **autoencoder model**, which had been trained separately for **image compression and reconstruction**. This strategy significantly reduced convergence time and improved segmentation stability across diverse facial angles.

After segmentation, the system performs **post-processing** to refine the extracted facial components. Using **KMeans clustering**, components are categorized based on their **size and spatial footprint**, allowing for structured organization and filtering. Components that are statistically unbalanced or poorly segmented are automatically removed or enhanced to ensure consistency.

To enhance realism, the final parsed components are **blended with various skin textures** using **alpha blending algorithm**, enabling the creation of composite faces with **natural transitions** between regions.

The output is saved in a **folder-based structure**, where each facial part (eyes, nose, mouth, etc.) is grouped by **category and size**, making it suitable for downstream tasks like **face synthesis** or **dataset generation**.

Additionally, the project includes a simple **graphical user interface (GUI)** built with **Tkinter**, allowing users to manually **compose new faces** by selecting and repositioning facial components. While functional, the GUI contains **minor bugs** in drag behavior and component alignment, which are marked for future refinement.

This project demonstrates a full pipeline from **semantic segmentation** to **interactive composition**, combining **deep learning**, **clustering**, and **image processing techniques** in a **modular and extensible Python framework**.


The system includes:

- 🧠 **U-Net & Attention U-Net Models** – For pixel-wise face segmentation  
- ⚡ **Autoencoder Weight Transfer** – Pretrained encoder used to initialize U-Net weights  
- 🗂️ **CelebAMask-1M Dataset** – Trained over 3 days using an NVIDIA RTX 3070 Ti GPU  
- 🧪 **Post-Processing Pipeline** – Includes removal of unbalanced components and alpha blending with skin textures  
- 📊 **KMeans Clustering** – Categorizes facial parts by size and stores them in structured folders  
- 🖥️ **Tkinter GUI** – Allows users to compose new faces by selecting and moving facial components (contains minor bugs)

---

## ⚙️ Requirements

To run this project, you’ll need:

- 🐍 Python 3.8 or newer  
- 📦 `tensorflow`, `keras`, `opencv-python`, `scikit-learn`, `numpy`, `matplotlib`, `tkinter`  
- 🖼️ CelebAMask-HQ dataset (download separately)  
- 💻 NVIDIA GPU (recommended for training)

---

## 🖼️ Screenshots

Here’s a preview of the GUI-based face composer:


---

## 🛡️ License

This project is licensed under the **MIT License**.  
By contributing, you agree that your contributions will be released under the same license.

---

## ✨ Highlights

- 🧠 Deep segmentation with U-Net and Attention U-Net  
- ⚡ Accelerated training via autoencoder weight reuse  
- 🎨 Alpha blending for realistic skin overlays  
- 📊 KMeans clustering for component organization  
- 🖥️ GUI-based face composition with Tkinter  
- 🧩 Modular codebase for research and experimentation

---

## 📬 Contact

Feel free to reach out if you have questions or feedback!  
Telegram: [@AmirDevil](https://t.me/AmirDevil)

---

## 🚀 Purpose

This project was developed to explore **semantic face segmentation**, **model optimization**, and **interactive composition tools** in Python. It reflects my interest in combining deep learning with creative applications in computer vision and user interface design.
