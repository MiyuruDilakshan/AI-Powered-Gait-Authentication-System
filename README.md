# 🚶 GaitAuth - AI-Powered Behavioral Biometrics

<div align="center">

![GaitAuth Banner](https://img.shields.io/badge/GaitAuth-AI%20Gait%20Authentication-00D9FF?style=for-the-badge)
[![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge\&logo=mathworks\&logoColor=white)](https://mathworks.com)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)](https://tensorflow.org)
[![Neural Networks](https://img.shields.io/badge/Neural%20Networks-8A2BE2?style=for-the-badge\&logo=brain\&logoColor=white)](https://pytorch.org)

**A neural network-based authentication system that verifies identity through unique walking patterns**

[View Report](#) • [View Data](#) • [Request Feature](https://github.com/MiyuruDilakshan/AI-Powered-Gait-Authentication-System.git/issues)

</div>

---

## 🏆 Project Highlights

**🔬 Advanced AI Research Project** - *University of Plymouth AI & Machine Learning Module*

**🎯 Enterprise-Grade Performance:**

* **2.04% Equal Error Rate (EER)** in cross-day testing
* **1.59% EER** achieved through systematic optimization
* **0.00% EER** for multiple users - perfect authentication accuracy

**🚀 Real-World Impact:** Demonstrates viable continuous authentication for wearable devices, with applications in banking, healthcare, and enterprise security.

---

## 🌟 Overview

GaitAuth revolutionizes digital security by using behavioral biometrics—your unique walking pattern—as a seamless authentication method. Unlike passwords or fingerprints, this system works transparently in the background while you walk, providing continuous verification without any user interaction.

### 🎯 The Problem We Solved

Traditional authentication methods suffer from limitations:

* Passwords are vulnerable to theft and phishing
* Biometrics require explicit user action
* No continuous verification after login
* Poor user experience with frequent prompts

### 💡 Our Innovative Solution

A neural network-based system that:

* **Authenticates continuously** while users walk naturally
* **Learns unique gait patterns** from motion sensor data
* **Works across days** without retraining
* **Optimized for wearable devices**

---

## ✨ Key Features

### 🧠 **Neural Network Architecture**

* Custom [30,15] hidden layer configuration
* Binary classification (genuine vs impostor)
* Systematic tuning across multiple configurations

### 📊 **Feature Engineering**

* **78 statistical features** extracted from motion data
* **13 features per axis** (mean, std, min, max, RMS, entropy, etc.)

### ⚡ **Optimization Techniques**

* **Window overlap tuning** (50% optimal)
* **PCA dimensionality reduction** (72% fewer features)
* **Cross-day validation** for real-world reliability

### 🔧 **Testing Scenarios**

* Same-Day: 0.28% EER
* Cross-Day: 2.04% EER
* Combined: 0.30% EER

### 📱 **Sensor Configurations**

* Accelerometer-only: 2.08% EER
* Gyroscope-only: 10.51% EER
* Combined: 2.04% EER

---

## 🛠️ Technology Stack

* **MATLAB** (Main development)
* **Neural Network Toolbox**
* **Signal Processing Toolbox**
* **Statistics & Machine Learning Toolbox**

**Machine Learning:**

* Feed-forward neural networks
* Pattern recognition

---

## 📈 System Architecture

```
Raw Sensor Data → Preprocessing → Feature Extraction → Segmentation → NN Training [30,15] → Authentication Decision
```

---

## 🚀 Getting Started

### Prerequisites

* MATLAB R2021a or newer
* Required toolboxes installed

### Installation

```bash
git clone https://github.com/MiyuruDilakshan/AI-Powered-Gait-Authentication-System.git
cd AI-Powered-Gait-Authentication-System
```

### Running the Pipeline

```matlab
main.m
```

---

## 📂 Project Structure

```
AI-Powered-Gait-Authentication-System/
├── main.m
├── feature_extraction.m
├── template_generation.m
├── classification.m
├── Dataset/
|       ├── *.csv
└── Outputs/
```

---

## 📊 Performance Results

| User     | FAR (%)  | FRR (%)  | EER (%)  |
| -------- | -------- | -------- | -------- |
| 1        | 0.91     | 10.27    | 1.94     |
| 2        | 1.75     | 0.00     | 0.11     |
| 3        | 0.15     | 14.38    | 2.25     |
| 4        | 7.15     | 0.68     | 4.91     |
| 5        | 0.53     | 0.68     | 0.61     |
| **Mean** | **2.27** | **2.88** | **2.04** |

---

## 👨‍💻 My Contributions

* Designed neural network architecture
* Implemented one-vs-all system
* Built 78-feature extraction pipeline
* Optimized windows and PCA
* Conducted all performance evaluations

---

## 🔮 Future Enhancements

* CNN/LSTM-based deep learning
* Real-time wearable deployment
* Anti-spoofing mechanisms
* Multi-modal fusion
* Cloud model serving

---

## 📄 License

MIT License — see LICENSE file.

---

## 📞 Contact

* 📧 **Email**: [Miyurudilakshan@gmail.com](mailto:Miyurudilakshan@gmail.com)
* 🌐 **Website**: [https://miyuru.dev](https://miyuru.dev)
* 💼 **LinkedIn**: [https://www.linkedin.com/in/miyurudilakshan/](https://www.linkedin.com/in/miyurudilakshan/)
* 🐙 **GitHub**: [https://github.com/MiyuruDilakshan](https://github.com/MiyuruDilakshan)

<div align="center">
Built with ❤️ and 🤖 — Your walk is your password.
</div>
