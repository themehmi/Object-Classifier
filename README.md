# Object-Classifier
This project, **VISION.PRO**, is an end-to-end computer vision application designed to identify objects in images using deep learning. It bridges the gap between complex neural network processing and a high-end, minimalist user experience.

Here is a breakdown of the core components and the technology stack driving the project.

### **1. The Core Engine (Machine Learning)**
The heart of the project is a **Convolutional Neural Network (CNN)** trained on the **CIFAR-10 dataset**. 
*   **Classification Power:** The model can recognize 10 distinct categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
*   **Data Processing:** When you upload an image (JPG, PNG, or WEBP), the system automatically handles the "alpha channel" (transparency), converts it to RGB, and resizes it to a $32 \times 32$ pixel grid—the exact resolution required by the CIFAR-10 architecture.
*   **Mathematical Inference:** The model outputs raw scores (logits) which are passed through a **Softmax function** to produce a probability distribution across all classes.



### **2. The Tech Stack**
The project is built using a modern, efficient stack designed for rapid deployment of AI tools:
*   **TensorFlow/Keras:** Used for loading the pre-trained `.h5` model and performing real-time inference.
*   **Streamlit:** Acts as the backend server and hosting framework. It manages the "Session State," allowing the app to remember if an image has been processed or needs to be reset.
*   **Pillow (PIL):** Handles the image manipulation, ensuring that high-resolution uploads are correctly formatted for the neural network.

### **3. The User Interface (UX/UI)**
The design philosophy is "Glassmorphism," a trend that uses transparency and blurred backgrounds to create a sense of depth and modernism.
*   **Interactive Scanning:** To provide visual feedback during the "black box" process of neural inference, a CSS-animated scanner bar travels across the image area once a source is provided.
*   **Dynamic Response:** The UI is reactive. It transitions from a "Waiting" state to a "Results" state, highlighting the **Master Prediction** (the label with the highest confidence) while still showing the top three possibilities in a grid.
*   **Zero-Footprint Customization:** Unlike standard data science apps, this project uses deep CSS injection to hide the default framework (Streamlit) and present a bespoke professional tool.



### **4. Key Features**
*   **Multi-Format Support:** Works with standard `JPG`, transparent `PNG`, and modern high-compression `WEBP` files.
*   **High Precision:** Displays confidence scores rounded to two decimal places to show the model's certainty.
*   **Seamless Reset:** Includes an integrated cylindrical "Try Another" button that clears the memory and prepares the engine for a new scan without requiring a page refresh.

This project serves as a bridge for developers moving from traditional web frameworks to AI-native deployment tools, focusing on how a technical model can be presented as a polished, consumer-ready product.
