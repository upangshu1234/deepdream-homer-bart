# 🌀 DeepDream on Homer & Bart Dataset

This project applies **Google's DeepDream algorithm** using the **InceptionV3** model on the [Homer & Bart Dataset](https://www.kaggle.com/datasets/juniorbueno/neural-networks-homer-and-bart-classification/data).  
It enhances patterns in the image to create psychedelic, dream-like visuals.

---

## 📂 Dataset

- **Name:** Homer & Bart Classification Dataset  
- **Source:** [Kaggle Link](https://www.kaggle.com/datasets/juniorbueno/neural-networks-homer-and-bart-classification/data)  
- **Description:** Contains `.bmp` images of Homer and Bart from *The Simpsons*.  
- **Structure:** Images are categorized into folders like `homer_bart_1/` containing files such as `homer26.bmp`.

---

## 🚀 Project Workflow

1. **Load InceptionV3 without top layers** (ImageNet pre-trained weights).  
2. **Select intermediate layers** (`mixed3`, `mixed5`, `mixed8`, `mixed9`) for feature amplification.  
3. **Preprocess the input image** from the dataset.  
4. **Define DeepDream functions**:
   - Error calculation (loss)
   - Gradient ascent step
   - Conversion back to a viewable image
5. **Run DeepDream** to enhance patterns.

---

## 🖥 Requirements

Make sure you have the following installed:

```bash
pip install tensorflow matplotlib numpy
````

If running in Google Colab:

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

---

## 📜 Code Example

```python
# Load the image from extracted dataset
image = tf.keras.preprocessing.image.load_img(
    '/content/homer_bart_1/homer26.bmp',
    target_size=(350, 375)
)

# Convert to array & preprocess
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.keras.applications.inception_v3.preprocess_input(image)

# Run DeepDream
# Running DeepDream with only 1 iteration because my laptop can't handle heavy processing.
# If your system is powerful enough, you can use the original setting:
# run_deep_dream(deep_dream_model, image, 8000, 0.001)
run_deep_dream(deep_dream_model, image, 1, 0.001)
```

---

## ⚡ Notes

* **Performance Tip:**

  * My laptop isn’t powerful enough for high iteration counts, so I use `1` instead of `8000` epochs.
  * If your system can handle it, restore the original:

    ```python
    run_deep_dream(deep_dream_model, image, 8000, 0.001)
    ```
* The generated image will be **psychedelic and abstract**, emphasizing the neural network’s learned features.

---


## 📌 References

* [TensorFlow DeepDream Tutorial](https://www.tensorflow.org/tutorials/generative/deepdream)
* [InceptionV3 Paper](https://arxiv.org/abs/1512.00567)
* [Homer & Bart Dataset](https://www.kaggle.com/datasets/juniorbueno/neural-networks-homer-and-bart-classification/data)

---

## 🏷 License

This project is open-source and available under the [MIT License](LICENSE).
