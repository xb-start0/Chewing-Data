# Chewing-Data
This project is created by ***Bin Xu*** for paper "**MastiSense: A Multi-Modal Dataset for Chewing Ability Detection with Visual and Bone-Conduction Signals**"
## 📘 ***Abstract***

Studies have shown that oral health is closely related to both physical and mental health, and chewing ability serves as a comprehensive indicator of oral health. Traditional methods for evaluating chewing ability often rely on subjective judgments, inconsistent standards, and complex procedures.The application of AI in the medical field enables efficient and objective evaluation of chewing ability via deep learning algorithms, facilitating oral health assessment. However, the current lack of publicly available datasets hinders research progress in this area. To address this, we've collected a multimodal dataset for chewing ability evaluation from three locations: the Brain Research Center of Anhui University, Wuhan Kane Hospital, and a nursing home in Wuhan.  It covers 75 participants aged 18 to 80 and will be expanded. With RGB video, bone microphone audio and event camera data, it's validated by a preliminary algorithm for efficient and accurate assessment. We'll make it public to support further research on relevant deep learning algorithms.

---

## 💻 ***Set Up***

### Pytorch 2.0 (CUDA 11.8)
Our experimental platform is configured with RTX 3090 GPU (CUDA 11.8), and the code runs in a PyTorch 2.0 environment.
For details on the environment, please refer to the [`requirements.txt`](requirements.txt) file.

**Run the installation command:**
```
pip install -r requirements.txt
```
## 🧪***Data Preprocessing***

> The suitable data for this step is the **Raw Data**. If you choose the Processed Data, you can skip this step.

### 🎞️ Frame-based Camera

If you want to process the data from a frame-based camera, you can run the [`video.py`](utils/video.py) in the in utils.We applied Face Mesh technology to precisely detect facial landmarks, enabling us to accurately isolate and subsequently crop out the mouth area.

```
python shipin.py
```
### ⚡ Event-based Camera
If you want to process the data from a event-based camera, you can run the [`event2frame.py`](utils/event2frame.py) in the in utils.To convert this four-dimensional event stream data into an RGB image, we adopt the method of accumulating polarities on a per-pixel basis.

```
python event2frame.py
```
## 📊 ***Technical Validation***
We use ***3D-ResNet*** to validate the dataset's validity. We used Mean Absolute Error(*MAE*) as the primary evaluation metric.

You need to run the [`main.py`](3D-ResNets/main.py) file in 3D-ResNets for the evaluation of chewing ability on our dataset.

For specific steps and model details, please refer to the provided reference.

## ***References***
* [3D-ResNet](https://github.com/kenshohara/3D-ResNets-PyTorch)
