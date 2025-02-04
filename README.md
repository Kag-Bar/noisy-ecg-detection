# noisy-ecg-detection

# 1. Data

## 1.a. Differences in the Datasets:
While the **CINC 2017** dataset used identical training and test conditions, the **CINC 2011** dataset differed not only in structure but also in the way it was acquired—whether in terms of the number of leads or sampling frequency. To align these datasets, two key adjustments were made:

1. **Resampling the CINC 2011 data from 300Hz to match CINC 2017:**  
   The choice to downsample rather than interpolate was made to maintain data integrity. Since the signals in **CINC 2017** effectively captured high-frequency noise components, the selected frequency was deemed sufficient for this task.

2. **Lead selection:**  
   According to the official documentation, the **CINC 2017** dataset was collected using a **Lead I** configuration.

   ![laed1](https://github.com/user-attachments/assets/e67d2bf2-5523-42c9-8b9b-40685196d519)

   To ensure consistency, only **Lead I** was used from the **CINC 2011** dataset as well.

---

## 1.b. Data Distribution:
The distribution of noisy signals varied significantly across datasets:

- **CINC 2011**: 24.8% of the signals were labeled as noisy.  
- **CINC 2017 (Test set)**: 10% of the signals were noisy.  
- **CINC 2017 (Training set)**: Only 0.5% were noisy.  

![CINC2017_Data_distribution](https://github.com/user-attachments/assets/0b8f573b-e39d-4fbc-a850-8ac94638bfc8)

   
As a result, the overall training dataset contained **only 2.8% noisy samples**, creating a **highly imbalanced challenge**. To address this, multiple **data balancing techniques** were applied.

---

## 1.c. Data Balancing Methods:

### **1. Data Augmentation**
Given that the **shortest** signal length was around **2700 samples**, the standard input size for the final model was set to **3000 samples**. However, many positive (noisy) signals contained **twice or three times** this length.  
To maximize the use of these longer signals:
- Each noisy signal was **split into as many 3000-sample segments as possible**.
- These segments were **only added to the training set if they originated from noisy signals**.
- **Data leakage prevention**: Signals from the same recording were ensured to be in the **same set (training/validation)** to avoid overfitting.

   ![pre_augmentation](https://github.com/user-attachments/assets/05658d4a-cd65-4063-97b8-d3d0b9241644)


This method **improved the noisy-to-clean ratio to 3.6%**, but it was still not sufficient.

### **2. Undersampling Clean Signals**
To further balance the dataset, **random undersampling** was performed on non-noisy signals.  
- Several target noise-to-clean ratios were tested (**between 0.2 - 0.4**).
- Sampling was performed **separately for each fold** in **cross-validation**.
- On average, the distribution was as follows:  

   - **Right:** Ratio of 0.2  
   - **Left:** Ratio of 0.4  

   ![post_augmentation](https://github.com/user-attachments/assets/cc35f40d-3eab-4f44-99f0-e46bb9adac8a)

---

## 1.d. Data Processing:

Before being fed into the models, all signals underwent **several transformations**:

### **1. Raw Signal Sampling**
A set of raw ECG signals was sampled:

   ![raw_samples](https://github.com/user-attachments/assets/d6769fc2-abfb-457c-af58-357bf534070e)


Key observations:
- **Noisy signals (left):** Highly variable, ranging from smooth and patternless to structured signals resembling clean ECGs.
- **Clean signals (right):** Also highly diverse, with some appearing similar to noisy signals—likely due to **arrhythmias**.
- **Challenge for classification:** Differentiating **noisy signals** from clean signals with **irregular heart rhythms** is difficult.


### **2. Signal Filtering**
To enhance signal clarity, **two different filters** were tested:

#### **Band-Pass Filter (BPF) 4-40Hz**:
- Designed to capture **key ECG frequencies** while filtering out **unnecessary noise**.
- Initially chosen based on prior academic knowledge.

  ![BPF_data](https://github.com/user-attachments/assets/813ef036-8ce4-4385-ab6b-24fbacbdfe28)


#### **High-Pass Filter (HPF) >4Hz**:
- Aimed at **removing low-frequency baseline drift (wandering baseline)**.
- Helped maintain **uniformity in clean signals** while **preserving noise characteristics** in corrupted signals.

   ![HPF_data](https://github.com/user-attachments/assets/0b9653ab-3f37-435f-b13b-0549815742c7)



### **3. Normalization and Standardization**
- **Two normalization techniques were tested**: **Standard scaling vs. Min-Max scaling (-1 to 1)**.
- **Standard normalization was chosen** for better consistency across different datasets.
- All signals were **truncated/padded** to a uniform **3000-sample length**.
- Despite the filtering methods, **class overlap remained high**, making classification difficult.  For example, one can see that samples **1156063** and **A06194** seems to be misclassified.

- ![norm_and_trimmed_data](https://github.com/user-attachments/assets/9417c44d-9356-48a1-9dc9-e255c2e4d20d)


### **4. Peak Detection and Signal-to-Noise Ratio (SNR) Calculation**
- **Peak detection thresholds were adjusted** to capture **QRS complex peaks**.
- Eventually, a **full peak detection approach was used** (not thresholds inserted not for highet, distance nor frequency), assuming noisy signals would contain **either too many or too few peaks**.

![peaks_detected_data](https://github.com/user-attachments/assets/b5c19f57-11b0-4d7d-a277-3c64245223b1)


However, the **distributions differed slightly**:
- **BPF (Top) vs. HPF (Bottom):**

![peaks_distribution](https://github.com/user-attachments/assets/d1a5a139-172d-4d37-8c4d-33b6b110cedc)


Neither filter fully separated the noisy vs. clean signals.  
Thus, **final model selection was based on validation performance**.


### **Final Dataset Preparation**
- The entire dataset was **split into training/validation** using **5-fold cross-validation**.
- The split maintained **80%-20% training-validation** while:
  - **Preventing data leakage** from the same sample appearing in both sets.
  - **Balancing data** by undersampling clean signals.
- Reported performance metrics were **averaged across all folds**.

---

# 2. Architecture

## 2.a. Initial Model Selection
Initially, a few naive models were tested based on **SNR (Signal-to-Noise Ratio)** and **peak count**. The models selected were:

- **Random Forest (RF):** Due to its excellent separation capabilities, considering that decisions could be made deterministically based on these values.
- **Gaussian Mixture Model (GMM):** Since the data comes from different distributions, a mixture model seemed suitable.
- **Logistic Regression:** Chosen for simplicity, given the straightforward nature of the extracted features.

These models were optimized through a quick **grid search** for ideal parameters but were not fully fine-tuned. However, they provided an initial indication that simple models were insufficient and that additional data features were needed.

![naive_models](https://github.com/user-attachments/assets/1358664a-18ce-49cc-99ee-5e09c8308be3)


Initially, additional features (such as patient age, type of arrhythmia, and recording duration) were not extracted because noisy signals could appear in any segment, regardless of patient metadata. Additionally, statistical features (mean, standard deviation, skewness, etc.) were omitted because the plan was to use **deep learning** to extract relevant representations automatically.

---

## 2.b. Basic Neural Network Approach
After the preliminary results, a **basic neural network** was implemented:
- **5 layers** to maintain simplicity given the relatively small dataset.
- **Dropout & Batch Normalization** to prevent overfitting.
- **Focal Loss** to handle class imbalance.
- **Class-weighted training** to adjust for label distribution.

Despite extensive hyperparameter tuning, the results showed that the network struggled to differentiate noisy signals effectively.

![mlp](https://github.com/user-attachments/assets/61fc708a-dfd6-4b74-9603-12fc37eaf3fa)

---

## 2.c. CNN-Based Approach
Understanding that **ECG signals** have a **spatial dependency** that convolutional neural networks (**CNNs**) can capture better, a **simple and shallow CNN** was implemented to prevent unnecessary complexity. Each signal was converted into a **50x60 image**, which was then processed by a classification network consisting of:
- **3 convolutional layers**
- **2 fully connected layers**

![2D_signals](https://github.com/user-attachments/assets/840739de-24fa-494f-9d02-89ba566c672e)


#### Why CNNs?
- CNNs could capture **structural patterns** within the signals.
- Distinct patterns were observed, reinforcing the effectiveness of CNN-based methods.
- Maintaining a **small kernel size** was crucial to capturing subtle variations in the signal, since distinct patterns can be observed also in the noisy samples.

After appropriate **class weights balancing** and hyperparameter tuning, the **best sensitivity achieved was around 50%**, making it the most effective approach so far-

![CNN](https://github.com/user-attachments/assets/ab8e8be7-f2f9-4738-a3fb-847da418c095)

---

## 2.d. Using Pretrained Models
Since ECG signal classification is challenging due to what seems to be **poor labeling quality** in the dataset, it was decided to utilize **pretrained CNN architectures** trained on high-quality datasets. This approach aimed to:
- Capture **generalizable patterns** such as specific frequency bands and orientations, which must have been learned by a pre-trained netwrok.
- Address the **dataset quality issues**, which were evident in labeling inconsistencies.

#### ResNet18
The first pretrained model tested was **ResNet18**, using:
- **Class-weighted training** for label imbalance.
- **Binary Cross-Entropy (BCE) loss** for binary classification.
- **Comparison between different dataset variations**:
  - Bandpass Filter (BPF) with noise-to-clean ratios of 0.2
  ![BPF_0 2](https://github.com/user-attachments/assets/a00aebea-a00b-4838-9a7d-0fce9bd3148d)

  - Bandpass Filter (BPF) with noise-to-clean ratios of 0.4
![BPF_0 4](https://github.com/user-attachments/assets/3834f65c-c294-4f1a-b030-16e62031f6e2)

  - Highpass Filter (HPF) with a noise-to-clean ratio of 0.4.
![HPF_0 4](https://github.com/user-attachments/assets/7982fef3-46c1-48d8-90a9-7c5834ec9a5a)


- One can see that at the end of the day it is a **trade-off between sensitivity and specificity**, and depending on the nature of the problem and the goal of the solution, I would optimize the algorithm accordingly.
- Based on these results, I decided that the data set that optimally fits the problem is the latest data - HPF and a noise/non-noise ratio of 0.4.
- Further investigations, such as **automatic class-weighting vs. manual weighting** and comparing **ResNet18 vs. AlexNet** (Which is considered a shallower and less complex network that could have improved the model's performance), were planned but were **limited by computational resources (GPU) and time**.

## 2.e. Final Model Choice
Based on **validation performance**, the final model selected was **ResNet18** because:
- CNNs effectively captured **spatial frequency information**.
- **Class weighting and BCE loss** helped improve classification balance.
- The **HPF dataset with a 0.4 noise ratio** provided the best trade-off between performance metrics.

An additional experiment was attempted, which involved **relabeling positive samples** using an **RF model** based on **SNR and peak count**. However, this method **did not yield promising results** and required further refinement, which was beyond the project’s available time constraints.

---
# 3. Performance

## 3.a. Sensitivity vs. Specificity Trade-off  

Given that detecting noisy data is crucial, the chosen model prioritizes **sensitivity over specificity**. The reasoning is that **collecting more data is easier** (considering the 1-lead ECG collector machine) than dealing with **incorrect predictions caused by noisy data**.  
Hence, the final model validation confusion matrix is-


![final_model](https://github.com/user-attachments/assets/f87f1d8f-fe98-483b-9e1a-09e01c13f647)

## 3.b. Key Observations  

- Many errors stem from **data quality issues**, as highlighted in the following images:  

 ![FP_val](https://github.com/user-attachments/assets/03bc8c29-e6db-4f55-b062-981be96f126d)
![FN_val](https://github.com/user-attachments/assets/892effbd-6d7b-4dcf-ab8a-0781a5424c54)


- **ROC Curve of the Model:**  
  ![ROC_final_model](https://github.com/user-attachments/assets/ff62a0c2-2dd9-459a-9676-c39f926b841c)


- **Peak Distribution Across Classes:**  Illustrating the difference between False Positives (FP) and True Positives (TP), which led to the idea of a complementary #peak-based model.
 ![peaks_distribution_in_final_model](https://github.com/user-attachments/assets/d6944602-7475-4b95-a780-5fe5f76a5075)


- **Final Data Distribution for the Model:**  
  ![data_distribution_in_final_model](https://github.com/user-attachments/assets/98a69deb-8df5-4b61-a932-1fb115d44ede)


### Loss Function and Training Stability  

- **12 epochs were chosen** based on general overfitting trends observed across folds.  
- Overfitting typically occurred after **epoch 10**, but due to **GPU limitations** (Google Colab’s 3-day restriction), further training was not possible.  
- The model still exhibited **instability**, indicating **more training could have improved results**.  

![loss](https://github.com/user-attachments/assets/e61926a9-43df-4445-aa15-20aa27c1e3ad)


## 3.c. Test Set Performance  

The final model achieved the following results on the test set:  

- **100% Sensitivity**  
- **87% Specificity**  
- **89% Overall Accuracy**

  ![results_test](https://github.com/user-attachments/assets/5faafa08-dea7-46e2-b5c7-dffb5d54dc74)

Despite the results, **manual inspection of the data** showed that these cases were indeed **challenging signals that could easily be mistaken for noise**.  

![FP_test](https://github.com/user-attachments/assets/ff104ac9-8387-4119-a3d7-358158c29c0c)


---

# 4. Further Discussion

## 4.a. Improving the Current Model

Given time and computational constraints, the following enhancements would be prioritized:

- **Deeper Grid Search** on network hyperparameters.
- **Different Class Weighting Strategies** to refine label balance.
- **Comparison with Additional Models** beyond ResNet18 (such as mentioned AlexNet).
- **Complementary Model Based on Statistical Parameters**, expanding beyond peak count and SNR.
- **Dataset Reliability Analysis** to remove misleading or low-quality samples for a more robust training set.

## 4.b. Handling Signals with Varying Resolutions

Two possible approaches were considered to adapt the system for detecting noisy segments across different resolutions:

#### Approach 1: Adaptive Network Training

Since **ResNet uses AdaptiveAvgPool2d before the final FC layer**, ensuring a fixed-length feature vector (512), an adaptive model could be trained on the final selected model. This would involve:

- Retraining the network **to handle varying resolutions adaptively**.
- Using a **sliding window approach**, inputting different signal segments at the desired resolution and generating predictions for each segment.

#### Approach 2: Removing the Classifier final layer and Using Raw Network Scores

Instead of using a classifier with a sigmoid activation, the **raw network output** would be used. This approach follows these steps:

1. **Sliding Window with Overlapping Segments:**

   - Example: For a 1-minute segment and a target resolution of 2 seconds, input **10-second segments with 2-second steps** (e.g., 0-10s, 2-12s, 4-14s, etc.).

2. **Network Scores for Each Segment:**

   - Each segment receives a score from the network.
   - Each **2-second slice** appears in multiple overlapping segments.

3. **Final Score Calculation:**

   - The **final score for each 2-second slice** is the **average of all overlapping segment scores**.
   - If the score is **above 0.5** (or a threshold fine-tuned based on statistical insights), the segment is classified as **noisy**.

A schematic representation of this algorithm is included below (Top: Model scores for each segment. Bottom: Final scores for the target resolution segments):

![scheme](https://github.com/user-attachments/assets/fc494f44-bc28-439f-b77c-a9abbd007524)

---
# 5. Conclusion

This task was both interesting and challenging given the limitations in time and computational resources. There are several critical next steps:

- **Better data management**, including outlier filtering and additional feature extraction.
- **Refining the model**, as detailed in section 4.a, to optimize its performance further.
- **Fine-tuning the model based on its final application**, balancing sensitivity and specificity to suit real-world use cases more effectively.




