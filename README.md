# 📘 Sampling Techniques on Imbalanced Dataset – Machine Learning Assignment

This project is completed as part of Predictive Analytics (UCS654) coursework at Thapar Institute of Engineering and Technology in collaboration with Coursera Guided Learning.

The objective of this assignment is to study the importance of sampling techniques for handling imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models.

------------------------------------------------------------

🎯 Objective

In many real-world problems like fraud detection, datasets are highly imbalanced where one class contains significantly more samples than the other. This imbalance causes machine learning models to become biased towards the majority class and perform poorly on the minority class.

The goals of this assignment are:

• Balance the dataset  
• Apply five sampling techniques  
• Train five machine learning models  
• Compare model accuracy  
• Identify which sampling method performs best  

------------------------------------------------------------

📂 Dataset

Dataset Used: Credit Card Fraud Detection Dataset  
File: Creditcard_data.csv  

This dataset contains transaction records classified into:
• Class 0 → Normal transactions  
• Class 1 → Fraud transactions  

The dataset is highly imbalanced, making it ideal for testing sampling techniques.

------------------------------------------------------------

⚙️ Methodology

Step 1 – Data Loading  
The dataset was loaded using pandas and basic inspection was performed.

Step 2 – Train-Test Split  
Data was split into:
• 80% Training data  
• 20% Testing data  

Step 3 – Apply Sampling Techniques  
Five sampling techniques were used to balance the training data:

1. Random Oversampling  
2. Random Undersampling  
3. SMOTE  
4. SMOTE + ENN  
5. SMOTE + Tomek Links  

These methods either increase minority samples or reduce majority samples to create balanced class distribution.

------------------------------------------------------------

🤖 Machine Learning Models Used

Five classifiers were trained on each balanced dataset:

M1 – Logistic Regression  
M2 – Decision Tree  
M3 – K-Nearest Neighbors (KNN)  
M4 – Random Forest  
M5 – Support Vector Machine (SVM)  

------------------------------------------------------------

🧪 Experiments

Each sampling technique was applied to each machine learning model.

Total experiments performed:

5 Sampling Methods × 5 Models = 25 Experiments

For every combination, accuracy was calculated using the test dataset.

------------------------------------------------------------

📊 Results

The results were stored in a comparison table showing accuracy values for every model and sampling technique.

Example format:

                Sampling1  Sampling2  Sampling3  Sampling4  Sampling5
M1 (LR)           xx         xx         xx         xx         xx
M2 (DT)           xx         xx         xx         xx         xx
M3 (KNN)          xx         xx         xx         xx         xx
M4 (RF)           xx         xx         xx         xx         xx
M5 (SVM)          xx         xx         xx         xx         xx

A heatmap visualization was also created to easily compare performance.

------------------------------------------------------------

✅ Observations

• Models trained on imbalanced data performed poorly  
• Sampling improved minority class prediction  
• Oversampling techniques generally performed better than undersampling  
• SMOTE and hybrid techniques gave more stable results  
• Random Forest and SVM achieved higher accuracy compared to other models  

------------------------------------------------------------

🏆 Conclusion

This assignment demonstrates that sampling plays a very important role when working with imbalanced datasets. Balancing the dataset significantly improves model performance and prediction accuracy. Among different techniques, SMOTE-based methods often provided better results.

Therefore, selecting the right sampling strategy is essential for building reliable machine learning models on imbalanced data.

------------------------------------------------------------

🛠️ Technologies Used

• Python  
• Pandas  
• NumPy  
• Scikit-learn  
• Imbalanced-learn  
• Matplotlib  
• Seaborn  

------------------------------------------------------------

📁 Repository Structure

Sampling_Assignment/
│── sampling.ipynb
│── Creditcard_data.csv
│── README.md

------------------------------------------------------------

🚀 How to Run

Install dependencies:

pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib

Then run:

sampling.ipynb

------------------------------------------------------------

👨‍💻 Author

Karanveer Singh Saini  
UCS654 – Predictive Analytics  
Thapar Institute of Engineering and Technology  

------------------------------------------------------------

#TIET  
#ThaparUniversity  
#ThaparOutcomeBasedLearning  
#ThaparCoursera  
#Coursera  
#MachineLearning  
#Sampling  
#ImbalancedLearning  
#PredictiveAnalytics
