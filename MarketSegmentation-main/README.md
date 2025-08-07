<h1>Market Segmentation</h1>

<h2>Overview</h2>
<p>The project aims to segment a customer base for targeted marketing campaigns by identifying distinct customer groups. Using machine learning techniques such as K-Means clustering, we analyzed customer data based on various attributes and clustered them into meaningful segments. The application provides insights into customer behaviors and preferences, helping businesses tailor their marketing strategies to improve customer engagement and optimize product offerings.</p>

<h2>Features</h2>
<ul>
  <li><strong>Customer Segmentation:</strong> Uses K-Means clustering to categorize customers into segments based on purchasing behaviors.</li>
  <li><strong>Data Visualization:</strong> Visualizes clusters using graphical representations such as scatter plots.</li>
  <li><strong>Prediction:</strong> Predicts the segment for new customers based on the trained model.</li>
  <li><strong>Model Training:</strong> Allows users to train the K-Means model on a custom dataset.</li>
  <li><strong>Interactive Dashboard:</strong> Displays detailed insights for each customer segment and visualizes the dataset.</li>
</ul>

<h2>Technologies Used</h2>
<ul>
  <li>Python (Flask, Scikit-learn, Matplotlib, Pandas)</li>
  <li>Jupyter Notebooks for model training</li>
  <li>Scikit-learn for building the web application</li>
</ul>

<h2>Project Structure</h2>
<pre>
Market Segmentation
├── static/                    - Static files (CSS, JS, images)
├── templates/                 - HTML templates for web pages
├── model/                     - Trained K-Means model (final_model.sav)
├── app.py                     - Main Flask application
├── requirements.txt           - Python dependencies
├── Clustered_Customer_Data.csv - Customer data with segment labels
├── Customer Data.csv          - Raw customer data
├── kmeans_model.pkl           - K-Means model file
├── train_market.ipynb         - Jupyter notebook for model training
└── README.md                  - Project documentation (this file)
</pre>

<h2>Installation</h2>
<p>Clone the repository:</p>
<pre><code>git clone https://github.com/HarikaCheruku/MarketSegmentation.git
cd MarketSegmentation
</code></pre>
<p>Install the dependencies:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>Model Training</h2>
<p>Before running the application, you need to train the K-Means model. You can do this using Jupyter Notebook:</p>
<pre><code>jupyter notebook train_market.ipynb</code></pre>
<p>This will generate the trained K-Means model file <strong>kmeans_model.pkl</strong>.</p>

<h2>Usage</h2>
<p>Start the Flask application:</p>
<pre><code>python app.py</code></pre>

<h2>Future Improvements</h2>
<ul>
  <li>Integrate more advanced clustering techniques like DBSCAN or hierarchical clustering.</li>
  <li>Expand the application to include customer behavior predictions based on segment attributes.</li>
  <li>Develop a user interface for uploading new datasets and retraining the model.</li>
</ul>

<h2>Contributing</h2>
<p>If you want to contribute to this project, feel free to open issues or submit pull requests. Make sure to follow the contribution guidelines.</p>
