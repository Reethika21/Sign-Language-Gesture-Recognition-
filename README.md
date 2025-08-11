<h1>Sign Language Recognition</h1>

<h2>Overview</h2>
<p>Sign Language Recognition is an innovative project aimed at improving communication between the deaf and hearing communities. This system uses machine learning and computer vision techniques to recognize American Sign Language (ASL) hand gestures and convert them into text. By leveraging deep learning models trained on hand gesture images, this project empowers users to communicate in real-time through a live camera interface. This system is designed for accessibility which makes it easier for everyone to understand the recognized signs.</p>

<p>Despite the advancements in speech-to-text and translation technologies, there is a gap in systems that can bridge communication for people using sign language. The goal of this project is to reduce this communication barrier and provide a reliable platform that can interpret ASL signs from visual inputs. This project also serves as a foundation for future developments in gesture-based communication systems.</p>

<h2>Features</h2>
<ul>
  <li><strong>Real-time Gesture Recognition:</strong> Uses a deep learning model trained to recognize ASL signs from real-time video feed captured through a camera.</li>
  <li><strong>Text Output:</strong> Translates the recognized hand gestures into text.</li>
  <li><strong>Data Collection:</strong> The system includes a data collection script to gather image data for training the model, allowing for custom dataset creation.</li>
  <li><strong>Model Training:</strong> The system allows you to train and test a model for hand gesture recognition using your own dataset.</li>
  <li><strong>Real-time Feedback:</strong> Users can instantly see the translated ASL signs displayed on the screen.</li>
  <li><strong>Interactive Interface:</strong> Developed with Streamlit, the web interface makes it easy for users to interact with the system through a simple UI.</li>
</ul>

<h2>Technologies Used</h2>
<ul>
  <li>Python (TensorFlow/Keras, OpenCV, Streamlit)</li>
  <li>Convolutional Neural Networks (CNNs) for gesture recognition</li>
  <li>Streamlit for creating the interactive web interface</li>
  <li>OpenCV for capturing real-time video and processing hand gestures</li>
  <li>Flask (optional, for deploying a web-based interface if needed)</li>
</ul>

<h2>Project Structure</h2>
<pre>
Sign Language Recognition
├── collect_data.py            - Script to collect hand gesture data for training
├── split.py                   - Splits the collected data into training and testing sets
├── app.py                     - Main Streamlit app that runs the live gesture recognition interface
├── train_test_data/           - Directory containing the collected and split training/testing data
├── model/                     - Trained model for recognizing hand gestures (optional, depending on project state)
├── requirements.txt           - Python dependencies for the project
├── README.md                  - Project documentation (this file)
└── utils/                     - Utility functions for image processing and model handling
</pre>

<h2>Installation</h2>
<p>Clone the repository:</p>
<pre><code>git clone https://github.com/HarikaCheruku/Sign-Language-Recognition.git
cd Sign-Language-Recognition
</code></pre>
<p>Install the dependencies:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>Usage</h2>
<p>To start the project and use the real-time gesture recognition feature, run the following command:</p>
<pre><code>streamlit run app.py</code></pre>

<h2>Data Collection and Model Training</h2>
<p>If you would like to create a custom dataset for training or retrain the model, you can use the <code>collect_data.py</code> script to collect images of hand gestures. The <code>split.py</code> script can be used to divide the dataset into training and testing subsets, which are required for model evaluation and improvement.</p>
<p>Once the dataset is ready, you can train a model by feeding the data into the training pipeline using TensorFlow/Keras, then test it using the <code>train_test_data</code> directory.</p>

<h2>Future Improvements</h2>
<ul>
  <li>Extend the recognition model to recognize full ASL phrases or sentences, rather than just individual letters.</li>
  <li>Improve the accuracy of the gesture recognition system by incorporating more advanced models like pre-trained models (e.g., ResNet, MobileNet).</li>
  <li>Integrate a mobile app version of the system for on-the-go sign language recognition.</li>
  <li>Enhance the system's ability to recognize gestures under varying lighting conditions or different backgrounds.</li>
</ul>

<h2>Contributing</h2>
<p>If you would like to contribute to this project, feel free to open issues or submit pull requests. Please make sure to follow the contribution guidelines and ensure your changes are well-documented. Contributions to improving the dataset, accuracy of the models, and interface design are highly encouraged!</p>       
