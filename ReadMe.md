<h2>PICTURE ResNet50V2</h2>

<div><img src="data\pic1.png"alt="Hello" style="width:100%;max-width:640px"></div>

<h2>Note</h2>

if you want use ResNet50V2 . The first , u need data 

📂 Load and Prepare Data:

📥 Read Data: Use OpenCV to read images from a directory.

📏 Resize Images: Resize images to 128x128.

🏷️ Label Encoding: Convert image labels to one-hot encoding.

🧠 Build ResNet50V2 Model:

🏗️ Architecture: Use ResNet50V2 pre-trained on ImageNet, exclude the top, and add Fully Connected (FC) and Dropout layers.

🔒 Freeze Layers: Ensure that the ResNet50V2 layers are not trainable.

📦 Input and Output: Input images of size 128x128x3, output 4 classes (softmax) corresponding to the labels.

⚙️ Compile Model:

🔨 Compile: Use categorical_crossentropy loss function, adam optimizer, and track accuracy.
🔄 Data Augmentation:


🌀 Augmentation: Use ImageDataGenerator to augment training data with rotation, zoom, shift, shear, and horizontal flip.

🏋️ Train Model:

🎬 Fit Model: Train the model with augmented training data using fit and validate on the validation set.

💾 Save Model: Use ModelCheckpoint callback to save the best model weights.

🔍 Evaluate Model:

📊 Evaluate: Evaluate the model on the test data to check accuracy and performance.
