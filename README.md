# Spam_Filter_Naive_bayes
>Group – FL-G04
>Gursimran Singh - 40080981
>Aravind Ashoka Reddy - 40103248
>Darshan Dhananjay -  40079241

This is a project to implement Naive Bayes algorithm for Email Spam classification.

#Submitted Files:
1. spamDetection.py - This file contains the following methods to read, train, test, save and evaluate the model:
- readFiles - This method reads the files from the disk.
- save_file - This method reads the files to the disk.
- generateModel - This method saves the model.txt contents into self.model
- length_utility - This method calculates the length and the contents of the vocabulary.
- display_result - This displays the confusion matrix for each class.
- calculate_results - This calculates the false positives, tp, fn etc.
- calculate_cond_probability - This calculates the conditional probability of each word for each class. 
- predictTestData - This method tests the test data by calculating score for ham and spam classes.

# Description to Run the code

- (a) Training and (b) testing.
1. Change the current directory to the project folder.
```
cd /Users/name/PycharmProjects/Spam_Filter_Naive_bayes
```
2. In the terminal paste the following command: This will call all the functions: training, testing, and evaluating the model.
```
python spamDetection.py
```
3. This will run the main method of the code with the following method calls:
   - spamDetection = SpamDetection()
	- spamDetection.readFiles("train","ham",1000) # Reading the training files from the disk for ham 
	- spamDetection.readFiles("train","spam",997) # Reading the training files from the disk for spam 
	- spamDetection.calculate_cond_probaility() # Calculating the conditional probabilities for all words in the vocabulary
	- spamDetection.generateModel() # Generating the model.txt contents with the above calculated conditional probabilities
	- spamDetection.predictTestData("test","ham",400,1) # Predicting ham test data with the generated model and calculating score for ham ans spam for each test email
	- spamDetection.predictTestData("test","spam",400,401) # Predicting spam test data with the generated model and calculating score for ham ans spam for each test email
	- spamDetection.save_file() # Saving the results.txt and model.txt files.
	- spamDetection.calculate_results("ham", "spam") #Calculating the metrics for accuracy, precision, recall and F-Measure and printing them for each class.
	- spamDetection.print_accuracy() # Printing the accuracy for each class.


