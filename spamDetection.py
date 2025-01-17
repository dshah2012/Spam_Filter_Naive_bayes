import re
import math

class SpamDetection:
	
	def __init__(self):
		'''
			COnstuctor for initialisation of the variables
		'''
		self.hamwords = {}
		self.spamwords = {}
		self.conditional_hamwords = {}
		self.conditional_spamwords = {}
		self.test_prediction = {}
		self.total_vocab = []
		self.prior_prob_ham = 1000/1997
		self.prior_prob_spam = 997/1997
		self.delta = 0.5
		self.model = ""
		self.result = ""
	
	def readFiles(self, fileType, classType, totalFiles):
		'''
			Reading files from the system
			fileType - train
			classType - (Spam or Ham)
			totalFiles - Number of files that is present for each class
		'''
		
		for i in range(1,totalFiles+1):
			fileNumber = classType+'-'+str(i).zfill(5)
			fileName = fileType+'/'+fileType+'-'+fileNumber+ '.txt'
			file =open(fileName, "r",encoding="utf8", errors='ignore')
			if file.mode == 'r':
				contents =file.read()
				file.close()
				contents=re.sub('[^A-Za-z]',' ',contents) #every char except alphabets is replaced
				contents=contents.lower() #converting the cleaned string to lower case
				contents = re.split(' ',contents)
				contents = [i for i in contents if i != ""]
				#Getting the frequency of each word in that email and putting it in dicitonary 
				for i in contents:
					if classType == "ham": 
						if i in self.hamwords.keys():
							self.hamwords[i] = self.hamwords[i]+1 
						else:
							self.hamwords[i] = 1
					else:
						if i in self.spamwords.keys():
							self.spamwords[i] = self.spamwords[i]+1
						else:
							self.spamwords[i] = 1
							
	def save_file(self):
		with open("model.txt", 'w+') as f:
			f.write(self.model)
		f.close()
		
		with open("result.txt", 'w+') as f:
			f.write(self.result)
		f.close()

	def generateModel(self):
		ham_words_length, spam_words_length, vocab, distinct_vocab_list = self.length_utility()
		#print(distinct_vocab_list)
		count=0
		distinct_vocab_list = sorted(distinct_vocab_list)
		for i in distinct_vocab_list:
			count = count+1
			self.model = self.model + str(count) + "  " + i + "  " + str(self.hamwords[i] if i in self.hamwords else 0.5) +"  " + str(self.conditional_hamwords[i] if i in self.conditional_hamwords else 0) + "  " + str(self.spamwords[i] if i in self.spamwords else 0.5) +"  " + str(self.conditional_spamwords[i] if i in self.conditional_spamwords else 0) + "\n"
		#print(self.total_vocab)

	def length_utility(self):
		ham_words_length = sum(list(self.hamwords.values()))
		spam_words_length = sum(list(self.spamwords.values()))
		total_vocab = set(list(self.hamwords.keys()) + list(self.spamwords.keys()))
		total_length = len(total_vocab)
		print("Total Vocab Size : ", total_length)
		vocab = total_length * 0.5 #total vocabulary size
		return ham_words_length,spam_words_length,vocab,total_vocab
	
	def display_result(self, tp, fn, fp, tn):
		'''
			Calculation of the precision, recall, f1measure and accuracy in this method.
		'''
		print("              Predicted")
		print("         Positive  | Negative")
		print("Positive "+str(tp)+ "| "+ str(fn))
		print("Negative "+str(fp)+ "| "+ str(tn))
		print("Results :")
		precision = tp/(tp+fp)
		recall = tp/(tp+fn)
		flscore = (2*precision*recall)/(precision+recall)
		print("Precision : ", precision)
		print("Recall : ", recall)
		print("Fl Score : ", flscore)
		print("Accuracy: ", (tp+tn)/(tp+fn+fp+tn))
		
		
	
	def calculate_results(self, hamType, spamType):
		
		tp_ham = sum([1 if i[0:3]==self.test_prediction[i] and i[0:3]==hamType else 0 for i in self.test_prediction.keys()])
		fn_ham = sum([1 if i[0:3]!=self.test_prediction[i] and i[0:3]==hamType else 0 for i in self.test_prediction.keys()])
		tp_spam = sum([1 if i[0:4]==self.test_prediction[i] and i[0:4]==spamType else 0 for i in self.test_prediction.keys()])
		fn_spam = sum([1 if i[0:4]!=self.test_prediction[i] and i[0:4]==spamType else 0 for i in self.test_prediction.keys()])
		print("-------HAM-----------")
		self.display_result(tp_ham, fn_ham, 0, 0)
		print("---------------------")
		
		print("-------SPAM-----------")	
		self.display_result(tp_spam, fn_spam, 0, 0)
		print("---------------------")
		
		
		print("------ Model results-------")
		tp = sum([1 if i[0:4]==self.test_prediction[i] and i[0:4]==spamType else 0 for i in self.test_prediction.keys()])
		fp = sum([1 if i[0:3]!=self.test_prediction[i] and i[0:3]==hamType else 0 for i in self.test_prediction.keys()])
		tn = sum([1 if i[0:3]==self.test_prediction[i] and i[0:3]==hamType else 0 for i in self.test_prediction.keys()])
		fn = sum([1 if i[0:4]!=self.test_prediction[i] and i[0:4]==spamType else 0 for i in self.test_prediction.keys()])
		self.display_result(tp, fn, fp, tn)
		print("---------------------")
		
		
	

	def calculate_cond_probaility(self):
		'''
			Conditional Probabilities are being calculated. That is (frequency of the word+ smooting)/(total number of words in that class + vocabulary)
		'''
		print("Length of  ham  words are ", len(self.hamwords))
		print("Length of spam words are ", len(self.spamwords))
		ham_words_length,spam_words_length,vocab,vocab_list = self.length_utility()
		for i in vocab_list:
			if i in self.hamwords.keys():
				self.conditional_hamwords[i] = (self.hamwords[i] + self.delta)/(ham_words_length + vocab)
			else:
				self.conditional_hamwords[i] = (self.delta)/(ham_words_length + vocab)
			
			if i in self.spamwords.keys():
				self.conditional_spamwords[i] = (self.spamwords[i] + self.delta)/(spam_words_length + vocab)
			else:
				self.conditional_spamwords[i] = (self.delta)/(spam_words_length + vocab)
			
	def predictTestData(self, fileType, classType, totalFiles, count):
		'''
			Prediction of the test files are being done .
		'''
		ham_words_length,spam_words_length,vocab, vocab_list = self.length_utility()
		
		for i in range(1,totalFiles+1):
			test_dictionary ={}
			fileNumber = classType+'-'+str(i).zfill(5)
			fileName = fileType+'/'+fileType+'-'+ fileNumber + '.txt'
			file = open(fileName, "r",encoding="utf8", errors='ignore')
			ham = math.log10(self.prior_prob_ham)
			spam = math.log10(self.prior_prob_spam)
			if file.mode == 'r':
				contents =file.read()
				file.close()
				contents=re.sub('[^A-Za-z]',' ',contents) #every char except alphabets is replaced
				contents=contents.lower() #converting the cleaned string to lower case
				contents = re.split(' ',contents)
				contents = [i for i in contents if i != ""]
				for i in contents:
					if i in test_dictionary.keys():
						test_dictionary[i] = test_dictionary[i]+1
					else:
						test_dictionary[i] = 1
						
				#print(test_dictionary)
				for i in test_dictionary.keys():
						#Formula for calulating the score is prior probability  * frequency of that word * conditional Probability 
						#If the word was not present earlier then we just do 0.5(smoothing)/(sum of all the frequency words in class + (vocabulary * 0.5))
						if i not in self.conditional_hamwords.keys():
							ham += test_dictionary[i] * math.log10(0.5/(ham_words_length+vocab))
						else:
							ham += test_dictionary[i] * math.log10(self.conditional_hamwords[i])
						
						if i not in self.conditional_spamwords.keys():
							spam +=  test_dictionary[i] * math.log10(0.5/(spam_words_length+vocab))
						else:
							spam +=  test_dictionary[i] * math.log10(self.conditional_spamwords[i])
			
			#print("Ham Score is ", ham)
			#print("Spam score is ", spam)
			#Checking for the prediction here based upon the values
			if ham > spam:
				self.test_prediction [fileNumber] = "ham"
			else:
				self.test_prediction [fileNumber] = "spam"
				
			self.result = self.result + str(count) + "  "+fileType+'-'+ fileNumber + '.txt' +"  "+ self.test_prediction [fileNumber] +"  "+ str(ham) +"  "+ str(spam) + "  "+classType + "  "+("right\n" if classType==self.test_prediction[fileNumber] else "wrong\n")
			count += 1
			
		
		#print(self.test_prediction.values())
					
		
if __name__ == "__main__":
	'''
		Main method 
	'''
	spamDetection = SpamDetection()
	spamDetection.readFiles("train","ham",1000) # Reading the training files from the disk for ham
	spamDetection.readFiles("train","spam",997) # Reading the training files from the disk for spam
	spamDetection.calculate_cond_probaility() # Calculating the conditional probabilities for all words in the vocabulary
	spamDetection.generateModel() # Generating the model.txt contents with the above calculated conditional probabilities
	spamDetection.predictTestData("test","ham",400,1) # Predicting ham test data with the generated model and calculating score for ham ans spam for each test email
	spamDetection.predictTestData("test","spam",400,401) # Predicting spam test data with the generated model and calculating score for ham ans spam for each test email
	spamDetection.save_file() # Saving the results.txt and model.txt files.
	spamDetection.calculate_results("ham", "spam") #Calculating the metrics for accuracy, precision, recall and F-Measure and printing them for each class.