import re

class SpamDetection:
	
	def __init__(self):
		'''
			COnstuctor for initialisation of the variables
		'''
		self.hamwords = {}
		self.spamwords = {}
		self.conditional_hamwords = {}
		self.conditional_spamwords = {}
		self.prior_prob_ham = 1000/1997
		self.prior_prob_spam = 997/1997
		self.delta = 0.5
	
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
				contents=re.sub('[^a-z\s]+',' ',contents,flags=re.IGNORECASE) #every char except alphabets is replaced
				contents=re.sub('(\s+)',' ',contents) #multiple spaces are replaced by single space
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
							
		print("Length of  ham  words are ", len(self.hamwords))
		print("Length of spam words are ", len(self.spamwords))
							
	def calculate_cond_probaility(self):
		'''
			Conditional Probabilities are being calculated. That is (frequency of the word+ smooting)/(total number of words in that class + vocabulary)
		'''
		ham_words_length = sum(list(self.hamwords.values()))
		spam_words_length = sum(list(self.spamwords.values()))
		vocab = (len(self.hamwords) + len(self.spamwords)) * 0.5 #total vocabulary size
		for i in self.hamwords.keys():
			self.conditional_hamwords[i] = (self.hamwords[i] + self.delta)/(ham_words_length + vocab)
		
		for i in self.spamwords.keys():
			self.conditional_spamwords[i] = (self.spamwords[i] + self.delta)/(spam_words_length + vocab)
			
	def predictTestData(self, fileType, classType, totalFiles):
		'''
			Prediction of the test files are being done .
		'''
		ham_words_length = sum(list(self.hamwords.values()))
		spam_words_length = sum(list(self.spamwords.values()))
		vocab = (len(self.hamwords) + len(self.spamwords)) * 0.5
		test_prediction = {}
		for i in range(1,totalFiles+1):
			test_dictionary ={}
			fileNumber = classType+'-'+str(i).zfill(5)
			fileName = fileType+'/'+fileType+'-'+ fileNumber + '.txt'
			file = open(fileName, "r",encoding="utf8", errors='ignore')
			ham = 0
			spam = 0
			if file.mode == 'r':
				contents =file.read()
				file.close()
				contents=re.sub('[^a-z\s]+',' ',contents,flags=re.IGNORECASE) #every char except alphabets is replaced
				contents=re.sub('(\s+)',' ',contents) #multiple spaces are replaced by single space
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
							ham += self.prior_prob_ham * test_dictionary[i] * (0.5/(ham_words_length+vocab))
						else:
							ham += self.prior_prob_ham * test_dictionary[i] *  self.conditional_hamwords[i]
						
						if i not in self.conditional_spamwords.keys():
							spam += self.prior_prob_spam * test_dictionary[i] * (0.5/(spam_words_length+vocab))
						else:
							spam += self.prior_prob_spam * test_dictionary[i] * self.conditional_spamwords[i]
			
			#print("Ham Score is ", ham)
			#print("Spam score is ", spam)
			#Checking for the prediction here based upon the values
			if ham > spam:
				test_prediction [fileNumber] = "ham"
			else:
				test_prediction [fileNumber] = "spam"
			
		print(test_prediction.values())
					
		
if __name__ == "__main__":
	'''
		Main method 
	'''
	spamDetection = SpamDetection()
	spamDetection.readFiles("train","ham",1000)
	spamDetection.readFiles("train","spam",997)
	spamDetection.calculate_cond_probaility()
	#spamDetection.predictTestData("test","ham",10)
	spamDetection.predictTestData("test","spam",20)