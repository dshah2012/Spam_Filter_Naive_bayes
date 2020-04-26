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
		self.precision = []
		self.recall = []
		self.accuracy = []
		self.f1measure = []
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
	
	def display_result(self, tp, fn):
		print("              Predicted")
		print("         Positive  | Negative")
		print("Positive "+str(tp)+ "| "+ str(fn))
		print("Negative 0 | 0")
		print("Results :")
		fp = 0
		precision = tp/(tp+fp)
		recall = tp/(tp+fn)
		flscore = (2*precision*recall)/(precision+recall)
		print("Precision : ", precision)
		print("Recall : ", recall)
		print("Fl Score : ", flscore)
		print("Accuracy: ", (tp)/(tp+fn))
		
		
	
	def calculate_results(self, hamType, spamType):
		
		tp_ham = sum([1 if i[0:3]==self.test_prediction[i] and i[0:3]==hamType else 0 for i in self.test_prediction.keys()])
		fn_ham = sum([1 if i[0:3]!=self.test_prediction[i] and i[0:3]==hamType else 0 for i in self.test_prediction.keys()])
		tp_spam = sum([1 if i[0:4]==self.test_prediction[i] and i[0:4]==spamType else 0 for i in self.test_prediction.keys()])
		fn_spam = sum([1 if i[0:4]!=self.test_prediction[i] and i[0:4]==spamType else 0 for i in self.test_prediction.keys()])
		print("-------HAM-----------")
		self.display_result(tp_ham,fn_ham)
		print("---------------------")
		
		print("-------SPAM-----------")	
		self.display_result(tp_spam,fn_spam)
		print("---------------------")
		
		
		#print("Model results")
		#self.display_result(tp_spam,fn_spam)
		
		#accuracy.append((tp+tn)/(tp+fp+tn+fn))
		
	
	def print_accuracy(self):
		ham = [0,0]
		spam =[0,0]
		for i in self.test_prediction.keys():
			if i[0:3]=="ham":
				if self.test_prediction[i] == "ham":
						ham[0] += 1
				else:	
						ham[1] += 1
			
			if i[0:4]=="spam":
				if self.test_prediction[i] == "spam":
						spam[0] += 1
				else:	
						spam[1] += 1
			
		
		print("Prediction Accuracy for Spam is ", (spam[0]/(spam[0]+spam[1]))*100)
		print("Prediction Accuracy for Ham is ", (ham[0]/(ham[0]+ham[1]))*100)
		
	

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
			
	def predictTestData(self, fileType, classType, totalFiles):
		'''
			Prediction of the test files are being done .
		'''
		ham_words_length,spam_words_length,vocab, vocab_list = self.length_utility()
		
		count=0
		for i in range(1,totalFiles+1):
			count += 1
			test_dictionary ={}
			fileNumber = classType+'-'+str(i).zfill(5)
			fileName = fileType+'/'+fileType+'-'+ fileNumber + '.txt'
			file = open(fileName, "r",encoding="utf8", errors='ignore')
			ham = math.log10(self.prior_prob_ham)
			spam = math.log10(self.prior_prob_spam)
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
			
		
		#print(self.test_prediction.values())
					
		
if __name__ == "__main__":
	'''
		Main method 
	'''
	spamDetection = SpamDetection()
	spamDetection.readFiles("train","ham",1000)
	spamDetection.readFiles("train","spam",997)
	spamDetection.calculate_cond_probaility()
	spamDetection.generateModel()
	spamDetection.predictTestData("test","ham",400)
	spamDetection.predictTestData("test","spam",400)
	spamDetection.save_file()
	spamDetection.calculate_results("ham", "spam")
	spamDetection.print_accuracy()