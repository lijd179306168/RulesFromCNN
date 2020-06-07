# RulesFromCNN
Requirements:
torchvision==0.5.0
torch==1.4.0
numpy==1.18.1
deap==1.3.1
scikit_learn==0.23.1

Step1, train CNN:
Model definition:	VGGClass.py, AlexnetClass.py, ResnetClass.py
Source file:		CNN_Pre_Train.py
Input:				image classification data
Output:				pre_trained models, such as “vggnet.pkl”
Description:	    train model for 100 epochs with learning_rate=0.1, 0.01,0.001, respectively

Step2, extract features from CNN:
Source file:		CNN_Extract_Features.py
Input:				pre_trained models, such as “vggnet.pkl”
Output:				feature data files, such as “cifar10train_50000_512.csv”, (for the limit of file size, the file only contains 500 examples)
Description:	for batch_idx, (inputs, targets) in enumerate(trainloader):   #change trainloader to testloader for extracting features from validate set

Step3, select filters
Source file:		Select_Filters.py
Input:				feature data files, such as “cifar10train_50000_512.csv”
Output:				array of selected filters, such as: “s= [0, 1, 22, 96, 110, 191, 231, 239, 242, 267, 317, 384, 398, 435]”
Description:	turn parameters activatoinsValue and activatoinsCount for obtaining appropriate filters

Step4, write activations of filters to file
Source file:	Write_Filters.py
Input:			feature data files, such as “cifar10train_50000_512.csv”
				array of selected filters, such as:“s= [0, 1, 22, 96, 110, 191, 231, 239, 242, 267, 317, 384, 398, 435]”
Output:			activations files, such as: “cifar10train_50000_12.csv”
Description:	none

Step5, learn fuzzy rules from activations
Source file:		Triangle.py, Data_CSV.py, Learn_Fuzzy_Rules_GA.py
Input:				activations train files, such as: “cifar10train_50000_12.csv”
Output:				the accuracy of each generation, “train.csv”
					the rules of each generation, “rule.csv”
Description:	none

Step6, validate fuzzy rules 
Source file:		Data_CSV_Valid.py, Valid_Rules.py
Input:				activations files of validate set, such as: “cifar10valid_10000_12.csv”
                    selected fuzzy rules file: such as: “selectrule_12.csv”
Output:				the accuracy of validate set
Description:	none

Step7, simplify fuzzy rules 
Source file:		Resume_Simplify_Rules.py
Input:				activations train files,such as: “cifar10train_50000_12.csv” selected fuzzy rules files: such as: “selectrule_12.csv”
Output:				the accuracy, rulecount, correctcount of train set
Description:	turn ElistPoolSize, fitness function to obtained different trade-offs 
