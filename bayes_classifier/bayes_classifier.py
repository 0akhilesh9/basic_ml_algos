import csv
import math
from optparse import OptionParser
from collections import defaultdict
from email.parser import Parser

# Objects to store the word feature count
doc_count_dict = defaultdict(int)
word_dict = defaultdict(lambda: defaultdict(int))
# For additional features to detect spam emails
feature_dict = defaultdict(lambda: defaultdict(int))
# The entire vocabulary of the model
vocab = set()
total_doc = 0
train_file = "train"
test_file = "test"
output_file = "output.csv"
# Different combinations to be tried for smoothing parameter
alpha_iterations = 12
# Identified additional features for SPAM detection
features = ["MUST READ - ALERT", "Huge Promo", "REAL company with real products", "BEFORE WE CONTINUE - VERY IMPORTANT", "Impress your girl", "Party Zone", "Best s0ftware prices", "Best software prices", "@solocom.com", "@sunero.com", "@about.com", "@accesspro.net", "@eximeno.com", "only $", "Limited stock", "all sold out","@szu.anet.cz", "@dreamwiz.com", "@fotofutura.com", "special price", "half the price", "half price", "@emb_tunis.intl.tn", "Unbelievable Price", "@wurldlink.net", "@msa.hinet.net"]
email_parser = Parser()

# Read the given file line by line and return the content as a list
def read_file(file_name):
  output = []
  with open(file_name) as file_obj:
    for line in file_obj:
      # add each line to the list
      output.append(line.split(" "))
  return output

# Learn the feature distribution - train the model
def train_model(train_data, feature_flag = False):
  global total_doc
  global doc_count_dict
  global word_dict
  # To use additional features from the actual email corpus
  
  # Not using additional features
  if feature_flag == False:
    # iterate over the training data
    for train_sample in train_data:
      total_doc = total_doc + 1   # Documents count
      email_id = train_sample[0]  # email number
      type = train_sample[1] # Type of email (HAM or SPAM)
      doc_count_dict[type] = doc_count_dict[type] + 1  # Count of documents for each category
      # Iterate over all the words
      for i in range(2, len(train_sample), 2):
        train_word = train_sample[i]
        vocab.add(train_word) # Add word to the main vocabulary
        word_dict[type][train_word] = word_dict[type][train_word] + int(train_sample[i+1])    # Update the frequency count of the word
        
    # Set frequency of unseen words to zero
    for word in vocab:
      for mail_type in word_dict.keys():
        word_dict[mail_type][word] = word_dict[mail_type][word] + 0
  # Use additional features
  else:
  # iterate over the training data
    for train_sample in train_data:
      total_doc = total_doc + 1   # Documents count
      email_id = train_sample[0]  # email number
      type = train_sample[1]    # Type of email (HAM or SPAM)
      doc_count_dict[type] = doc_count_dict[type] + 1   # Count of documents for each category
      # Iterate over all the words
      for i in range(2, len(train_sample), 2):
        train_word = train_sample[i]
        vocab.add(train_word) # Add word to the main vocabulary
        word_dict[type][train_word] = word_dict[type][train_word] + int(train_sample[i+1])    # Update the frequency count of the word
      # Extract additional features from actual email data
      body = ""
      subject = ""
      from_ = ""
      filepath = 'trec/data' + train_sample[0]    # filepath for the actual email file
      email = ""
      with open(filepath, 'r') as file_obj:
        email = email_parser.parse(file_obj)    # parse the email data
      if email.is_multipart():
        for part in email.walk():
          body = body + str(part.get_payload(decode=False))   # extract mail body
          subject = subject + str(email.get('Subject'))  # extract subject
          from_ = from_ + str(email.get('From'))   # extract sender email address
      else:
        body = str(email.get_payload(decode=False))  # extract mail body
        subject = str(email.get('Subject'))    # extract subject
        from_ = str(email.get('From'))    # extract sender email address
      
      if body==None:
        body = ""
      if subject==None:
        subject = ""
      if from_==None:
        from_ = ""
      
      # Combine the extracted feature information
      mail_text = (body + subject + from_).lower().replace(" ","") 
      # iterate over the selected feature set
      for feature_val in features:
        vocab.add(feature_val)    # add feature to the main vocabulary
        word_dict[type][feature_val] = word_dict[type][feature_val] + mail_text.count(feature_val.lower().replace(" ",""))    # Update frequency of the feature
    # Set frequency of unseen words to zero
    for word in vocab:
      for mail_type in word_dict.keys():
        word_dict[mail_type][word] = word_dict[mail_type][word] + 0
      
# Basic email text classification
def classify(test_sample, alpha):
  global vocab
  global word_dict
  global doc_count_dict
  
  result = []
  # iterate over different mail types
  for mail_type in doc_count_dict.keys():
    # word_count = len(word_dict[mail_type].keys())
    
    word_count = 0
    # word count of this mail category
    for key in word_dict[mail_type].keys():
      word_count = word_count + word_dict[mail_type][key]
    
    log_cond_prob_val = 0.0
    # Loop over the words
    for i in range(2, len(test_sample), 2):
      test_word = test_sample[i]
      # If the word is present in the main vocabulary
      if test_word in vocab:
      # Laplacian smoothing is used to prevent the probability of classification from becoming zero for words not present in the vocabulary
        # log_cond_prob_val = log_cond_prob_val + math.log10((word_dict[mail_type][test_word] + alpha)/(float)(word_count + (alpha*len(vocab))))
        tmp_var = int(test_sample[i+1]) * (word_dict[mail_type][test_word] + alpha)/float(word_count + (alpha*len(vocab)))
      # If the word is not present in the vocabulary
      else:
        #log_cond_prob_val = log_cond_prob_val + math.log10(alpha/(float)(word_count + (alpha*len(vocab))))
        tmp_var = alpha/(float)(word_count + (alpha*len(vocab)))
    
      # Calculate the probability value
      if tmp_var == 0:
        log_cond_prob_val = log_cond_prob_val + 0.0
      else:
        log_cond_prob_val = log_cond_prob_val + math.log10(tmp_var)
    # Prior
    log_prior_prob_val = math.log10(word_count / (float)(len(vocab)))
    # Final probability value
    log_prob_val = log_cond_prob_val + log_prior_prob_val
    # add to a list
    result.append([mail_type, log_prob_val])
  # sort the results list and send the classification that has highest classification probability
  result.sort(key = lambda x : x[1], reverse=True)
  return result[0]
      
# Perform classification using additional features
def classify_more_features(test_sample, alpha):
  global vocab
  global word_dict
  global doc_count_dict
  
  result = []
  for mail_type in doc_count_dict.keys():
    # word_count = len(word_dict[mail_type].keys())
    body = ""
    subject = ""
    from_ = ""
    filepath = 'trec/data/' + test_sample[0]
    email = ""
    with open(filepath, 'r') as file_obj:
      email = email_parser.parse(file_obj)
    if email.is_multipart():
      
      for part in email.walk():
        body = body + str(part.get_payload(decode=False))
        subject = subject + str(email.get('Subject'))
        from_ = from_ + str(email.get('From'))
    else:
      body = str(email.get_payload(decode=False))
      subject = str(email.get('Subject'))
      from_ = str(email.get('From'))
    
    if body==None:
      body = ""
    if subject==None:
      subject = ""
    if from_==None:
      from_ = ""
    
    mail_text = (body + subject + from_).lower().replace(" ","")
    
    word_count = 0
    for key in word_dict[mail_type].keys():
      word_count = word_count + word_dict[mail_type][key]
    
    log_cond_prob_val = 0.0
    for i in range(2, len(test_sample), 2):
      test_word = test_sample[i]
      if test_word in vocab:
        # log_cond_prob_val = log_cond_prob_val + math.log10((word_dict[mail_type][test_word] + alpha)/(float)(word_count + (alpha*len(vocab))))
        #log_cond_prob_val = log_cond_prob_val + math.log10((word_dict[mail_type][test_word] + alpha)/float(word_count + (alpha*len(vocab))))
        tmp_var = int(test_sample[i+1]) * (word_dict[mail_type][test_word] + alpha)/float(word_count + (alpha*len(vocab)))
      else:
        #log_cond_prob_val = log_cond_prob_val + math.log10(alpha/(float)(word_count + (alpha*len(vocab))))
        tmp_var = alpha/(float)(word_count + (alpha*len(vocab)))
      
      if tmp_var == 0:
        log_cond_prob_val = log_cond_prob_val + 0.0
      else:
        log_cond_prob_val = log_cond_prob_val + math.log10(tmp_var)
    
    for feature_val in features:
      tmp_var = mail_text.count(feature_val.lower().replace(" ","")) * (word_dict[mail_type][feature_val] + alpha)/float(word_count + (alpha*len(vocab)))
    
      if tmp_var == 0:
          log_cond_prob_val = log_cond_prob_val + 0.0
      else:
        log_cond_prob_val = log_cond_prob_val + math.log10(tmp_var)
    
    log_prior_prob_val = math.log10(word_count / (float)(len(vocab)))
    
    log_prob_val = log_cond_prob_val + log_prior_prob_val
    
    result.append([mail_type, log_prob_val])
  
  result.sort(key = lambda x : x[1], reverse=True)
  return result[0]

import pandas as pd
columns = ["Alpha", "Incorrect HAM", "Correct HAM", "Incorrect SPAM", "Correct SPAM", "Precision", "Recall", "F-Measure"]
a=pd.DataFrame(columns=columns)  
def update(alpha, incorrectHam, incorrectSpam, correctHam, correctSpam):
  global a
  precision = float(correctSpam/float(correctSpam+incorrectSpam)) * 100
  recall = float(correctSpam/float(correctSpam+incorrectHam)) * 100
  fmeasure = (2*precision*recall)/(precision+recall) # harmonic mean of precision and recall
  a = a.append(pd.DataFrame([[alpha, incorrectHam, correctHam, incorrectSpam, correctSpam, precision, recall, fmeasure]],columns=columns))
  
      
if __name__ == "__main__":
  usageStr = """
      USAGE:      python q2_classifier.py --f1 <train_dataset> --f2 <test_dataset> --o <output_file>
      EXAMPLE:    python q2_classifier.py --f1 train --f2 test --o output.csv
      """
  parser = OptionParser(usageStr)

  parser.add_option("--f1", type=str, dest="train_file", default=train_file, help="Train dataset")
  parser.add_option("--f2", type=str, dest="test_file", default=test_file, help="Test dataset")
  parser.add_option("--o", type=str, dest="output_file", default=output_file, help="Output file")
  parser.add_option("--feature_flag", action="store_true", dest="feature_flag", default=False, help="Consider additional features")

  options, otherjunk = parser.parse_args()

  train_file = options.train_file
  test_file = options.test_file
  output_file = options.output_file

  train_data = read_file(train_file)
  train_model(train_data, options.feature_flag)
  
  test_data = read_file(test_file)
  
  for alpha in range(alpha_iterations):
    correct_count = 0.0
    final_output = []
    for mail in test_data:
      if options.feature_flag:
        tmp_result = classify_more_features(mail, alpha)
      else:
        tmp_result = classify(mail, alpha)
      final_output.append([mail[0], tmp_result[0], mail[1]])
      if tmp_result[0] == mail[1]:
        correct_count = correct_count + 1
        
    correctSpam = 0
    correctHam = 0
    incorrectSpam = 0
    incorrectHam = 0

    for i in range(len(test_data)):
      if (final_output[i][1] == "spam"):
        if(final_output[i][2] == "spam"):
            correctSpam += 1
            final_output[i] = final_output[i][:2]
        elif(final_output[i][2] == "ham"):
            incorrectSpam += 1
            final_output[i] = final_output[i][:2]

      elif (final_output[i][1] == "ham"):
        if(final_output[i][2] == "ham"):
            correctHam += 1
            final_output[i] = final_output[i][:2]
        elif (final_output[i][2] == "spam"):
            incorrectHam += 1
            final_output[i] = final_output[i][:2]
    
    # print("Incorrect HAM: %d"%incorrectHam)
    # print("Incorrect SPAM: %d"%incorrectSpam)
    # print("Correct HAM: %d"%correctHam)
    # print("Correct SPAM: %d"%correctSpam)
    
    precision = float(correctSpam/float(correctSpam+incorrectSpam)) * 100
    recall = float(correctSpam/float(correctSpam+incorrectHam)) * 100
    fmeasure = (2*precision*recall)/(precision+recall) # harmonic mean of precision and recall
    
    update(alpha, incorrectHam, incorrectSpam, correctHam, correctSpam)
    
    # print("Alpha value: " + str(alpha) + ", Precision: " + str(precision)+"," + " Recall: " + str(recall)+"," + " F-Measure: "+ str(fmeasure))
  #print(a)    
  
  #a.to_csv("feature.csv")
  
  with open(output_file, 'wb') as file_obj:
    wr = csv.writer(file_obj, quoting=csv.QUOTE_ALL)
    wr.writerows(final_output)